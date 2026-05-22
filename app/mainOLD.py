from __future__ import annotations

import base64
import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

# =============================
# Configuration
# =============================
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "app" / "static"
RUNS_DIR = BASE_DIR / "runs"
CACHE_DIR = RUNS_DIR / "cache"
PATCH_DIR = RUNS_DIR / "patches"
REPORTS_DIR = RUNS_DIR / "reports"
PROFILES_PATH = BASE_DIR / "profiles.json"

for d in [RUNS_DIR, CACHE_DIR, PATCH_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FIGMA_RENDER_SCALE = float(os.getenv("FIGMA_RENDER_SCALE", "2.0"))
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "4096"))
PATCH_GRID = int(os.getenv("PATCH_GRID", "3"))
MAX_PATCHES = int(os.getenv("MAX_PATCHES", "5"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
PORT = int(os.getenv("PORT", "8000"))

# =============================
# App
# =============================
app = FastAPI(title="Linterface Workshop Auditor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# =============================
# In-memory job store
# =============================
JOB_STORE: Dict[str, Dict[str, Any]] = {}
JOB_LOCK = threading.Lock()


class AuditRequest(BaseModel):
    profile_name: str = Field(..., description="Profile name from profiles.json")
    target_url: str = Field(..., description="Target Figma URL")
    target_page_index: int = 0
    ds_page_index: int = 0


@dataclass
class Profile:
    name: str
    design_system_url: str
    working_file_url: str = ""
    description: str = ""


# =============================
# Utilities
# =============================
def safe_name(name: str) -> str:
    return re.sub(r"[^\w\- ]+", "", name).replace(" ", "_")[:60]


def read_profiles() -> List[Profile]:
    if not PROFILES_PATH.exists():
        sample = [
            {
                "name": "Example Profile",
                "design_system_url": "https://www.figma.com/design/REPLACE_WITH_DESIGN_SYSTEM_FILE_KEY/example-design-system",
                "working_file_url": "https://www.figma.com/design/REPLACE_WITH_TARGET_FILE_KEY/example-target-file",
                "description": "Replace with your Figma file URLs"
            }
        ]
        PROFILES_PATH.write_text(json.dumps(sample, indent=2), encoding="utf-8")

    raw = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    return [Profile(**item) for item in raw]


def get_profile(name: str) -> Profile:
    for profile in read_profiles():
        if profile.name == name:
            return profile
    raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")


def extract_figma_key(url: str) -> str:
    m = re.search(r"figma\.com/(file|design)/([a-zA-Z0-9]+)", url)
    return m.group(2) if m else url.strip()


def figma_get(url: str, token: str, params: Optional[dict] = None) -> dict:
    headers = {"X-Figma-Token": token}
    last_error: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(attempt * 2)
    raise RuntimeError(f"Figma API failed for {url}: {last_error}")


def download_img(url: str, out_path: Path) -> None:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    out_path.write_bytes(response.content)


def ensure_reasonable_size(img_path: Path) -> Path:
    img = Image.open(img_path).convert("RGB")
    if max(img.size) <= MAX_IMAGE_SIDE:
        return img_path
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    img.save(img_path)
    return img_path


def get_ink_score(img: Image.Image) -> float:
    g = img.convert("L")
    px = list(g.getdata())
    dark = sum(1 for v in px if v < 220)
    return dark / (len(px) or 1)


def create_patches(img_path: Path, out_dir: Path, job_id: str) -> List[Tuple[float, Path, tuple]]:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    pw, ph = max(1, w // PATCH_GRID), max(1, h // PATCH_GRID)
    scored_patches: List[Tuple[float, Path, tuple]] = []

    for r in range(PATCH_GRID):
        for c in range(PATCH_GRID):
            left = c * pw
            top = r * ph
            right = w if c == PATCH_GRID - 1 else (c + 1) * pw
            bottom = h if r == PATCH_GRID - 1 else (r + 1) * ph
            box = (left, top, right, bottom)
            patch = img.crop(box)
            score = get_ink_score(patch)
            p_path = out_dir / f"{job_id}_{img_path.stem}_p_{r}_{c}.png"
            patch.save(p_path)
            scored_patches.append((score, p_path, box))

    scored_patches.sort(key=lambda x: x[0], reverse=True)
    return scored_patches[:MAX_PATCHES]


def set_job(job_id: str, **updates: Any) -> None:
    with JOB_LOCK:
        JOB_STORE.setdefault(job_id, {})
        JOB_STORE[job_id].update(updates)


def append_job_log(job_id: str, message: str) -> None:
    with JOB_LOCK:
        JOB_STORE.setdefault(job_id, {})
        JOB_STORE[job_id].setdefault("logs", [])
        JOB_STORE[job_id]["logs"].append(message)


# =============================
# OpenAI vision client
# =============================
class VisionAuditor:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def _prepare_image_data_url(self, path: Path) -> str:
        with Image.open(path) as img:
            if max(img.size) > MAX_IMAGE_SIDE:
                img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
                img.save(path, "PNG", optimize=True)

        b64_data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{b64_data}"

    def run_pass(self, ds_paths: List[Path], target_path: Path, preflight: str) -> str:
        prompt = f"""
You are an elite UI/UX QA engineer.
Audit 'MY DESIGN' against the 'DS REFERENCES'.

Rules:
- Return only a Markdown table.
- Be concrete and actionable.
- Use this table exactly:
| Category | Element | Issue Found | REQUIRED FIX (FROM -> TO) |
| :--- | :--- | :--- | :--- |
- Compare typography, spacing, radius, borders, colors, alignment, and component consistency.
- If you can infer likely design-system token names or values from the reference, mention them.
- If an issue is ambiguous, say what seems visually inconsistent and why.

Context:
{preflight}
""".strip()

        content: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                ],
            }
        ]

        for p in ds_paths:
            content[0]["content"].append(
                {
                    "type": "input_image",
                    "image_url": self._prepare_image_data_url(p),
                    "detail": "high",
                }
            )

        content[0]["content"].append(
            {"type": "input_text", "text": "MY DESIGN UNDER TEST:"}
        )
        content[0]["content"].append(
            {
                "type": "input_image",
                "image_url": self._prepare_image_data_url(target_path),
                "detail": "high",
            }
        )

        payload = {
            "model": self.model,
            "input": content,
            "max_output_tokens": 3500,
            "temperature": 0,
        }

        response = requests.post(
            f"{OPENAI_BASE_URL}/responses",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        if "output_text" in data and data["output_text"]:
            return data["output_text"]

        # Fallback parsing for compatibility
        parts: List[str] = []
        for item in data.get("output", []):
            for piece in item.get("content", []):
                if piece.get("type") in {"output_text", "text"} and piece.get("text"):
                    parts.append(piece["text"])
        if parts:
            return "\n".join(parts)

        raise RuntimeError("OpenAI response did not contain text output.")


# =============================
# Audit pipeline
# =============================
def render_figma_page(figma_token: str, file_key: str, page_index: int, label: str, job_id: str) -> Path:
    file_data = figma_get(f"https://api.figma.com/v1/files/{file_key}", figma_token, {"depth": 1})
    pages = file_data["document"]["children"]
    if not pages:
        raise RuntimeError("Figma file has no pages")
    if page_index < 0 or page_index >= len(pages):
        raise RuntimeError(f"Page index {page_index} is out of range for {label}")

    page = pages[page_index]
    page_id = page["id"]
    page_name = page.get("name", label)
    image_api = f"https://api.figma.com/v1/images/{file_key}"
    image_data = figma_get(
        image_api,
        figma_token,
        {"ids": page_id, "format": "png", "scale": FIGMA_RENDER_SCALE},
    )

    img_url = image_data.get("images", {}).get(page_id)
    if not img_url:
        raise RuntimeError(f"Could not render page '{page_name}' for {label}")

    out_path = CACHE_DIR / f"{job_id}_{safe_name(label)}_{safe_name(page_name)}.png"
    download_img(img_url, out_path)
    ensure_reasonable_size(out_path)
    return out_path


def perform_audit(job_id: str, profile_name: str, target_url: str, target_page_index: int, ds_page_index: int) -> None:
    try:
        set_job(job_id, status="running", started_at=time.time())
        append_job_log(job_id, "Loading profile...")
        profile = get_profile(profile_name)

        figma_token = os.getenv("FIGMA_TOKEN", "").strip()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not figma_token:
            raise RuntimeError("Missing FIGMA_TOKEN environment variable")
        if not openai_key:
            raise RuntimeError("Missing OPENAI_API_KEY environment variable")

        ds_key = extract_figma_key(profile.design_system_url)
        tg_key = extract_figma_key(target_url)

        append_job_log(job_id, "Rendering design-system page from Figma...")
        ds_img = render_figma_page(figma_token, ds_key, ds_page_index, "design_system", job_id)

        append_job_log(job_id, "Rendering target page from Figma...")
        tg_img = render_figma_page(figma_token, tg_key, target_page_index, "target_design", job_id)

        preflight = (
            "Compare the target against the design-system reference. "
            "Focus on typography, spacing, radius, colors, borders, alignment, and component styling."
        )

        auditor = VisionAuditor(openai_key, OPENAI_MODEL)

        append_job_log(job_id, f"Running full-page audit with {OPENAI_MODEL}...")
        full_results = auditor.run_pass([ds_img], tg_img, preflight)

        append_job_log(job_id, "Running patch analysis for dense UI regions...")
        patches = create_patches(tg_img, PATCH_DIR, job_id)
        patch_findings: List[str] = []

        for idx, (score, patch_path, box) in enumerate(patches, start=1):
            append_job_log(job_id, f"Patch {idx}/{len(patches)} | ink score {score:.2f} | area {box}")
            result = auditor.run_pass(
                [ds_img],
                patch_path,
                "Deep zoom pass: inspect micro-typography, alignment, spacing, and border/radius details.",
            )
            patch_findings.append(f"### Area {box}\n{result}\n")

        report = f"""# Figma Design Audit Report

**Date:** {time.ctime()}
**Profile:** {profile.name}
**Model:** {OPENAI_MODEL}
**Target URL:** {target_url}

## Full Page Findings
{full_results}

## Patch Findings
{''.join(patch_findings)}
"""

        report_path = REPORTS_DIR / f"{job_id}.md"
        report_path.write_text(report, encoding="utf-8")

        set_job(
            job_id,
            status="completed",
            finished_at=time.time(),
            report_markdown=report,
            report_path=str(report_path),
            model=OPENAI_MODEL,
        )
        append_job_log(job_id, "Audit completed.")

    except Exception as exc:
        set_job(job_id, status="failed", error=str(exc), finished_at=time.time())
        append_job_log(job_id, f"Failed: {exc}")


# =============================
# Routes
# =============================
@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model": OPENAI_MODEL,
        "has_figma_token": bool(os.getenv("FIGMA_TOKEN")),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.get("/api/profiles")
def list_profiles() -> List[Dict[str, str]]:
    return [asdict(profile) for profile in read_profiles()]


@app.get("/api/runs")
def list_runs() -> List[Dict[str, Any]]:
    with JOB_LOCK:
        runs = []
        for job_id, payload in JOB_STORE.items():
            runs.append(
                {
                    "id": job_id,
                    "status": payload.get("status", "queued"),
                    "started_at": payload.get("started_at"),
                    "finished_at": payload.get("finished_at"),
                    "error": payload.get("error"),
                }
            )
        runs.sort(key=lambda r: r.get("started_at") or 0, reverse=True)
        return runs


@app.post("/api/audit")
def start_audit(req: AuditRequest) -> Dict[str, str]:
    _ = get_profile(req.profile_name)
    job_id = uuid.uuid4().hex[:12]
    set_job(
        job_id,
        status="queued",
        created_at=time.time(),
        profile_name=req.profile_name,
        target_url=req.target_url,
        logs=[],
    )

    thread = threading.Thread(
        target=perform_audit,
        args=(job_id, req.profile_name, req.target_url, req.target_page_index, req.ds_page_index),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id}


@app.get("/api/audit/{job_id}")
def get_audit(job_id: str) -> Dict[str, Any]:
    with JOB_LOCK:
        if job_id not in JOB_STORE:
            raise HTTPException(status_code=404, detail="Run not found")
        return JOB_STORE[job_id]


@app.get("/api/audit/{job_id}/report")
def get_report(job_id: str) -> Dict[str, Any]:
    with JOB_LOCK:
        if job_id not in JOB_STORE:
            raise HTTPException(status_code=404, detail="Run not found")
        payload = JOB_STORE[job_id]
        if payload.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Report not ready")
        return {
            "job_id": job_id,
            "markdown": payload.get("report_markdown", ""),
            "model": payload.get("model"),
        }
