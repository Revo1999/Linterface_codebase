from __future__ import annotations

import base64
import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, asdict, field
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

# how deep to fetch Figma JSON for preflight
FIGMA_PREFLIGHT_DEPTH = int(os.getenv("FIGMA_PREFLIGHT_DEPTH", "8"))

# matching tolerances
COLOR_DISTANCE_THRESHOLD = float(os.getenv("COLOR_DISTANCE_THRESHOLD", "0.06"))
RADIUS_TOLERANCE = float(os.getenv("RADIUS_TOLERANCE", "1.0"))
SPACING_TOLERANCE = float(os.getenv("SPACING_TOLERANCE", "2.0"))
FONT_SIZE_TOLERANCE = float(os.getenv("FONT_SIZE_TOLERANCE", "1.0"))
LINE_HEIGHT_TOLERANCE = float(os.getenv("LINE_HEIGHT_TOLERANCE", "2.0"))

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
# Preflight models
# =============================
@dataclass
class PreflightIssue:
    severity: str
    category: str
    node_id: str
    node_name: str
    property_name: str
    actual: Any
    expected: Any
    hint: str
    bbox: Optional[Tuple[float, float, float, float]] = None
    source: str = "Preflight"

    def to_prompt_line(self) -> str:
        bbox_txt = f" | bbox={tuple(round(v, 1) for v in self.bbox)}" if self.bbox else ""
        return (
            f"- [{self.severity.upper()}] {self.category} | "
            f"node='{self.node_name}' ({self.node_id}) | "
            f"{self.property_name}: actual={self.actual} | expected={self.expected} | "
            f"hint={self.hint}{bbox_txt}"
        )


@dataclass
class PreflightReport:
    summary: Dict[str, Any]
    issues: List[PreflightIssue]
    target_nodes_scanned: int
    ds_roles_found: List[str]

    def to_prompt_text(self, max_issues: int = 25) -> str:
        lines: List[str] = []
        lines.append("Deterministic preflight summary:")
        for k, v in self.summary.items():
            lines.append(f"- {k}: {v}")
        if self.ds_roles_found:
            lines.append(f"- ds_roles_found: {', '.join(self.ds_roles_found[:20])}")

        lines.append("")
        lines.append("Top deterministic preflight issues:")
        if not self.issues:
            lines.append("- No structural issues were detected by the preflight heuristics.")
        else:
            for issue in self.issues[:max_issues]:
                lines.append(issue.to_prompt_line())

        lines.append("")
        lines.append("Use the findings above as targeted hints, but verify them visually.")
        return "\n".join(lines)

    def to_markdown(self, max_issues: int = 200) -> str:
        out = []
        out.append("## Deterministic Preflight Summary")
        for k, v in self.summary.items():
            out.append(f"- **{k}**: {v}")
        if self.ds_roles_found:
            out.append(f"- **ds_roles_found**: {', '.join(self.ds_roles_found[:30])}")

        out.append("")
        out.append("## Deterministic Preflight Issues")
        if not self.issues:
            out.append("No preflight issues detected.")
        else:
            out.append("| Severity | Category | Node | Property | Actual | Expected | Hint |")
            out.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for issue in self.issues[:max_issues]:
                node_label = f"{issue.node_name} ({issue.node_id})"
                out.append(
                    f"| {issue.severity} | {issue.category} | {node_label} | "
                    f"{issue.property_name} | {issue.actual} | {issue.expected} | {issue.hint} |"
                )
        out.append("")
        return "\n".join(out)


@dataclass
class DSRoleSignature:
    role: str
    sample_node_name: str
    fills: List[Dict[str, Any]] = field(default_factory=list)
    text_style_ids: List[str] = field(default_factory=list)
    font_sizes: List[float] = field(default_factory=list)
    line_heights: List[float] = field(default_factory=list)
    corner_radii: List[float] = field(default_factory=list)
    item_spacings: List[float] = field(default_factory=list)
    paddings: List[Tuple[float, float, float, float]] = field(default_factory=list)

    def expected_fill_has_variable(self) -> bool:
        return any(bool(f.get("has_bound_variable")) for f in self.fills)

    def representative_fill(self) -> Optional[Dict[str, Any]]:
        if not self.fills:
            return None
        bound = [f for f in self.fills if f.get("has_bound_variable")]
        if bound:
            return bound[0]
        return self.fills[0]

    def representative_text_style_id(self) -> Optional[str]:
        return self.text_style_ids[0] if self.text_style_ids else None

    def representative_font_size(self) -> Optional[float]:
        return median(self.font_sizes)

    def representative_line_height(self) -> Optional[float]:
        return median(self.line_heights)

    def representative_corner_radius(self) -> Optional[float]:
        return median(self.corner_radii)

    def representative_item_spacing(self) -> Optional[float]:
        return median(self.item_spacings)

    def representative_padding(self) -> Optional[Tuple[float, float, float, float]]:
        if not self.paddings:
            return None
        lefts = [p[0] for p in self.paddings]
        rights = [p[1] for p in self.paddings]
        tops = [p[2] for p in self.paddings]
        bottoms = [p[3] for p in self.paddings]
        return (median(lefts), median(rights), median(tops), median(bottoms))


@dataclass
class DSRegistry:
    role_signatures: Dict[str, DSRoleSignature]
    all_fill_samples: List[Dict[str, Any]]
    known_text_style_ids: List[str]


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
                "description": "Replace with your Figma file URLs",
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


def set_job(job_id: str, **updates: Any) -> None:
    with JOB_LOCK:
        JOB_STORE.setdefault(job_id, {})
        JOB_STORE[job_id].update(updates)


def append_job_log(job_id: str, message: str) -> None:
    with JOB_LOCK:
        JOB_STORE.setdefault(job_id, {})
        JOB_STORE[job_id].setdefault("logs", [])
        JOB_STORE[job_id]["logs"].append(message)


def append_job_progress(job_id: Optional[str], message: str) -> None:
    if not job_id:
        return
    append_job_log(job_id, message)


def median(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    vals = sorted(vals)
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return float(vals[mid])
    return float(vals[mid - 1] + vals[mid]) / 2.0


def normalize_name(name: str) -> str:
    name = (name or "").strip().lower()
    name = re.sub(r"[/_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name


def infer_role(sig: Dict[str, Any]) -> str:
    n = normalize_name(sig.get("name", ""))
    t = sig.get("type", "")

    if t == "TEXT":
        if any(k in n for k in ["headline", "heading", "hero title", "title"]):
            return "heading"
        if "label" in n:
            return "label"
        if any(k in n for k in ["caption", "helper", "supporting"]):
            return "caption"
        return "text"

    if any(k in n for k in ["button", "cta", "primary action", "secondary action"]):
        return "button"
    if any(k in n for k in ["input", "field", "textbox", "text field"]):
        return "input"
    if "card" in n:
        return "card"
    if any(k in n for k in ["chip", "pill", "tag"]):
        return "chip"
    if any(k in n for k in ["nav", "tab", "menu item"]):
        return "navigation"
    if any(k in n for k in ["modal", "dialog"]):
        return "modal"

    if t in {"FRAME", "GROUP", "INSTANCE", "COMPONENT", "COMPONENT_SET"}:
        return "container"

    return (t or "unknown").lower()


def figma_bbox_to_tuple(bbox: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float, float, float]]:
    if not bbox:
        return None
    x = float(bbox.get("x", 0))
    y = float(bbox.get("y", 0))
    w = float(bbox.get("width", 0))
    h = float(bbox.get("height", 0))
    return (x, y, x + w, y + h)


def bbox_intersects(a: Optional[Tuple[float, float, float, float]], b: Optional[Tuple[float, float, float, float]]) -> bool:
    if not a or not b:
        return False
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def walk_nodes(node: dict, out: Optional[List[dict]] = None) -> List[dict]:
    if out is None:
        out = []
    out.append(node)
    for child in node.get("children", []) or []:
        walk_nodes(child, out)
    return out


def get_page_node(file_data: dict, page_index: int) -> dict:
    pages = file_data["document"]["children"]
    if not pages:
        raise RuntimeError("Figma file has no pages")
    if page_index < 0 or page_index >= len(pages):
        raise RuntimeError(f"Page index {page_index} out of range")
    return pages[page_index]


def figma_get(
    url: str,
    token: str,
    params: Optional[dict] = None,
    job_id: Optional[str] = None,
    context: str = "Figma request",
) -> dict:
    headers = {"X-Figma-Token": token}
    last_error: Optional[Exception] = None

    for attempt in range(1, 8):
        try:
            append_job_progress(job_id, f"{context} (attempt {attempt}/7)...")

            response = requests.get(url, headers=headers, params=params, timeout=120)

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                plan_tier = response.headers.get("X-Figma-Plan-Tier")
                rate_type = response.headers.get("X-Figma-Rate-Limit-Type")

                wait_s = 1
                try:
                    if retry_after is not None:
                        wait_s = max(1, int(float(retry_after)))
                except Exception:
                    wait_s = min(60, 2 ** attempt)

                append_job_progress(
                    job_id,
                    (
                        f"Waiting for Figma rate limit reset: {wait_s}s "
                        f"(attempt {attempt}/7"
                        f"{f', plan={plan_tier}' if plan_tier else ''}"
                        f"{f', type={rate_type}' if rate_type else ''})"
                    ),
                )

                last_error = RuntimeError(
                    f"429 Too Many Requests for {url} | "
                    f"retry_after={wait_s}s | plan_tier={plan_tier} | rate_type={rate_type}"
                )
                time.sleep(wait_s)

                append_job_progress(job_id, "Retrying Figma request after rate limit wait...")
                continue

            response.raise_for_status()
            return response.json()

        except requests.HTTPError as exc:
            last_error = exc
            status = getattr(exc.response, "status_code", None)

            if status in {500, 502, 503, 504} and attempt < 7:
                wait_s = min(30, 2 ** attempt)
                append_job_progress(
                    job_id,
                    f"Figma temporary server error ({status}). Waiting {wait_s}s before retry...",
                )
                time.sleep(wait_s)
                continue

            raise RuntimeError(f"Figma API failed for {url}: {exc}") from exc

        except requests.RequestException as exc:
            last_error = exc
            if attempt < 7:
                wait_s = min(30, 2 ** attempt)
                append_job_progress(
                    job_id,
                    f"Figma network issue: {exc}. Waiting {wait_s}s before retry...",
                )
                time.sleep(wait_s)
                continue
            raise RuntimeError(f"Figma API failed for {url}: {exc}") from exc

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


def fetch_file_json(
    figma_token: str,
    file_key: str,
    depth: int = FIGMA_PREFLIGHT_DEPTH,
    job_id: Optional[str] = None,
    context: str = "Fetching Figma file JSON",
) -> dict:
    return figma_get(
        f"https://api.figma.com/v1/files/{file_key}",
        figma_token,
        {"depth": depth},
        job_id=job_id,
        context=context,
    )


def rgb01_to_hex(r: float, g: float, b: float) -> str:
    rr = max(0, min(255, round(r * 255)))
    gg = max(0, min(255, round(g * 255)))
    bb = max(0, min(255, round(b * 255)))
    return f"#{rr:02X}{gg:02X}{bb:02X}"


def color_distance(c1: Dict[str, Any], c2: Dict[str, Any]) -> float:
    try:
        dr = float(c1["r"]) - float(c2["r"])
        dg = float(c1["g"]) - float(c2["g"])
        db = float(c1["b"]) - float(c2["b"])
        return math.sqrt(dr * dr + dg * dg + db * db)
    except Exception:
        return 999.0


def extract_bound_variable_names(value: Any) -> List[str]:
    names: List[str] = []

    def _walk(v: Any) -> None:
        if isinstance(v, dict):
            for k, val in v.items():
                if k.lower() in {"name", "variable_name"} and isinstance(val, str):
                    names.append(val)
                else:
                    _walk(val)
        elif isinstance(v, list):
            for item in v:
                _walk(item)

    _walk(value)
    return names


def simplify_paint(paint: dict) -> dict:
    if not isinstance(paint, dict):
        return {}
    color = paint.get("color", {}) or {}
    bound_vars = paint.get("boundVariables")
    return {
        "type": paint.get("type"),
        "visible": paint.get("visible", True),
        "opacity": paint.get("opacity", 1),
        "r": color.get("r"),
        "g": color.get("g"),
        "b": color.get("b"),
        "hex": rgb01_to_hex(color.get("r", 0), color.get("g", 0), color.get("b", 0)) if color else None,
        "boundVariables": bound_vars,
        "boundVariableNames": extract_bound_variable_names(bound_vars),
        "has_bound_variable": bool(bound_vars),
    }


def extract_node_signature(node: dict) -> dict:
    style = node.get("style", {}) if node.get("type") == "TEXT" else {}
    fills = [simplify_paint(p) for p in node.get("fills", []) if isinstance(p, dict)]
    strokes = [simplify_paint(p) for p in node.get("strokes", []) if isinstance(p, dict)]

    return {
        "id": node.get("id"),
        "name": node.get("name"),
        "type": node.get("type"),
        "visible": node.get("visible", True),
        "bbox": figma_bbox_to_tuple(node.get("absoluteBoundingBox")),
        "componentId": node.get("componentId"),
        "styles": node.get("styles", {}) or {},
        "boundVariables": node.get("boundVariables", {}) or {},
        "fills": fills,
        "strokes": strokes,
        "cornerRadius": node.get("cornerRadius"),
        "rectangleCornerRadii": node.get("rectangleCornerRadii"),
        "strokeWeight": node.get("strokeWeight"),
        "itemSpacing": node.get("itemSpacing"),
        "paddingLeft": node.get("paddingLeft"),
        "paddingRight": node.get("paddingRight"),
        "paddingTop": node.get("paddingTop"),
        "paddingBottom": node.get("paddingBottom"),
        "layoutMode": node.get("layoutMode"),
        "primaryAxisSizingMode": node.get("primaryAxisSizingMode"),
        "counterAxisSizingMode": node.get("counterAxisSizingMode"),
        "characters": node.get("characters") if node.get("type") == "TEXT" else None,
        "fontFamily": style.get("fontFamily") if style else None,
        "fontWeight": style.get("fontWeight") if style else None,
        "fontSize": style.get("fontSize") if style else None,
        "lineHeightPx": style.get("lineHeightPx") if style else None,
        "letterSpacing": style.get("letterSpacing") if style else None,
        "textAlignHorizontal": style.get("textAlignHorizontal") if style else None,
        "textAlignVertical": style.get("textAlignVertical") if style else None,
    }


def is_actionable_node(sig: Dict[str, Any]) -> bool:
    if not sig.get("visible", True):
        return False
    t = sig.get("type")
    return t in {
        "TEXT",
        "FRAME",
        "INSTANCE",
        "COMPONENT",
        "RECTANGLE",
        "ELLIPSE",
        "GROUP",
        "VECTOR",
        "POLYGON",
        "STAR",
        "LINE",
        "COMPONENT_SET",
    }


# =============================
# DS registry + preflight
# =============================
def build_ds_registry(ds_file: dict, page_index: int) -> DSRegistry:
    page = get_page_node(ds_file, page_index)
    nodes = walk_nodes(page)

    role_signatures: Dict[str, DSRoleSignature] = {}
    all_fill_samples: List[Dict[str, Any]] = []
    known_text_style_ids: List[str] = []

    for node in nodes:
        sig = extract_node_signature(node)
        if not is_actionable_node(sig):
            continue

        role = infer_role(sig)
        role_sig = role_signatures.setdefault(
            role,
            DSRoleSignature(role=role, sample_node_name=sig.get("name", role)),
        )

        text_style_id = sig.get("styles", {}).get("text")
        if text_style_id:
            role_sig.text_style_ids.append(str(text_style_id))
            known_text_style_ids.append(str(text_style_id))

        if sig.get("fontSize") is not None:
            role_sig.font_sizes.append(float(sig["fontSize"]))

        if sig.get("lineHeightPx") is not None:
            role_sig.line_heights.append(float(sig["lineHeightPx"]))

        cr = sig.get("cornerRadius")
        if isinstance(cr, (int, float)):
            role_sig.corner_radii.append(float(cr))
        else:
            radii = sig.get("rectangleCornerRadii")
            if isinstance(radii, list) and radii:
                numeric = [float(v) for v in radii if isinstance(v, (int, float))]
                if numeric:
                    role_sig.corner_radii.append(median(numeric) or numeric[0])

        if isinstance(sig.get("itemSpacing"), (int, float)):
            role_sig.item_spacings.append(float(sig["itemSpacing"]))

        paddings = (
            sig.get("paddingLeft"),
            sig.get("paddingRight"),
            sig.get("paddingTop"),
            sig.get("paddingBottom"),
        )
        if all(isinstance(v, (int, float)) for v in paddings):
            role_sig.paddings.append(
                (
                    float(sig["paddingLeft"]),
                    float(sig["paddingRight"]),
                    float(sig["paddingTop"]),
                    float(sig["paddingBottom"]),
                )
            )

        for fill in sig.get("fills", []) or []:
            if fill.get("type") != "SOLID":
                continue
            sample = {
                "role": role,
                "node_name": sig.get("name"),
                "hex": fill.get("hex"),
                "r": fill.get("r"),
                "g": fill.get("g"),
                "b": fill.get("b"),
                "has_bound_variable": fill.get("has_bound_variable", False),
                "boundVariableNames": fill.get("boundVariableNames", []),
            }
            all_fill_samples.append(sample)
            role_sig.fills.append(sample)

    return DSRegistry(
        role_signatures=role_signatures,
        all_fill_samples=all_fill_samples,
        known_text_style_ids=sorted(set(known_text_style_ids)),
    )


def nearest_fill_sample(target_fill: Dict[str, Any], ds_registry: DSRegistry, role: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if target_fill.get("type") != "SOLID":
        return None

    candidates = ds_registry.all_fill_samples
    if role and role in ds_registry.role_signatures:
        role_candidates = ds_registry.role_signatures[role].fills
        if role_candidates:
            candidates = role_candidates

    best = None
    best_dist = 999.0
    for sample in candidates:
        dist = color_distance(target_fill, sample)
        if dist < best_dist:
            best_dist = dist
            best = sample

    if best is not None:
        best = dict(best)
        best["distance"] = best_dist
    return best


def make_issue(
    severity: str,
    category: str,
    sig: Dict[str, Any],
    property_name: str,
    actual: Any,
    expected: Any,
    hint: str,
) -> PreflightIssue:
    return PreflightIssue(
        severity=severity,
        category=category,
        node_id=str(sig.get("id")),
        node_name=str(sig.get("name")),
        property_name=property_name,
        actual=actual,
        expected=expected,
        hint=hint,
        bbox=sig.get("bbox"),
    )


def check_text_style(sig: Dict[str, Any], role_sig: DSRoleSignature) -> List[PreflightIssue]:
    issues: List[PreflightIssue] = []
    if sig.get("type") != "TEXT":
        return issues

    actual_style_id = sig.get("styles", {}).get("text")
    expected_style_id = role_sig.representative_text_style_id()

    if expected_style_id and not actual_style_id:
        issues.append(
            make_issue(
                "high",
                "Typography",
                sig,
                "text_style",
                "missing text style",
                expected_style_id,
                "DS role usually uses a text style, but this text appears unstyled or locally overridden.",
            )
        )
    elif expected_style_id and actual_style_id and str(actual_style_id) != str(expected_style_id):
        issues.append(
            make_issue(
                "medium",
                "Typography",
                sig,
                "text_style",
                actual_style_id,
                expected_style_id,
                "Text style differs from the most typical DS style for this role.",
            )
        )

    exp_font_size = role_sig.representative_font_size()
    if exp_font_size is not None and isinstance(sig.get("fontSize"), (int, float)):
        actual_fs = float(sig["fontSize"])
        if abs(actual_fs - exp_font_size) > FONT_SIZE_TOLERANCE:
            issues.append(
                make_issue(
                    "medium",
                    "Typography",
                    sig,
                    "font_size",
                    actual_fs,
                    exp_font_size,
                    "Font size differs from the DS role median.",
                )
            )

    exp_lh = role_sig.representative_line_height()
    if exp_lh is not None and isinstance(sig.get("lineHeightPx"), (int, float)):
        actual_lh = float(sig["lineHeightPx"])
        if abs(actual_lh - exp_lh) > LINE_HEIGHT_TOLERANCE:
            issues.append(
                make_issue(
                    "low",
                    "Typography",
                    sig,
                    "line_height",
                    actual_lh,
                    exp_lh,
                    "Line height differs from the DS role median.",
                )
            )

    return issues


def check_fill_binding(sig: Dict[str, Any], role_sig: DSRoleSignature, ds_registry: DSRegistry) -> List[PreflightIssue]:
    issues: List[PreflightIssue] = []
    fills = sig.get("fills") or []
    solid_fills = [f for f in fills if f.get("type") == "SOLID"]
    if not solid_fills:
        return issues

    expected_fill = role_sig.representative_fill()
    if not expected_fill:
        return issues

    target_fill = solid_fills[0]
    has_bound = bool(target_fill.get("has_bound_variable"))
    expected_has_bound = expected_fill.get("has_bound_variable", False)

    if expected_has_bound and not has_bound:
        nearest = nearest_fill_sample(target_fill, ds_registry, role_sig.role)
        hint = "Fill appears hardcoded where DS role commonly uses a variable-bound fill."
        expected_desc = expected_fill.get("boundVariableNames") or expected_fill.get("hex")
        actual_desc = target_fill.get("hex")
        if nearest and nearest.get("distance", 999) <= COLOR_DISTANCE_THRESHOLD:
            hint += f" Color is visually close to DS sample {nearest.get('hex')}."
        issues.append(
            make_issue(
                "high",
                "Color / Tokenization",
                sig,
                "fill",
                actual_desc,
                expected_desc,
                hint,
            )
        )

    nearest = nearest_fill_sample(target_fill, ds_registry, role_sig.role)
    if nearest and nearest.get("distance", 999) <= COLOR_DISTANCE_THRESHOLD:
        if not has_bound and nearest.get("has_bound_variable"):
            issues.append(
                make_issue(
                    "medium",
                    "Color / Tokenization",
                    sig,
                    "fill_near_match",
                    target_fill.get("hex"),
                    nearest.get("boundVariableNames") or nearest.get("hex"),
                    "Hardcoded fill is very close to a DS variable-bound color and may be a manual copy instead of a token.",
                )
            )
    elif nearest and nearest.get("distance", 999) > COLOR_DISTANCE_THRESHOLD * 2:
        issues.append(
            make_issue(
                "medium",
                "Color",
                sig,
                "fill_color",
                target_fill.get("hex"),
                expected_fill.get("boundVariableNames") or expected_fill.get("hex"),
                "Fill color differs notably from the DS role sample.",
            )
        )

    return issues


def check_corner_radius(sig: Dict[str, Any], role_sig: DSRoleSignature) -> List[PreflightIssue]:
    issues: List[PreflightIssue] = []
    expected_radius = role_sig.representative_corner_radius()
    if expected_radius is None:
        return issues

    actual_radius: Optional[float] = None
    cr = sig.get("cornerRadius")
    if isinstance(cr, (int, float)):
        actual_radius = float(cr)
    else:
        radii = sig.get("rectangleCornerRadii")
        if isinstance(radii, list) and radii:
            numeric = [float(v) for v in radii if isinstance(v, (int, float))]
            if numeric:
                actual_radius = median(numeric)

    if actual_radius is None:
        return issues

    if abs(actual_radius - expected_radius) > RADIUS_TOLERANCE:
        issues.append(
            make_issue(
                "medium",
                "Radius",
                sig,
                "corner_radius",
                actual_radius,
                expected_radius,
                "Corner radius differs from the DS role median.",
            )
        )
    return issues


def check_spacing(sig: Dict[str, Any], role_sig: DSRoleSignature) -> List[PreflightIssue]:
    issues: List[PreflightIssue] = []

    expected_spacing = role_sig.representative_item_spacing()
    actual_spacing = sig.get("itemSpacing")
    if expected_spacing is not None and isinstance(actual_spacing, (int, float)):
        actual_spacing_f = float(actual_spacing)
        if abs(actual_spacing_f - expected_spacing) > SPACING_TOLERANCE:
            issues.append(
                make_issue(
                    "low",
                    "Spacing",
                    sig,
                    "item_spacing",
                    actual_spacing_f,
                    expected_spacing,
                    "Item spacing differs from the DS role median.",
                )
            )

    expected_padding = role_sig.representative_padding()
    if expected_padding:
        paddings = (
            sig.get("paddingLeft"),
            sig.get("paddingRight"),
            sig.get("paddingTop"),
            sig.get("paddingBottom"),
        )
        if all(isinstance(v, (int, float)) for v in paddings):
            actual_padding = tuple(float(v) for v in paddings)
            diffs = [abs(a - e) for a, e in zip(actual_padding, expected_padding)]
            if any(d > SPACING_TOLERANCE for d in diffs):
                issues.append(
                    make_issue(
                        "low",
                        "Spacing",
                        sig,
                        "padding",
                        actual_padding,
                        tuple(round(v, 2) for v in expected_padding),
                        "Padding differs from the DS role median.",
                    )
                )

    return issues


def dedupe_issues(issues: List[PreflightIssue]) -> List[PreflightIssue]:
    seen = set()
    out = []
    for issue in issues:
        key = (
            issue.category,
            issue.node_id,
            issue.property_name,
            str(issue.actual),
            str(issue.expected),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(issue)

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda i: (severity_rank.get(i.severity, 99), i.category, i.node_name))
    return out


def run_preflight_checks(ds_registry: DSRegistry, target_file: dict, page_index: int) -> PreflightReport:
    page = get_page_node(target_file, page_index)
    nodes = walk_nodes(page)
    issues: List[PreflightIssue] = []
    scanned = 0

    for node in nodes:
        sig = extract_node_signature(node)
        if not is_actionable_node(sig):
            continue
        scanned += 1

        role = infer_role(sig)
        role_sig = ds_registry.role_signatures.get(role)

        if not role_sig and role in {"heading", "label", "caption"}:
            role_sig = ds_registry.role_signatures.get("text")
        if not role_sig and role in {"container", "frame", "group", "instance", "component"}:
            role_sig = ds_registry.role_signatures.get("card") or ds_registry.role_signatures.get("container")

        if not role_sig:
            continue

        issues.extend(check_text_style(sig, role_sig))
        issues.extend(check_fill_binding(sig, role_sig, ds_registry))
        issues.extend(check_corner_radius(sig, role_sig))
        issues.extend(check_spacing(sig, role_sig))

    issues = dedupe_issues(issues)
    summary = {
        "target_nodes_scanned": scanned,
        "total_issues": len(issues),
        "high_severity": sum(1 for i in issues if i.severity == "high"),
        "medium_severity": sum(1 for i in issues if i.severity == "medium"),
        "low_severity": sum(1 for i in issues if i.severity == "low"),
    }

    return PreflightReport(
        summary=summary,
        issues=issues,
        target_nodes_scanned=scanned,
        ds_roles_found=sorted(ds_registry.role_signatures.keys()),
    )


def filter_issues_for_patch(
    issues: List[PreflightIssue],
    patch_box: Tuple[float, float, float, float],
    max_issues: int = 12,
) -> List[PreflightIssue]:
    matched = [i for i in issues if bbox_intersects(i.bbox, patch_box)]
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    matched.sort(key=lambda i: (severity_rank.get(i.severity, 99), i.category, i.node_name))
    return matched[:max_issues]


def build_patch_preflight_text(
    patch_box: Tuple[float, float, float, float],
    issues: List[PreflightIssue],
) -> str:
    lines = []
    lines.append(f"Patch focus area: {tuple(round(v, 1) for v in patch_box)}")
    lines.append("Deep zoom pass: inspect micro-typography, alignment, spacing, border/radius, and color consistency.")
    if issues:
        lines.append("Preflight issues intersecting this patch:")
        for issue in issues:
            lines.append(issue.to_prompt_line())
    else:
        lines.append("No deterministic preflight issues intersect this patch directly. Look for subtle visual inconsistencies anyway.")
    return "\n".join(lines)


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
You are an elite UI linting auditor combining:
1. deterministic Figma preflight findings
2. visual comparison of rendered UI

Audit 'MY DESIGN' against the 'DS REFERENCES'.

Instructions:
- Verify whether the deterministic findings appear visually true.
- Find additional visual inconsistencies not captured by preflight.
- Prioritize high-confidence issues.
- If preflight suggests a likely token/style mismatch, inspect that region/category first.
- Distinguish structural hints from visual confirmation.
- If you see anything which does not fit design principles at all like a green delete button please tell
- Also check if spelling is wrong

Return only a Markdown table with exactly these columns:
| Source | Category | Element | Issue Found | REQUIRED FIX (FROM -> TO) |
| :--- | :--- | :--- | :--- | :--- |

Rules:
- Source must be Preflight, Visual, or Both.
- Be concrete and actionable.
- Compare typography, spacing, radius, borders, colors, alignment, and component consistency.
- If you can infer likely design-system token names or values from the reference or preflight, mention them.
- If an issue is ambiguous, explain what seems inconsistent and why.
- Only show clear/severe mistakes

Deterministic preflight:
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
    file_data = figma_get(
        f"https://api.figma.com/v1/files/{file_key}",
        figma_token,
        {"depth": 1},
        job_id=job_id,
        context=f"Fetching page metadata for {label}",
    )

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
        job_id=job_id,
        context=f"Requesting rendered image from Figma for {label} / {page_name}",
    )

    img_url = image_data.get("images", {}).get(page_id)
    if not img_url:
        raise RuntimeError(f"Could not render page '{page_name}' for {label}")

    append_job_log(job_id, f"Downloading rendered image for {label} / {page_name}...")
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

        append_job_log(job_id, "Fetching design-system JSON for preflight...")
        ds_file = fetch_file_json(
            figma_token,
            ds_key,
            depth=FIGMA_PREFLIGHT_DEPTH,
            job_id=job_id,
            context="Fetching design-system JSON for preflight",
        )

        append_job_log(job_id, "Fetching target file JSON for preflight...")
        tg_file = fetch_file_json(
            figma_token,
            tg_key,
            depth=FIGMA_PREFLIGHT_DEPTH,
            job_id=job_id,
            context="Fetching target file JSON for preflight",
        )

        append_job_log(job_id, "Building design-system registry...")
        ds_registry = build_ds_registry(ds_file, ds_page_index)

        append_job_log(job_id, "Running deterministic preflight checks...")
        preflight_report = run_preflight_checks(ds_registry, tg_file, target_page_index)
        preflight_text = preflight_report.to_prompt_text()

        append_job_log(job_id, "Rendering design-system page from Figma...")
        ds_img = render_figma_page(figma_token, ds_key, ds_page_index, "design_system", job_id)

        append_job_log(job_id, "Rendering target page from Figma...")
        tg_img = render_figma_page(figma_token, tg_key, target_page_index, "target_design", job_id)

        auditor = VisionAuditor(openai_key, OPENAI_MODEL)

        append_job_log(job_id, f"Running full-page audit with {OPENAI_MODEL}...")
        full_results = auditor.run_pass([ds_img], tg_img, preflight_text)

        append_job_log(job_id, "Running patch analysis for dense UI regions...")
        patches = create_patches(tg_img, PATCH_DIR, job_id)
        patch_findings: List[str] = []

        for idx, (score, patch_path, box) in enumerate(patches, start=1):
            append_job_log(job_id, f"Patch {idx}/{len(patches)} | ink score {score:.2f} | area {box}")
            patch_issues = filter_issues_for_patch(preflight_report.issues, box, max_issues=12)
            patch_preflight = build_patch_preflight_text(box, patch_issues)
            result = auditor.run_pass([ds_img], patch_path, patch_preflight)
            patch_findings.append(f"### Area {box}\n{result}\n")

        report = f"""# Linterface - Figma Design Audit Report

**Date:** {time.ctime()}
**Profile:** {profile.name}
**Model:** {OPENAI_MODEL}
**Target URL:** {target_url}

## Full Page Findings
{full_results}
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
            preflight_summary=preflight_report.summary,
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
                    "preflight_summary": payload.get("preflight_summary"),
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
            "preflight_summary": payload.get("preflight_summary"),
        }
