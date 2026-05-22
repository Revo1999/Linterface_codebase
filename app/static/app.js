// static/app.js
const frontView = document.getElementById("frontView");
const resultView = document.getElementById("resultView");

const profileForm = document.getElementById("profileForm");
const profileSelect = document.getElementById("profileSelect");
const profileMeta = document.getElementById("profileMeta");
const profileWorkingFileName = document.getElementById("profileWorkingFileName");

const selectedProfileName = document.getElementById("selectedProfileName");
const selectedWorkingFileName = document.getElementById("selectedWorkingFileName");

const analyzeBtn = document.getElementById("analyzeBtn");
const backBtn = document.getElementById("backBtn");
const historyLink = document.getElementById("historyLink");

const statusText = document.getElementById("statusText");
const emptyState = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");
const errorState = document.getElementById("errorState");
const markdownOutput = document.getElementById("markdownOutput");

let profiles = [];
let currentProfile = null;
let latestMarkdown = "";
let currentJobId = null;

const API = {
  profiles: "./api/profiles",
  audit: "./api/audit"
};

marked.setOptions({
  breaks: true,
  gfm: true
});

function showView(name) {
  frontView.classList.remove("active");
  resultView.classList.remove("active");

  if (name === "front") {
    frontView.classList.add("active");
  } else {
    resultView.classList.add("active");
  }
}

function setStatus(text) {
  statusText.textContent = text;
}

function showLoading() {
  emptyState.classList.add("hidden");
  errorState.classList.add("hidden");
  markdownOutput.classList.add("hidden");
  loadingState.classList.remove("hidden");
}

function showEmpty(text = 'Select a profile and press "Analyze".') {
  emptyState.innerHTML = text;
  loadingState.classList.add("hidden");
  errorState.classList.add("hidden");
  markdownOutput.classList.add("hidden");
  emptyState.classList.remove("hidden");
}

function showError(message) {
  errorState.textContent = message;
  loadingState.classList.add("hidden");
  emptyState.classList.add("hidden");
  markdownOutput.classList.add("hidden");
  errorState.classList.remove("hidden");
}

function showMarkdown(markdown) {
  latestMarkdown = markdown || "";
  markdownOutput.innerHTML = marked.parse(latestMarkdown);
  loadingState.classList.add("hidden");
  emptyState.classList.add("hidden");
  errorState.classList.add("hidden");
  markdownOutput.classList.remove("hidden");
}

function getProfileName(profile) {
  return profile?.name ?? "Profile";
}

function getWorkingFileUrl(profile) {
  return (
    profile?.working_file_url ||
    profile?.target_url ||
    profile?.figma_url ||
    ""
  );
}

function getWorkingFileLabel(profile) {
  const url = getWorkingFileUrl(profile);
  if (!url) return "No working file configured";

  try {
    const parsed = new URL(url);
    const parts = parsed.pathname.split("/").filter(Boolean);
    return decodeURIComponent(parts[parts.length - 1] || url);
  } catch {
    return url;
  }
}

function fillProfileSelect(items) {
  profileSelect.innerHTML = "";

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Choose profile";
  placeholder.disabled = true;
  placeholder.selected = true;
  profileSelect.appendChild(placeholder);

  items.forEach((profile) => {
    const option = document.createElement("option");
    option.value = profile.name;
    option.textContent = getProfileName(profile);
    profileSelect.appendChild(option);
  });
}

async function loadProfiles() {
  try {
    const res = await fetch(API.profiles);
    if (!res.ok) throw new Error(`Failed to load profiles (${res.status})`);

    profiles = await res.json();

    if (!profiles.length) {
      profileSelect.innerHTML = `<option value="" disabled selected>No profiles found</option>`;
      return;
    }

    fillProfileSelect(profiles);
  } catch (err) {
    console.error(err);
    profileSelect.innerHTML = `<option value="" disabled selected>Could not load profiles</option>`;
  }
}

function resolveProfileByName(name) {
  return profiles.find((p) => p.name === name) || null;
}

function updateProfilePreview(profile) {
  if (!profile) {
    profileWorkingFileName.textContent = "—";
    profileMeta.classList.add("hidden");
    return;
  }

  profileWorkingFileName.textContent = getWorkingFileLabel(profile);
  profileMeta.classList.remove("hidden");
}

async function pollJob(jobId) {
  while (true) {
    const res = await fetch(`./api/audit/${jobId}`);
    if (!res.ok) {
      throw new Error(`Could not fetch job status (${res.status})`);
    }

    const data = await res.json();

    if (Array.isArray(data.logs) && data.logs.length) {
      setStatus(data.logs[data.logs.length - 1]);
    }

    if (data.status === "completed") {
      const reportRes = await fetch(`./api/audit/${jobId}/report`);
      if (!reportRes.ok) {
        throw new Error("Report was completed but could not be loaded.");
      }
      const reportData = await reportRes.json();
      return reportData.markdown || "";
    }

    if (data.status === "failed") {
      throw new Error(data.error || "Audit failed.");
    }

    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

async function runAnalysis() {
  if (!currentProfile) {
    showError("No profile selected.");
    return;
  }

  const workingFileUrl = getWorkingFileUrl(currentProfile);
  if (!workingFileUrl) {
    showError("This profile has no working_file_url configured in profiles.json.");
    return;
  }

  showLoading();
  setStatus(`Starting audit for ${getProfileName(currentProfile)}...`);

  try {
    const res = await fetch(API.audit, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        profile_name: currentProfile.name,
        target_url: workingFileUrl,
        target_page_index: 0,
        ds_page_index: 0
      })
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `Audit failed (${res.status})`);
    }

    const data = await res.json();
    currentJobId = data.job_id;

    const markdown = await pollJob(currentJobId);
    showMarkdown(markdown);
    setStatus(`Done: ${getProfileName(currentProfile)}`);
  } catch (err) {
    console.error(err);
    showError(err.message || "Something went wrong while analyzing.");
    setStatus(`Error: ${getProfileName(currentProfile)}`);
  }
}

profileSelect.addEventListener("change", () => {
  const profile = resolveProfileByName(profileSelect.value);
  updateProfilePreview(profile);
});

profileForm.addEventListener("submit", (event) => {
  event.preventDefault();

  const profile = resolveProfileByName(profileSelect.value);
  if (!profile) return;

  currentProfile = profile;
  selectedProfileName.textContent = getProfileName(profile);
  selectedWorkingFileName.textContent = getWorkingFileLabel(profile);

  showView("result");
  showEmpty('Press <strong>Analyze</strong> to run the audit and render the markdown result.');
  setStatus(`Ready: ${getProfileName(profile)}`);
});

analyzeBtn.addEventListener("click", runAnalysis);

backBtn.addEventListener("click", () => {
  showView("front");
});

historyLink.addEventListener("click", (event) => {
  event.preventDefault();

  if (resultView.classList.contains("active")) {
    const target = markdownOutput.classList.contains("hidden") ? emptyState : markdownOutput;
    target.scrollIntoView({ behavior: "smooth", block: "start" });
    return;
  }

  showView("result");
  selectedProfileName.textContent = currentProfile ? getProfileName(currentProfile) : "—";
  selectedWorkingFileName.textContent = currentProfile ? getWorkingFileLabel(currentProfile) : "—";

  if (latestMarkdown) {
    showMarkdown(latestMarkdown);
  } else {
    showEmpty('Press <strong>Analyze</strong> to run the audit and render the markdown result.');
  }
});

loadProfiles();
showView("front");