// ── Theme ─────────────────────────────────────────────────────────────────────

const root = document.documentElement;
const themeToggle = document.getElementById("theme-toggle");

function setTheme(theme) {
  root.dataset.theme = theme;
  localStorage.setItem("caption-benchmark-theme", theme);
  if (themeToggle) {
    themeToggle.textContent = theme === "dark" ? "Light" : "Dark";
    themeToggle.setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
  }
}

const savedTheme = localStorage.getItem("caption-benchmark-theme");
const preferredTheme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
setTheme(savedTheme || preferredTheme);
themeToggle?.addEventListener("click", () => setTheme(root.dataset.theme === "dark" ? "light" : "dark"));

// ── Save ──────────────────────────────────────────────────────────────────────

const saveStatus = document.getElementById("save-status");
let saveTimer = null;

function setStatus(text) {
  if (saveStatus) saveStatus.textContent = text;
}

function collectCategoryAnnotations() {
  const out = {};
  document.querySelectorAll('.cat-cell input[type="radio"]:checked').forEach(r => {
    const { model, panel, field } = r.dataset;
    out[model] = out[model] || {};
    out[model][panel] = out[model][panel] || {};
    out[model][panel][field] = r.value === "true";
  });
  return out;
}

function collectSummaryAnnotations() {
  const out = {};
  document.querySelectorAll('.sum-panel-group input[type="radio"]:checked').forEach(r => {
    const { model, panel, field } = r.dataset;
    out[model] = out[model] || {};
    out[model][panel] = out[model][panel] || {};
    out[model][panel][field] = r.value;
  });
  return out;
}

async function saveBoth() {
  if (!window.BENCHMARK) return;
  const { imageName, saveCategoryUrl, saveSummaryUrl } = window.BENCHMARK;
  setStatus("Saving…");
  try {
    await Promise.all([
      fetch(saveCategoryUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_name: imageName, annotations: collectCategoryAnnotations() }),
      }),
      fetch(saveSummaryUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_name: imageName, annotations: collectSummaryAnnotations() }),
      }),
    ]);
    setStatus("Saved ✓");
  } catch {
    setStatus("Save failed");
  }
}

// Save button
document.getElementById("save-button")?.addEventListener("click", () => {
  clearTimeout(saveTimer);
  saveBoth();
});

// Ctrl/Cmd+S
document.addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
    e.preventDefault();
    clearTimeout(saveTimer);
    saveBoth();
  }
});

// ── Flag ──────────────────────────────────────────────────────────────────────

const flagButton = document.getElementById("flag-button");

async function toggleFlag() {
  if (!window.BENCHMARK || !flagButton) return;
  const newFlagged = !window.BENCHMARK.flagged;
  window.BENCHMARK.flagged = newFlagged;
  flagButton.textContent = newFlagged ? "⚑ Flagged" : "⚐ Flag";
  flagButton.classList.toggle("is-flagged", newFlagged);
  flagButton.setAttribute("aria-pressed", String(newFlagged));
  await fetch(window.BENCHMARK.saveFlagUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_name: window.BENCHMARK.imageName, flagged: newFlagged }),
  });
}

flagButton?.addEventListener("click", toggleFlag);

// ── Keyboard shortcuts ────────────────────────────────────────────────────────

document.addEventListener("keydown", e => {
  const tag = document.activeElement?.tagName;
  const type = document.activeElement?.type;
  const inText = tag === "TEXTAREA" || (tag === "INPUT" && type !== "radio");
  if (inText || e.ctrlKey || e.metaKey || e.altKey) return;

  // ] or n = next image
  if (e.key === "]" || e.key === "n") {
    if (window.BENCHMARK?.nextUrl) window.location.href = window.BENCHMARK.nextUrl;
    return;
  }
  // [ or p = prev image
  if (e.key === "[" || e.key === "p") {
    if (window.BENCHMARK?.prevUrl) window.location.href = window.BENCHMARK.prevUrl;
    return;
  }
  // g = toggle flag
  if (e.key === "g") {
    toggleFlag();
    return;
  }

  // Radio shortcuts — only when a radio is focused
  if (type !== "radio") return;

  const focused = document.activeElement;
  const { field, name: _name } = focused.dataset;
  const groupName = focused.name;

  let targetValue = null;
  if (field === "category_correct" || field === "subtype_correct") {
    if (e.key === "t") targetValue = "true";
    else if (e.key === "f") targetValue = "false";
  } else if (field === "subcaption_quality" || field === "summary_quality") {
    if (e.key === "1") targetValue = "poor";
    else if (e.key === "2") targetValue = "average";
    else if (e.key === "3") targetValue = "good";
  }

  if (targetValue) {
    e.preventDefault();
    const radio = document.querySelector(`input[name="${CSS.escape(groupName)}"][value="${targetValue}"]`);
    if (radio && !radio.checked) {
      radio.checked = true;
      radio.dispatchEvent(new Event("change", { bubbles: true }));
    }
    advanceToNextGroup(focused);
  }
});

function advanceToNextGroup(currentRadio) {
  const all = Array.from(document.querySelectorAll('input[type="radio"]'));
  const name = currentRadio.name;
  // find last radio in current group, then jump to first of next group
  let pastGroup = false;
  for (const r of all) {
    if (!pastGroup && r.name === name) { pastGroup = true; continue; }
    if (pastGroup && r.name !== name) { r.focus(); return; }
  }
}

