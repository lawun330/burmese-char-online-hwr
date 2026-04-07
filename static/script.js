let lines = [];
let currentIndex = 0;
let currentUser = null;
let strokes = [];
let currentStroke = null;
let isDrawing = false;

let saveCountForLine = 0;

let lineSaveCounts = {};

let SAVES_TO_ADVANCE = parseInt(
  localStorage.getItem("savesToAdvance") || "3",
  10,
);

const canvas = document.getElementById("draw-canvas");
const ctx = canvas.getContext("2d");

function resizeCanvas() {
  const wrap = document.getElementById("canvas-wrap");
  const dpr = window.devicePixelRatio || 1;
  const w = wrap.clientWidth - 32;
  const h = wrap.clientHeight - 16;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  ctx.scale(dpr, dpr);
  redraw();
}
window.addEventListener("resize", resizeCanvas);

function getPos(e) {
  const r = canvas.getBoundingClientRect();
  const src = e.touches ? e.touches[0] : e;
  return {
    x: Math.round(src.clientX - r.left),
    y: Math.round(src.clientY - r.top),
    t: parseFloat((Date.now() / 1000).toFixed(6)),
  };
}

function startDraw(e) {
  e.preventDefault();
  currentStroke = [getPos(e)];
  isDrawing = true;
}
function moveDraw(e) {
  if (!isDrawing || !currentStroke) return;
  e.preventDefault();
  currentStroke.push(getPos(e));
  redraw();
}
function endDraw(e) {
  if (!isDrawing) return;
  e.preventDefault();
  if (currentStroke && currentStroke.length > 1) strokes.push(currentStroke);
  currentStroke = null;
  isDrawing = false;
  updateStrokeCount();
  redraw();
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", moveDraw);
canvas.addEventListener("mouseup", endDraw);
canvas.addEventListener("touchstart", startDraw, { passive: false });
canvas.addEventListener("touchmove", moveDraw, { passive: false });
canvas.addEventListener("touchend", endDraw, { passive: false });

function redraw() {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.width / dpr;
  const h = canvas.height / dpr;
  const isLt = document.documentElement.classList.contains("light");

  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = isLt ? "rgba(0,0,0,0.045)" : "rgba(255,255,255,0.025)";
  ctx.lineWidth = 1;
  const step = 30;
  for (let x = 0; x <= w; x += step) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y <= h; y += step) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  if (strokes.length === 0 && !currentStroke) {
    ctx.strokeStyle = isLt ? "rgba(0,0,0,0.08)" : "rgba(255,255,255,0.04)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 6]);
    ctx.beginPath();
    ctx.moveTo(w / 2, h / 2 - 20);
    ctx.lineTo(w / 2, h / 2 + 20);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(w / 2 - 20, h / 2);
    ctx.lineTo(w / 2 + 20, h / 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.strokeStyle =
    getComputedStyle(document.documentElement)
      .getPropertyValue("--accent")
      .trim() || "#1895FF";
  ctx.lineWidth = 2.5;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  const drawS = (s) => {
    if (s.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(s[0].x, s[0].y);
    for (let i = 1; i < s.length; i++) ctx.lineTo(s[i].x, s[i].y);
    ctx.stroke();
  };
  strokes.forEach(drawS);
  if (currentStroke) drawS(currentStroke);
}

function updateStrokeCount() {
  document.getElementById("stroke-count").textContent =
    strokes.length === 1 ? "1 stroke" : `${strokes.length} strokes`;
}

function clearCanvas() {
  strokes = [];
  currentStroke = null;
  isDrawing = false;
  updateStrokeCount();
  redraw();
}
function undoStroke() {
  if (strokes.length > 0) {
    strokes.pop();
    updateStrokeCount();
    redraw();
  }
}

function getActualSaves(lineIndex) {
  return lineSaveCounts[String(lineIndex)] || 0;
}

function savesNeededForLine(lineIndex) {
  const done = getActualSaves(lineIndex);
  return Math.max(0, SAVES_TO_ADVANCE - done);
}

function computeWeightedProgress() {
  let earned = 0;
  const n = lines.length;
  for (let i = 0; i < n; i++) {
    const actual = getActualSaves(i);
    earned += Math.min(actual, SAVES_TO_ADVANCE);
  }
  return { earned, total: n * SAVES_TO_ADVANCE };
}

function updateProgressBar() {
  const { earned, total } = computeWeightedProgress();
  const pct = total ? (earned / total) * 100 : 0;
  const pctStr = pct.toFixed(1) + "%";
  document.getElementById("prog-fill").style.width = pct + "%";
  document.getElementById("prog-label").textContent =
    `${earned} / ${total} · ${pctStr}`;
}

function findFirstIncompleteLine() {
  for (let i = 0; i < lines.length; i++) {
    if (getActualSaves(i) < SAVES_TO_ADVANCE) return i;
  }
  return lines.length - 1;
}

function updateCharBadge() {
  const badge = document.getElementById("char-progress-badge");

  if (SAVES_TO_ADVANCE <= 1) {
    badge.classList.add("hidden");
    return;
  }

  const actual = getActualSaves(currentIndex);
  const target = SAVES_TO_ADVANCE;

  badge.textContent = `${actual} / ${target}`;
  badge.classList.remove(
    "hidden",
    "badge-complete",
    "badge-over",
    "badge-progress",
    "badge-none",
  );

  if (actual === 0) badge.classList.add("badge-none");
  else if (actual === target) badge.classList.add("badge-complete");
  else if (actual > target) badge.classList.add("badge-over");
  else badge.classList.add("badge-progress");
}

function updateDisplay() {
  if (!lines.length) return;
  const container = document.getElementById("syllable-text");
  container.innerHTML = "";

  const indexSpan = document.createElement("span");
  indexSpan.className = "index";
  indexSpan.textContent = currentIndex + 1 + ": ";

  const textNode = document.createTextNode(lines[currentIndex]);

  container.appendChild(indexSpan);
  container.appendChild(textNode);
  updateCharBadge();
}

function prevLine() {
  if (currentIndex > 0) {
    currentIndex--;
    saveCountForLine = 0;
    updateDisplay();
  }
}
function nextLine() {
  if (currentIndex < lines.length - 1) {
    currentIndex++;
    saveCountForLine = 0;
    updateDisplay();
  }
}

async function saveSample() {
  if (!currentUser) {
    toast("Select a user first", true);
    return;
  }
  if (strokes.length === 0) {
    toast("Draw something first", true);
    return;
  }

  const payload = { user: currentUser, index: currentIndex, strokes };

  try {
    const r = await fetch("/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const d = await r.json();

    if (d.ok) {
      lineSaveCounts[String(currentIndex)] =
        (lineSaveCounts[String(currentIndex)] || 0) + 1;

      const actual = getActualSaves(currentIndex);
      const remaining = Math.max(0, SAVES_TO_ADVANCE - actual);

      if (remaining > 0) {
        toast(`Saved → ${d.file}  (${actual}/${SAVES_TO_ADVANCE})`);
      } else {
        toast(`Saved → ${d.file}  ✓ moving to next line`);
      }

      updateProgressBar();
      updateCharBadge();
      clearCanvas();

      if (actual >= SAVES_TO_ADVANCE) {
        saveCountForLine = 0;
        if (currentIndex < lines.length - 1) {
          setTimeout(() => {
            currentIndex++;
            saveCountForLine = 0;
            updateDisplay();
          }, 400);
        }
      }
    } else {
      toast(d.error || "Save failed", true);
    }
  } catch {
    toast("Network error", true);
  }
}

async function loadLineSaveCounts() {
  if (!currentUser) {
    lineSaveCounts = {};
    return;
  }
  try {
    const r = await fetch(
      `/api/line_counts/${encodeURIComponent(currentUser)}`,
    );
    const d = await r.json();
    lineSaveCounts = d.counts || {};
  } catch {
    lineSaveCounts = {};
  }
}

async function loadUsers() {
  const r = await fetch("/api/users");
  return r.json();
}

async function selectUser(name) {
  currentUser = name;
  saveCountForLine = 0;
  await loadLineSaveCounts();

  updateProgressBar();

  currentIndex = findFirstIncompleteLine();
  updateDisplay();
  clearCanvas();

  closeOverlay("user-overlay");
  closeOverlay("settings-overlay");
  toast(`User: ${name} — line ${currentIndex + 1}`);
}

async function createUser() {
  const name = document.getElementById("f-name").value.trim();
  const age = document.getElementById("f-age").value.trim();
  const sex = document.getElementById("f-sex").value;
  const edu = document.getElementById("f-edu").value.trim();

  if (!name) {
    toast("Name required", true);
    return;
  }

  const r = await fetch("/api/users", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, age, sex, education: edu }),
  });
  const d = await r.json();
  if (d.ok) {
    closeOverlay("newuser-overlay");
    await selectUser(d.name);
    ["f-name", "f-age", "f-edu"].forEach(
      (id) => (document.getElementById(id).value = ""),
    );
    document.getElementById("f-sex").value = "";
    openSettings();
  } else {
    toast(d.error || "Error", true);
  }
}

function openNewUserSheet() {
  closeOverlay("user-overlay");
  document.getElementById("newuser-overlay").classList.add("open");
}
function openNewUserFromSettings() {
  closeOverlay("settings-overlay");
  document.getElementById("newuser-overlay").classList.add("open");
}

async function openSettings() {
  document.getElementById("settings-sta").value = SAVES_TO_ADVANCE;

  const d = await (await fetch("/api/users")).json();
  const sel = document.getElementById("settings-user-sel");
  sel.innerHTML = '<option value="">— select user —</option>';
  d.users.forEach((u) => {
    const o = document.createElement("option");
    o.value = u.name;
    o.textContent = u.name;
    if (u.name === currentUser) o.selected = true;
    sel.appendChild(o);
  });

  const total = lines.length || 0;
  const inp = document.getElementById("nav-line-input");
  inp.max = total;
  inp.value = currentIndex + 1;
  document.getElementById("nav-input-max").textContent = total;
  document.getElementById("nav-preview-text").textContent =
    lines[currentIndex] || "—";
  document.getElementById("nav-input-label").textContent = "Line";
  inp.style.color = "var(--accent)";

  document.getElementById("settings-light-toggle").checked =
    document.documentElement.classList.contains("light");

  document.getElementById("settings-overlay").classList.add("open");
}

function onNavInput(val) {
  const total = lines.length;
  const n = parseInt(val);
  const inp = document.getElementById("nav-line-input");
  const lbl = document.getElementById("nav-input-label");
  const prev = document.getElementById("nav-preview-text");

  if (!val) {
    inp.style.color = "var(--accent)";
    lbl.textContent = "Line";
    prev.textContent = "—";
    return;
  }

  if (n >= 1 && n <= total) {
    inp.style.color = "var(--accent)";
    lbl.textContent = "Line";
    prev.textContent = lines[n - 1] || "—";
  } else {
    inp.style.color = "#ef4444";
    lbl.textContent = n > total ? "Too high" : "Invalid";
    prev.textContent = "—";
  }
}

function applySavesToAdvance(val) {
  const n = parseInt(val, 10);
  if (n >= 1 && n <= 20) {
    SAVES_TO_ADVANCE = n;
    localStorage.setItem("savesToAdvance", n);
    updateProgressBar();
    updateCharBadge();
    toast(`Saves to advance: ${n}`);
  }
}
function stepSTA(delta) {
  const inp = document.getElementById("settings-sta");
  const next = Math.min(
    20,
    Math.max(1, (parseInt(inp.value, 10) || SAVES_TO_ADVANCE) + delta),
  );
  inp.value = next;
  applySavesToAdvance(next);
}

async function settingsSelectUser(name) {
  if (!name) return;
  await selectUser(name);
}

function settingsJumpLine() {
  const val = parseInt(document.getElementById("nav-line-input").value, 10);
  if (!isNaN(val) && val >= 1 && val <= lines.length) {
    currentIndex = val - 1;
    saveCountForLine = 0;
    updateDisplay();
    clearCanvas();
    closeOverlay("settings-overlay");
    toast(`Jumped to line ${val}`);
  } else {
    toast(`Enter 1 – ${lines.length}`, true);
  }
}

function goToLatestLine() {
  if (!currentUser) {
    toast("Select a user first", true);
    return;
  }
  currentIndex = findFirstIncompleteLine();
  saveCountForLine = 0;
  updateDisplay();
  clearCanvas();
  closeOverlay("settings-overlay");
  toast(`Jumped to line ${currentIndex + 1}`);
}

function toggleTheme(isLight) {
  document.documentElement.classList.toggle("light", isLight);
  localStorage.setItem("theme", isLight ? "light" : "dark");
  redraw();
}
if (localStorage.getItem("theme") === "light") {
  document.documentElement.classList.add("light");
}

function closeOverlay(id) {
  document.getElementById(id).classList.remove("open");
}
function closeSheet(e, id) {
  if (e.target.id === id) closeOverlay(id);
}

let toastTimer;
function toast(msg, isError = false) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = "show" + (isError ? " error" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.className = ""), 2400);
}

let galleryMode = "single";
const galleryCache = {};

function galleryCacheKey(user, mode) {
  return `${user}|${mode}`;
}

async function openBrowser() {
  galleryMode = "single";
  document.querySelectorAll(".color-tab").forEach((b) => {
    b.classList.toggle("active", b.dataset.mode === "single");
  });

  document.getElementById("gallery-grid").innerHTML = "";
  document.getElementById("gallery-grid").style.display = "";
  document.getElementById("alllist-wrap").style.display = "none";
  document.getElementById("alllist-wrap").classList.remove("visible");
  document.getElementById("gallery-status").textContent = "";

  const nameEl = document.getElementById("browser-user-name");
  nameEl.textContent = currentUser
    ? `· ${currentUser}`
    : "— no user selected —";

  document.getElementById("browser-overlay").classList.add("open");

  if (currentUser) {
    browserLoadGallery();
  } else {
    document.getElementById("gallery-status").textContent =
      "Select a user in Settings first.";
  }
}

function switchGalleryTab(mode, btn) {
  if (mode === galleryMode) return;
  galleryMode = mode;
  document
    .querySelectorAll(".color-tab")
    .forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");

  const grid = document.getElementById("gallery-grid");
  const allWrap = document.getElementById("alllist-wrap");

  if (mode === "alllist") {
    grid.style.display = "none";
    allWrap.style.display = "flex";
    allWrap.classList.add("visible");
    renderAllList();
  } else {
    grid.style.display = "";
    allWrap.style.display = "none";
    allWrap.classList.remove("visible");
    browserLoadGallery();
  }
}

async function browserLoadGallery() {
  const user = currentUser;
  const grid = document.getElementById("gallery-grid");
  const status = document.getElementById("gallery-status");
  grid.innerHTML = "";
  if (!user) {
    status.textContent = "";
    return;
  }

  const key = galleryCacheKey(user, galleryMode);
  const entry = galleryCache[key];

  if (entry && entry.items.length > 0) {
    entry.items.forEach((item) => grid.appendChild(item._card));
    status.textContent = `${entry.items.length} sample${entry.items.length !== 1 ? "s" : ""} — ${galleryMode} mode (checking…)`;
  } else {
    status.textContent = `Generating ${galleryMode} images…`;
  }

  const knownTxts = entry ? entry.items.map((i) => i.txt) : [];

  let d;
  try {
    const r = await fetch(
      `/api/gallery/${encodeURIComponent(user)}/${galleryMode}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ known: knownTxts }),
      },
    );
    d = await r.json();
  } catch {
    status.textContent = "Network error.";
    return;
  }
  if (d.error) {
    status.textContent = d.error;
    return;
  }

  if (!galleryCache[key]) galleryCache[key] = { items: [] };
  const cacheEntry = galleryCache[key];

  d.items.forEach((item) => {
    item._card = makeCard(item, user);
    cacheEntry.items.push(item);
    grid.appendChild(item._card);
  });

  if (cacheEntry.items.length === 0) {
    grid.innerHTML =
      '<div style="color:var(--muted);font-size:12px;grid-column:1/-1;padding:8px">No samples yet.</div>';
  }

  const total = cacheEntry.items.length;
  status.textContent = `${total} sample${total !== 1 ? "s" : ""}`;
}

function makeCard(item, user) {
  const card = document.createElement("div");
  card.className = "gallery-card";
  card.dataset.txt = item.txt;
  card.innerHTML = `
    <img src="${item.img}" alt="${item.txt}" loading="lazy">
    <div class="card-meta">
      <div class="card-label">${item.label || "—"}</div>
      <div class="card-fname">${item.txt}</div>
      <button class="card-del" onclick="galleryDelete('${user}','${item.txt}',this)">Delete</button>
    </div>
  `;
  return card;
}

async function galleryDelete(user, fname, btn) {
  if (!confirm(`Delete ${fname} and all its images?`)) return;
  const r = await fetch(
    `/api/delete/${encodeURIComponent(user)}/${encodeURIComponent(fname)}`,
    { method: "DELETE" },
  );
  const d = await r.json();
  if (d.ok) {
    ["single", "stroke", "time"].forEach((mode) => {
      const key = galleryCacheKey(user, mode);
      if (galleryCache[key]) {
        galleryCache[key].items = galleryCache[key].items.filter(
          (i) => i.txt !== fname,
        );
      }
    });
    btn.closest(".gallery-card").remove();

    if (user === currentUser) {
      if (d.lineCounts) {
        Object.assign(lineSaveCounts, d.lineCounts);
      } else if (typeof d.progress === "number") {
        await loadLineSaveCounts();
      }
      updateProgressBar();
      updateCharBadge();
    }

    const remaining = document.querySelectorAll(
      "#gallery-grid .gallery-card",
    ).length;
    document.getElementById("gallery-status").textContent =
      `${remaining} sample${remaining !== 1 ? "s" : ""} — ${galleryMode} mode`;
    toast(`Deleted ${fname}`);
  } else {
    toast(d.error || "Delete failed", true);
  }
}

let allListData = [];

function buildAllListData() {
  allListData = lines.map((char, i) => {
    const actual = getActualSaves(i);
    const target = SAVES_TO_ADVANCE;
    let statusClass, label;

    if (actual === 0) {
      statusClass = "prog-none";
      label = `0 / ${target}`;
    } else if (actual === target) {
      statusClass = "prog-complete";
      label = `${actual} / ${target}`;
    } else if (actual > target) {
      statusClass = "prog-over";
      label = `${actual} / ${target}`;
    } else {
      statusClass = "prog-progress";
      label = `${actual} / ${target}`;
    }

    return { index: i, char, actual, target, statusClass, label };
  });
}

function renderAllList(filter = "") {
  if (!currentUser) {
    document.getElementById("alllist-container").innerHTML =
      '<div style="color:var(--muted);font-size:12px;padding:8px">Select a user first.</div>';
    return;
  }

  buildAllListData();

  const q = filter.trim().toLowerCase();
  const data = q
    ? allListData.filter(
        (d) => d.char.includes(q) || String(d.index + 1).includes(q),
      )
    : allListData;

  const container = document.getElementById("alllist-container");
  container.innerHTML = "";

  if (data.length === 0) {
    container.innerHTML =
      '<div style="color:var(--muted);font-size:12px;padding:8px">No results.</div>';
    return;
  }

  const BATCH = 200;
  let offset = 0;

  const renderBatch = () => {
    const frag = document.createDocumentFragment();
    const slice = data.slice(offset, offset + BATCH);
    slice.forEach((d) => {
      const row = document.createElement("div");
      row.className = "alllist-row";
      row.innerHTML = `
        <span class="alllist-linenum">${d.index + 1}</span>
        <span class="alllist-char">${d.char}</span>
        <span class="alllist-prog ${d.statusClass}">${d.label}</span>
      `;
      row.addEventListener("click", () => {
        currentIndex = d.index;
        saveCountForLine = 0;
        updateDisplay();
        clearCanvas();
        closeOverlay("browser-overlay");
        toast(`Jumped to line ${d.index + 1}`);
      });
      frag.appendChild(row);
    });
    container.appendChild(frag);
    offset += BATCH;
    if (offset < data.length) requestAnimationFrame(renderBatch);
  };
  renderBatch();
}

function filterAllList(val) {
  renderAllList(val);
}

async function init() {
  const r = await fetch("/api/lines");
  const d = await r.json();
  lines = d.lines;

  updateDisplay();
  updateProgressBar();
  resizeCanvas();
}

init();
