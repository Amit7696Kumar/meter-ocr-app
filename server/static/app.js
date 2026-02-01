function qs(sel, root=document){ return root.querySelector(sel); }
function qsa(sel, root=document){ return Array.from(root.querySelectorAll(sel)); }

function safeJsonParse(str) {
  try { return JSON.parse(str); } catch { return null; }
}

function setupTabs() {
  const tabs = qsa(".tab");
  const contents = qsa(".tab-content");
  if (!tabs.length) return;

  tabs.forEach(t => {
    t.addEventListener("click", () => {
      tabs.forEach(x => x.classList.remove("active"));
      contents.forEach(c => c.classList.remove("active"));
      t.classList.add("active");
      const name = t.dataset.tab;
      const content = qs(`[data-tab-content="${name}"]`);
      if (content) content.classList.add("active");
    });
  });
}

function setupFiltering(tableId, searchId, meterSelectId) {
  const table = qs(`#${tableId}`);
  const search = qs(`#${searchId}`);
  const meterSel = qs(`#${meterSelectId}`);

  if (!table || !search || !meterSel) return;

  const filter = () => {
    const q = (search.value || "").toLowerCase().trim();
    const meter = meterSel.value;

    qsa("tbody tr", table).forEach(tr => {
      const meterOk = (meter === "all") || (tr.dataset.meter === meter);
      const text = tr.innerText.toLowerCase();
      const qOk = !q || text.includes(q);
      tr.style.display = (meterOk && qOk) ? "" : "none";
    });
  };

  search.addEventListener("input", filter);
  meterSel.addEventListener("change", filter);
}

function openModal({title, sub, images, jsonText}) {
  const modal = qs("#modal");
  const modalTitle = qs("#modalTitle");
  const modalSub = qs("#modalSub");
  const modalImages = qs("#modalImages");
  const modalJson = qs("#modalJson");

  modalTitle.textContent = title || "Reading";
  modalSub.textContent = sub || "";
  modalImages.innerHTML = "";
  modalJson.textContent = jsonText || "";

  (images || []).forEach(url => {
    if (!url) return;
    const img = document.createElement("img");
    img.src = url;
    img.loading = "lazy";
    modalImages.appendChild(img);
  });

  modal.classList.add("show");
}

function closeModal() {
  const modal = qs("#modal");
  if (modal) modal.classList.remove("show");
}

function wireModal(opts = {}) {
  const openImageInModal = opts.openImageInModal === true;

  const modal = document.getElementById("modal");
  const modalClose = document.getElementById("modalClose");
  const modalTitle = document.getElementById("modalTitle");
  const modalSub = document.getElementById("modalSub");
  const modalImages = document.getElementById("modalImages");
  const modalJson = document.getElementById("modalJson");

  if (!modal) return;

  function openModal(row) {
    const created = row.dataset.created || "";
    const meter = row.dataset.meter || "";
    const value = row.dataset.value || "";
    const team = row.dataset.team || "";
    const user = row.dataset.user || "";
    const label = row.dataset.label || "";
    const filename = row.dataset.filename || "";

    modalTitle.textContent = `Team ${team} | ${user} | ${meter.toUpperCase()} = ${value || "-"}`;
    modalSub.textContent = `${label} • ${created}`;

    modalImages.innerHTML = "";

    const imgs = [];
    if (filename) imgs.push({ label: "Uploaded", url: `/uploads/${filename}` });
    if (row.dataset.debugYolo) imgs.push({ label: "YOLO", url: row.dataset.debugYolo });
    if (row.dataset.debugCrop) imgs.push({ label: "Crop", url: row.dataset.debugCrop });
    if (row.dataset.debugPrep) imgs.push({ label: "Preprocess", url: row.dataset.debugPrep });

    imgs.forEach(i => {
      const wrap = document.createElement("div");
      wrap.className = "imgcard";
      wrap.innerHTML = `
        <div class="small-muted" style="margin-bottom:6px;"><b>${i.label}</b></div>
        <img src="${i.url}" style="max-width:100%; border:2px solid #333; border-radius:10px;" />
      `;
      modalImages.appendChild(wrap);
    });

    let raw = row.dataset.ocrjson || "";
    try {
      const obj = JSON.parse(raw);
      modalJson.textContent = JSON.stringify(obj, null, 2);
    } catch (e) {
      modalJson.textContent = raw || "(empty)";
    }

    modal.classList.add("open");
  }

  function closeModal() {
    modal.classList.remove("open");
  }

  modalClose?.addEventListener("click", closeModal);
  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeModal();
  });

  // Click value -> modal
  document.querySelectorAll(".js-open-reading").forEach(btn => {
    btn.addEventListener("click", () => {
      const row = btn.closest("tr");
      if (row) openModal(row);
    });
  });

  // Click Open -> modal (same page)
  document.querySelectorAll(".js-open-image").forEach(btn => {
    btn.addEventListener("click", () => {
      const row = btn.closest("tr");
      if (!row) return;

      if (openImageInModal) {
        openModal(row);
      } else {
        const filename = row.dataset.filename;
        if (filename) window.open(`/uploads/${filename}`, "_blank");
      }
    });
  });
}

function decodeHtml(s) {
  const txt = document.createElement("textarea");
  txt.innerHTML = s;
  return txt.value;
}

function prettyJson(raw) {
  const obj = safeJsonParse(raw);
  if (!obj) return raw || "";
  return JSON.stringify(obj, null, 2);
}

/** Build trend chart from table rows */
function buildTrendChartFromTable(tableId, canvasId) {
  const table = qs(`#${tableId}`);
  const canvas = qs(`#${canvasId}`);
  if (!table || !canvas || !window.Chart) return;

  // group per day per meter_type: avg
  const rows = qsa("tbody tr", table);
  const buckets = {}; // {day: {earthing: [v], temp:[v]}}

  rows.forEach(tr => {
    const meter = tr.dataset.meter;
    const v = parseFloat(tr.dataset.value);
    if (!meter || Number.isNaN(v)) return;

    // created_at is text; take first 10 chars as YYYY-MM-DD if possible
    const created = tr.dataset.created || "";
    const day = created.slice(0, 10) || "unknown";

    if (!buckets[day]) buckets[day] = {earthing: [], temp: []};
    if (meter === "earthing") buckets[day].earthing.push(v);
    if (meter === "temp") buckets[day].temp.push(v);
  });

  const days = Object.keys(buckets).sort();
  const avg = (arr) => arr.length ? (arr.reduce((a,b)=>a+b,0)/arr.length) : null;

  const earthingSeries = days.map(d => avg(buckets[d].earthing));
  const tempSeries = days.map(d => avg(buckets[d].temp));

  new Chart(canvas, {
    type: "line",
    data: {
      labels: days,
      datasets: [
        { label: "Earthing (avg)", data: earthingSeries },
        { label: "Temp (avg)", data: tempSeries }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: true } },
      scales: { y: { beginAtZero: true } }
    }
  });
}

/** Auto refresh alerts for admin/coadmin without reload */
async function pollAlerts() {
  const badge = qs("#unreadBadge");
  const list = qs("#alertsLive");
  const role = (list && list.dataset.role) || null;
  const team = (list && list.dataset.team) || null;

  if (!role) return;

  try {
    const url = role === "admin"
      ? `/api/alerts/admin`
      : `/api/alerts/coadmin?team=${encodeURIComponent(team || "")}`;

    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) return;
    const data = await res.json();

    if (badge) badge.textContent = data.unread_count ?? 0;

    if (list) {
      list.innerHTML = "";
      (data.alerts || []).forEach(a => {
        const div = document.createElement("div");
        div.className = "toast " + ((a.severity === "high" || a.severity === "low") ? "error" : "ok");
        div.innerHTML = `
          <div style="display:flex; justify-content:space-between; gap:10px;">
            <div>
              <div><span class="pill ${a.severity}">${String(a.severity).toUpperCase()}</span>
              <span style="margin-left:8px; font-weight:800;">${a.message}</span></div>
              <div class="small-muted">Created: ${a.created_at} • ${a.is_read ? "Read" : "Unread"}</div>
            </div>
            ${a.is_read ? "" : `
              <form method="post" action="/alerts/${a.id}/read">
                <button class="btn small" type="submit">Mark Read</button>
              </form>
            `}
          </div>
        `;
        list.appendChild(div);
      });
    }
  } catch {}
}
function wireModal() {
  const modal = document.getElementById("modal");
  const closeBtn = document.getElementById("modalClose");

  function openModal(title, sub, imagesHtml, jsonText) {
    document.getElementById("modalTitle").textContent = title || "Reading";
    document.getElementById("modalSub").textContent = sub || "";
    document.getElementById("modalImages").innerHTML = imagesHtml || "";
    document.getElementById("modalJson").textContent = jsonText || "";
    modal.classList.add("open");
  }

  function closeModal() {
    modal.classList.remove("open");
  }

  closeBtn?.addEventListener("click", closeModal);
  modal?.addEventListener("click", (e) => {
    if (e.target === modal) closeModal();
  });

  // ✅ Click VALUE -> opens debug + JSON
  document.querySelectorAll(".js-open-reading").forEach(btn => {
    btn.addEventListener("click", () => {
      const tr = btn.closest("tr");
      if (!tr) return;

      const filename = tr.dataset.filename || "";
      const ocrjson = tr.dataset.ocrjson || "";
      const yolo = tr.dataset.debugYolo || "";
      const crop = tr.dataset.debugCrop || "";
      const prep = tr.dataset.debugPrep || "";

      const imgs = [];
      if (filename) imgs.push({ label: "Uploaded", url: "/uploads/" + filename });
      if (yolo) imgs.push({ label: "YOLO", url: yolo });
      if (crop) imgs.push({ label: "Crop", url: crop });
      if (prep) imgs.push({ label: "Preprocess", url: prep });

      const imagesHtml = imgs.map(x => `
        <div class="imgcard">
          <div class="small-muted" style="margin-bottom:6px;">${x.label}</div>
          <img src="${x.url}" style="width:100%;border-radius:10px;border:1px solid #333;" />
        </div>
      `).join("");

      let pretty = ocrjson;
      try { pretty = JSON.stringify(JSON.parse(ocrjson), null, 2); } catch(e) {}

      openModal("Reading Details", "", imagesHtml, pretty);
    });
  });

  // ✅ Click IMAGE -> opens uploaded image in SAME modal (not new tab)
  document.querySelectorAll(".js-open-image").forEach(btn => {
    btn.addEventListener("click", () => {
      const src = btn.getAttribute("data-src");
      if (!src) return;

      const imagesHtml = `
        <div class="imgcard">
          <div class="small-muted" style="margin-bottom:6px;">Uploaded Image</div>
          <img src="${src}" style="width:100%;border-radius:10px;border:1px solid #333;" />
        </div>
      `;
      openModal("Uploaded Image", "", imagesHtml, "");
    });
  });
}
document.addEventListener("DOMContentLoaded", () => {
  wireModal();
  setupTabs();
  // only if live alerts container exists
  const list = qs("#alertsLive");
  if (list) {
    pollAlerts();
    setInterval(pollAlerts, 10000);
  }
});