// Video Reach estimator: runs the random forest exported by src/export_site_model.py.
// Title features match src/preprocess.py (length, capital share, digits, keyword flags).
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const VOCAB = ["official", "trailer", "live", "remix", "tutorial", "news", "review", "shorts", "asmr", "vlog"];
  const LABEL = { publish_age_days: "Days since upload", channel_subscribers: "Channel subscribers", duration_seconds: "Length", title_caps_ratio: "Share of capitals in the title",
    title_len: "Title length", has_numbers_title: "Number in the title" };
  let M = null, age = 30, shownExample = null;

  function titleFeatures(t) {
    const chars = [...t];
    const caps = chars.filter((c) => c !== c.toLowerCase() && c === c.toUpperCase()).length;
    const low = t.toLowerCase();
    const f = { title_len: chars.length, title_caps_ratio: caps / Math.max(chars.length, 1), has_numbers_title: /\d/.test(t) ? 1 : 0 };
    VOCAB.forEach((w) => (f["kw_" + w] = low.includes(w) ? 1 : 0));
    return f;
  }
  function parseDuration(s) {
    const parts = String(s).trim().split(":").map((p) => parseFloat(p));
    if (parts.some((p) => !Number.isFinite(p))) return NaN;
    return parts.reduce((acc, p) => acc * 60 + p, 0);
  }
  const parseNum = (s) => { const m = String(s).toLowerCase().replace(/[, _]/g, "").match(/^([\d.]+)([km]?)$/); if (!m) return NaN; return parseFloat(m[1]) * (m[2] === "k" ? 1e3 : m[2] === "m" ? 1e6 : 1); };
  const fmtDur = (sec) => { sec = Math.round(sec); const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60; return h ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `${m}:${String(s).padStart(2, "0")}`; };
  function compact(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(n >= 1e10 ? 0 : 1).replace(/\.0$/, "") + "B";
    if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1).replace(/\.0$/, "") + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1).replace(/\.0$/, "") + "K";
    return String(Math.round(n));
  }

  function predictLog1p(row) {
    const x = M.features.map((f) => row[f] ?? 0);
    let sum = 0;
    for (const t of M.trees) {
      let n = 0;
      while (t.l[n] !== -1) n = x[t.f[n]] <= t.t[n] ? t.l[n] : t.r[n];
      sum += t.v[n];
    }
    return sum / M.trees.length;
  }
  const views = (row) => Math.expm1(predictLog1p(row));

  function readForm() {
    const title = $("title").value;
    const dur = parseDuration($("duration").value), subs = parseNum($("subs").value);
    return { title, dur, subs, ok: title.trim() && Number.isFinite(dur) && dur > 0 && Number.isFinite(subs) && subs >= 0 };
  }

  function similar(f, k = 5) {
    const L = (v) => Math.log10(1 + Math.max(0, v || 0));
    return M.videos.map((v) => [v, (L(v.subs) - L(f.subs)) ** 2 + (L(v.age) - L(age)) ** 2 + 0.5 * (L(v.dur) - L(f.dur)) ** 2])
      .sort((a, b) => a[1] - b[1]).slice(0, k).map(([v]) => v);
  }

  function chart(base) {
    const days = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 240, 300, 365];
    const pts = days.map((d) => [d, views({ ...base, publish_age_days: d })]);
    const lx = (d) => (Math.log10(d) / Math.log10(365)) * 560 + 30;
    const vals = pts.flatMap(([, v]) => [v * 10 ** M.band_log10[0], v * 10 ** M.band_log10[1]]);
    const lo = Math.log10(Math.max(1, Math.min(...vals))), hi = Math.log10(Math.max(...vals));
    const ly = (v) => 180 - ((Math.log10(Math.max(1, v)) - lo) / (hi - lo || 1)) * 165;
    const line = pts.map(([d, v], i) => `${i ? "L" : "M"}${lx(d).toFixed(1)},${ly(v).toFixed(1)}`).join("");
    const top = pts.map(([d, v]) => `${lx(d).toFixed(1)},${ly(v * 10 ** M.band_log10[1]).toFixed(1)}`);
    const bot = pts.map(([d, v]) => `${lx(d).toFixed(1)},${ly(v * 10 ** M.band_log10[0]).toFixed(1)}`).reverse();
    const ticks = [[1, "1 day"], [7, "1 week"], [30, "1 month"], [90, "3 months"], [365, "1 year"]];
    const ygrid = [];
    for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) ygrid.push(e);
    const cur = views({ ...base, publish_age_days: age });
    return `<svg viewBox="0 0 600 200" role="img" aria-label="Projected views over the first year">
      ${ygrid.map((e) => `<line class="grid" x1="30" x2="590" y1="${ly(10 ** e)}" y2="${ly(10 ** e)}" /><text x="0" y="${ly(10 ** e) + 4}">${compact(10 ** e)}</text>`).join("")}
      <polygon class="band" points="${top.concat(bot).join(" ")}" />
      <path class="line" d="${line}" />
      <circle class="dot" cx="${lx(age)}" cy="${ly(cur)}" r="5" />
      ${ticks.map(([d, t]) => `<text x="${lx(d)}" y="198" text-anchor="${d === 1 ? "start" : d === 365 ? "end" : "middle"}">${t}</text>`).join("")}
    </svg>`;
  }

  const THUMBS = ["#7c3aed", "#ff4f8b", "#0ea5e9", "#f97316", "#10b981", "#ef4444", "#6366f1", "#eab308"];
  function preview(f, v) {
    let h = 0;
    for (const ch of f.title) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    const c1 = THUMBS[h % THUMBS.length], c2 = THUMBS[(h + 1 + ((h >> 4) % 6)) % THUMBS.length];
    $("thumb").style.background = `linear-gradient(135deg, ${c1}, ${c2})`;
    $("thumb-title").textContent = f.title || "Your title";
    $("badge").textContent = Number.isFinite(f.dur) && f.dur > 0 ? fmtDur(f.dur) : "";
    $("pv-title").textContent = f.title || "Your title";
    const when = { 1: "1 day ago", 7: "1 week ago", 30: "1 month ago", 365: "1 year ago" }[age] || `${age} days ago`;
    $("pv-line").innerHTML = `Your channel · ${Number.isFinite(f.subs) ? compact(f.subs) : "?"} subscribers<br /><em>${v === null ? "?" : compact(v)} views</em> · ${when}`;
  }

  function render() {
    if (!M) return;
    const f = readForm();
    const tf = titleFeatures(f.title);
    const kws = VOCAB.filter((w) => tf["kw_" + w]);
    $("detected").innerHTML = `<span>${tf.title_len} characters</span><span>${Math.round(tf.title_caps_ratio * 100)}% capitals</span>${tf.has_numbers_title ? "<span>has a number</span>" : ""}${kws.map((k) => `<span class="kw">${k}</span>`).join("")}`;
    if (!f.ok) { preview(f, null); $("out").innerHTML = `<div class="empty"><div><b>Check the inputs</b>Use a length like 12:30 and a subscriber count like 250,000 or 250K.</div></div>`; return; }
    const base = { ...tf, duration_seconds: f.dur, channel_subscribers: f.subs };
    const v = views({ ...base, publish_age_days: age });
    const lo = v * 10 ** M.band_log10[0], hi = v * 10 ** M.band_log10[1];
    preview(f, v);
    const ageLabel = { 1: "after 1 day", 7: "after 1 week", 30: "after 1 month", 365: "after 1 year" }[age] || `after ${age} days`;
    const imp = Object.entries(M.importance).filter(([k]) => LABEL[k]).sort((a, b) => b[1] - a[1]);
    const ex = shownExample && shownExample.title === f.title ? shownExample : null;
    $("out").innerHTML = `<div class="fade">
      <div class="estimate"><div>
        <span class="label">Estimated views ${ageLabel}</span>
        <p class="big">${compact(v)}</p>
        <p class="range">Likely between <b>${compact(lo)}</b> and <b>${compact(hi)}</b></p>
      </div>${ex ? `<div class="actual">This video actually had <b>${compact(ex.views)}</b> views at ${Math.round(ex.age)} days.</div>` : ""}</div>
      <div class="chart"><header><b>First-year projection</b><span>Shaded band holds 80% of past out-of-sample results</span></header>${chart(base)}</div>
      <div class="two">
        <div><h2 class="sub">Comparable videos in the training data</h2>
          <ul class="vids">${similar(f).map((s) => `<li>${esc(s.title)}<span>${esc(s.channel)} · ${compact(s.subs)} subscribers · ${Math.round(s.age)} days old</span><b>${compact(s.views)}</b></li>`).join("")}</ul></div>
        <div><h2 class="sub">What the model leans on</h2>
          <div class="bars">${imp.slice(0, 6).map(([k, w]) => `<div class="bar"><span>${LABEL[k]}</span><i><em style="width:${(w / imp[0][1]) * 100}%"></em></i><output>${Math.round(w * 100)}%</output></div>`).join("")}</div>
          <p class="note">Age and channel size explain most of it. The title words barely move the estimate, which is the honest finding of this project: metadata predicts reach, not quality.</p></div>
      </div>
      <p class="note">Random forest of ${M.trees.length} trees trained on ${M.n} videos. Out of sample it explains ${Math.round(M.cv_r2_log10 * 100)}% of the variance in log views.</p>
    </div>`;
  }

  let timer = null;
  const later = () => { clearTimeout(timer); timer = setTimeout(render, 120); };
  ["title", "duration", "subs"].forEach((id) => $(id).addEventListener("input", later));
  $("form").addEventListener("submit", (e) => { e.preventDefault(); render(); });
  document.querySelectorAll(".presets").forEach((g) => g.addEventListener("click", (e) => {
    const b = e.target.closest("[data-v]"); if (!b) return;
    $(g.dataset.target).value = b.dataset.v; render();
  }));
  $("age").addEventListener("click", (e) => {
    const b = e.target.closest("[data-d]"); if (!b) return;
    age = +b.dataset.d;
    document.querySelectorAll("#age button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    render();
  });
  $("example").addEventListener("click", () => {
    if (!M) return;
    const pool = M.videos.filter((v) => v.dur > 0 && v.subs > 0);
    const v = pool[Math.floor(Math.random() * pool.length)];
    shownExample = v;
    $("title").value = v.title; $("duration").value = fmtDur(v.dur); $("subs").value = Math.round(v.subs).toLocaleString("en-US");
    const opts = [1, 7, 30, 365];
    age = Math.max(1, Math.round(v.age));
    document.querySelectorAll("#age button").forEach((x) => x.setAttribute("aria-pressed", String(+x.dataset.d === age)));
    if (!opts.includes(age)) document.querySelectorAll("#age button").forEach((x) => x.setAttribute("aria-pressed", "false"));
    render();
  });

  fetch("model.json").then((r) => r.json()).then((m) => { M = m; render(); })
    .catch(() => { $("out").innerHTML = '<div class="empty"><div><b>The model failed to load</b>Refresh to try again.</div></div>'; });
})();
