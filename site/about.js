// Data: python src/export_site_data.py (out-of-fold predictions, same folds as src/evaluate_cv.py).
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const W = 640, H = 520, P = { l: 56, r: 12, t: 12, b: 44 };
let D = null;

const fmtViews = (lg) => { const v = 10 ** lg; return v >= 1e9 ? `${(v / 1e9).toFixed(1)}B` : v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `${Math.round(v / 1e3)}K` : `${Math.round(v)}`; };
const fmtEng = (v) => `${(v * 100).toFixed(1)}%`;

const TASKS = {
  views: { lo: 2, hi: 10, ticks: [2, 4, 6, 8, 10], fmt: fmtViews, axis: "views",
    verdict: "Predictions hug the diagonal across seven orders of magnitude. Most of the signal comes from channel subscribers: big channels get big numbers." },
  engagement: { lo: 0, hi: 0.12, ticks: [0, 0.03, 0.06, 0.09, 0.12], fmt: fmtEng, axis: "likes and comments per view",
    verdict: "No diagonal to hug. Most videos sit near the same low rate, and the outliers the model misses swamp what it gets right. Titles, length and keywords say little about how much viewers interact. Points past the axis are pinned to its edge." },
};

function render(k) {
  const t = TASKS[k], d = D[k];
  document.querySelectorAll("#task button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.k === k)));
  const X = (v) => P.l + ((Math.min(Math.max(v, t.lo), t.hi) - t.lo) / (t.hi - t.lo)) * (W - P.l - P.r);
  const Y = (v) => H - P.b - ((Math.min(Math.max(v, t.lo), t.hi) - t.lo) / (t.hi - t.lo)) * (H - P.t - P.b);
  let s = "";
  t.ticks.forEach((v) => {
    s += `<line class="grid" x1="${P.l}" x2="${W - P.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${P.l - 8}" y="${Y(v) + 4}" text-anchor="end">${t.fmt(v)}</text>`;
    s += `<text x="${X(v)}" y="${H - P.b + 18}" text-anchor="middle">${t.fmt(v)}</text>`;
  });
  s += `<line class="ideal" x1="${X(t.lo)}" y1="${Y(t.lo)}" x2="${X(t.hi)}" y2="${Y(t.hi)}"/>`;
  s += `<text x="${(P.l + W - P.r) / 2}" y="${H - 4}" text-anchor="middle">Actual ${t.axis}</text>`;
  s += `<text transform="translate(12 ${(H - P.b) / 2}) rotate(-90)" text-anchor="middle">Predicted</text>`;
  s += d.actual.map((a, i) => `<circle class="pt" data-i="${i}" cx="${X(a)}" cy="${Y(d.pred[i])}" r="4.5"/>`).join("");
  $("chart").innerHTML = s;
  $("r2").textContent = d.r2.toFixed(2);
  $("r2").className = d.r2 < 0.3 ? "weak" : "";
  $("verdict").textContent = t.verdict;
  $("chart").querySelectorAll(".pt").forEach((c) => {
    c.addEventListener("pointerenter", () => {
      const i = +c.dataset.i, tip = $("tip"), box = $("chart").getBoundingClientRect(), scale = box.width / W;
      tip.innerHTML = `<b>${esc(D.title[i])}</b><span>${esc(D.channel[i])}</span><span>Actual ${t.fmt(d.actual[i])} · predicted ${t.fmt(d.pred[i])}</span>`;
      tip.hidden = false;
      const x = c.cx.baseVal.value * scale, y = c.cy.baseVal.value * scale;
      tip.style.left = `${Math.min(x + 12, box.width - 290)}px`;
      tip.style.top = `${Math.max(0, y - 70)}px`;
    });
    c.addEventListener("pointerleave", () => ($("tip").hidden = true));
  });
}

document.querySelectorAll("#task button").forEach((b) => b.addEventListener("click", () => render(b.dataset.k)));
fetch("data.json").then((r) => r.json()).then((d) => { D = d; render("views"); });
