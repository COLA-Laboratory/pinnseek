/* "Knowledge and agency" on the landing page: the discovery loop of the Methods, animated round by round with
   the recorded Wing3D run (#how-data), a convergence plot of the run, detail panels for every part of the loop,
   the formulation rendered with KaTeX, and the ablation result on LDC. */
(function () {
  var NS = "http://www.w3.org/2000/svg";
  var ROUND_MS = 2600, HOLD_MS = 3500;
  var SUB = "₀₁₂₃₄₅₆₇₈₉";
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  function ready(fn) { if (document.readyState !== "loading") fn(); else document.addEventListener("DOMContentLoaded", fn); }
  function el(tag, a, p) { var e = document.createElementNS(NS, tag); for (var k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; }
  function txt(x, y, s, cls, p, anchor) { var t = el("text", {x: x, y: y, class: cls}, p); if (anchor) t.setAttribute("text-anchor", anchor); t.textContent = s; return t; }
  function fmt(v) { return v >= 0.1 ? v.toFixed(3) : v >= 0.01 ? v.toFixed(4) : v.toPrecision(3); }
  function clamp(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
  function sub(n) { return String(n).replace(/\d/g, function (c) { return SUB[c]; }); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]; }); }
  function pad(n) { return String(n).padStart(3, "0"); }

  function katexIn(root) {
    if (!window.katex) return setTimeout(function () { katexIn(root); }, 120);
    root.querySelectorAll(".pfh-math").forEach(function (e) {
      if (!e.dataset.done) { katex.render(e.getAttribute("data-tex"), e, {displayMode: true, throwOnError: false}); e.dataset.done = 1; }
    });
    root.querySelectorAll(".pfh-math-i").forEach(function (e) {
      if (!e.dataset.done) { katex.render(e.getAttribute("data-tex"), e, {displayMode: false, throwOnError: false}); e.dataset.done = 1; }
    });
  }

  ready(function () {
    katexIn(document);
    var svg = document.getElementById("ka-loop"), dataEl = document.getElementById("how-data");
    if (!svg || !dataEl) return;
    var D = JSON.parse(dataEl.textContent), R = D.rounds.length - 1;
    var stage = document.querySelector(".pfh-ka-stage");
    var citedAll = {};
    D.rounds.forEach(function (r) { r.cited.forEach(function (n) { citedAll[n] = (citedAll[n] || 0) + 1; }); });


    /* ---------------- boxes: viewBox 24..1076 across, 50..BOT + 8 down; the bottom row ends at BOT */
    var BOT = 730, KE0 = 470;
    var VBX = 24, VBY = 50, VBW = 1350;                      // top row: 𝒦ₗ, task and agent, E/score/incumbent, field;
    svg.setAttribute("viewBox", VBX + " " + VBY + " " + VBW + " " + (BOT + 8 - VBY));   // bottom row: summaries, 𝒦ₑ, convergence
    var defs = el("defs", {}, svg);                         // a dot where an edge arrives: at the agent, or from it
    [["ka-arr-in", "rgba(255, 244, 230, .7)"], ["ka-arr-out", "rgba(255, 244, 230, .7)"]].forEach(function (m) {
      var mk = el("marker", {id: m[0], viewBox: "0 0 10 10", refX: 5, refY: 5, markerWidth: 4.5, markerHeight: 4.5}, defs);
      el("circle", {cx: 5, cy: 5, r: 5, fill: m[1]}, mk);
    });
    function box(x, y, w, h, title, sub_, key) {
      var g = el("g", {class: "ka-box ka-hit ka-" + key, "data-panel": key, tabindex: 0, role: "button", "aria-label": title + ": details"}, svg);
      el("rect", {x: x, y: y, width: w, height: h, rx: 12}, g);
      txt(x + 16, y + 26, title, "ka-t", g);
      if (sub_) txt(x + 16, y + 46, sub_, "ka-s", g);
      el("circle", {cx: x + w - 14, cy: y + 14, r: 3, class: "ka-more"}, g);
      return g;
    }
    var task = box(419, 72, 262, 62, "Task package 𝒯", "description · standard PINN · public files", "task");
    var kl = box(36, 72, 262, KE0 - 106, "Literature base 𝒦ₗ", D.n_notes + " human-verified notes", "kl");
    var dots = [];
    for (var i = 0; i < D.n_notes; i++) dots.push(el("rect", {x: 52 + (i % 10) * 23.56, y: 132 + Math.floor(i / 10) * 22.4, width: 18, height: 18, rx: 3, class: "ka-note"}, kl));
    box(802, 72, 262, 62, "Evaluation tool 𝐸", "trains a candidate for τ on one GPU", "eval");
    var sc = box(802, 223, 262, 62, "Score service", "the only reader of the reference", "score");
    el("path", {d: "M1030 245 h12 v-5 a6 6 0 0 0 -12 0 z M1027 245 h18 v14 h-18 z", class: "ka-lock"}, sc);
    var inc = box(802, KE0 - 96, 262, 62, "Incumbent", "lowest error of the full evaluations", "inc");
    var incV = txt(1040, KE0 - 66, "", "ka-incv", inc, "end");

    var SY = KE0 + 92, AY = BOT - 94;           // rows of the experience base: labels above, content below
    var ke = box(334, KE0, 496, BOT - KE0, "Experience base 𝒦ₑ", "summaries and artifacts of every round", "ke");
    txt(350, SY - 10, "Summaries", "ka-row", ke);
    var tiles = [];
    for (var k = 0; k <= R; k++) {
      var g = el("g", {class: "ka-tile"}, ke);
      el("rect", {x: 350 + k * 42.4, y: SY, width: 37.4, height: 42, rx: 6}, g);
      txt(368.7 + k * 42.4, SY + 26.5, "m" + sub(k), "ka-tile-t", g, "middle");
      tiles.push(g);
    }
    txt(350, AY, "Artifacts", "ka-row", ke);
    var artP = txt(350, AY + 22, "", "ka-art", ke), artE = txt(582, AY + 22, "", "ka-art", ke);
    var cumC = [], cumE = [], cc = 0, ce = 0;
    D.rounds.forEach(function (r) { cc += r.cand || 0; ce += r.evals || 0; cumC.push(cc); cumE.push(ce); });
    var artTiles = [];
    for (var a = 0; a < 112; a++) artTiles.push(el("rect", {x: 350 + (a % 56) * 8.29, y: AY + 34 + Math.floor(a / 56) * 24, width: 6.3, height: 18, rx: 1.5, class: "ka-art-t"}, ke));

    /* ---------------- convergence of the run */
    var CV0 = KE0;
    var cv = box(866, CV0, 496, BOT - CV0, "Convergence curve", "relative L² error", "conv");
    var PX0 = 916, PX1 = 1344, PY0 = CV0 + 62, PY1 = BOT - 40, lo = Math.log10(0.01), hi = Math.log10(1.2);
    function CX(x) { return PX0 + (PX1 - PX0) * x / R; }
    function CY(e) { return PY0 + (PY1 - PY0) * (1 - (Math.log10(Math.max(e, 0.01)) - lo) / (hi - lo)); }
    var AXX = PX0 - 6, AXY = PY1 + 6;                        // the axes, just outside the plotting area
    el("line", {x1: AXX, x2: AXX, y1: CY(1.2), y2: AXY, class: "ka-cv-axis"}, cv);
    el("line", {x1: AXX, x2: PX1, y1: AXY, y2: AXY, class: "ka-cv-axis"}, cv);
    [1, 0.1, 0.01].forEach(function (v) {                      // decades: grid line, tick and value
      el("line", {x1: PX0, x2: PX1, y1: CY(v), y2: CY(v), class: "ka-cv-grid"}, cv);
      el("line", {x1: AXX - 5, x2: AXX, y1: CY(v), y2: CY(v), class: "ka-cv-axis"}, cv);
      txt(AXX - 9, CY(v) + 4, v === 1 ? "1" : String(v), "ka-cv-tick", cv, "end");
    });
    [0.01, 0.1].forEach(function (d) {                         // minor ticks of the log scale
      for (var m = 2; m <= 9; m++) el("line", {x1: AXX - 3, x2: AXX, y1: CY(m * d), y2: CY(m * d), class: "ka-cv-axis"}, cv);
    });
    for (var q = 0; q <= R; q++) el("line", {x1: CX(q), x2: CX(q), y1: AXY, y2: AXY + 5, class: "ka-cv-axis"}, cv);   // round boundaries
    for (var q = 1; q <= R; q++) txt((CX(q - 1) + CX(q)) / 2, BOT - 14, q, "ka-cv-tick", cv, "middle");
    var pts = D.points.slice().sort(function (p1, p2) { return p1[0] - p2[0]; });
    var cdots = pts.map(function (p) { return el("circle", {cx: CX(p[0]), cy: CY(p[1]), r: 2.4, class: "ka-cv-pt"}, cv); });
    var bestPath = [], b0 = Infinity;
    pts.forEach(function (p) { if (p[1] < b0) { if (bestPath.length) bestPath.push([p[0], b0]); b0 = p[1]; bestPath.push([p[0], b0]); } });
    var cline = el("path", {d: "", class: "ka-cv-best"}, cv);
    var chead = el("circle", {r: 4.5, class: "ka-cv-head"}, cv);

    /* ---------------- the agent */
    var AX = 550, AYc = 266, RING = 84, CORE = 74;
    var ag = el("g", {class: "ka-agent ka-hit", "data-panel": "agent", tabindex: 0, role: "button", "aria-label": "Coding agent: details"}, svg);
    var ring = el("circle", {cx: AX, cy: AYc, r: RING, class: "ka-ring"}, ag);
    el("circle", {cx: AX, cy: AYc, r: 74, class: "ka-core"}, ag);
    var bot = el("g", {class: "ka-bot", transform: "translate(" + (AX - 23) + " " + (AYc - 62) + ") scale(0.7667)"}, ag);
    bot.innerHTML =                                         // the robot of the replay video
      '<g fill="#756C5E"><path d="m10 34h6v7h-6z"/><path d="m44 34h6v7h-6z"/><path d="m26 26h8v6h-8z"/>' +
      '<path d="m13 8h2v12h-2c-2.208 0-4-1.792-4-4v-4c0-2.208 1.792-4 4-4z"/>' +
      '<path d="m49 8h2v12h-2c-2.208 0-4-1.792-4-4v-4c0-2.208 1.792-4 4-4z" transform="matrix(-1 0 0 -1 96 28)"/></g>' +
      '<rect fill="#B4A994" height="26" rx="7" width="34" x="13" y="1"/>' +
      '<rect fill="#2A2620" height="26" rx="5" transform="matrix(0 1 -1 0 44 -16)" width="16" x="22" y="1"/>' +
      '<circle class="ka-eye" cx="24" cy="13" r="3"/><circle class="ka-eye" cx="36" cy="13" r="3"/>' +
      '<path d="m18 31h24c1.656 0 3 1.344 3 3v10c0 8.279-6.721 15-15 15-8.279 0-15-6.721-15-15v-10c0-1.656 1.344-3 3-3z" fill="#B4A994"/>' +
      '<path d="m11 52v2.5c0 1.38-1.12 2.5-2.5 2.5s-2.5-1.12-2.5-2.5v-8.5z" fill="#756C5E"/>' +
      '<path d="m11 33v19s-5 0-5-6v-10c0-1.1.45-2.1 1.17-2.83.73-.72 1.73-1.17 2.83-1.17.55 0 1 .45 1 1z" fill="#B4A994"/>' +
      '<path d="m54 46v8.5c0 1.38-1.12 2.5-2.5 2.5s-2.5-1.12-2.5-2.5v-2.5z" fill="#756C5E"/>' +
      '<path d="m54 36v10c0 6-5 6-5 6v-19c0-.55.45-1 1-1 1.1 0 2.1.45 2.83 1.17.72.73 1.17 1.73 1.17 2.83z" fill="#B4A994"/>' +
      '<rect class="ka-eye" height="10" rx="1" width="6" x="27" y="36"/>' +
      '<path class="ka-eye" d="m31 19h-2c-.553 0-1-.447-1-1s.447-1 1-1h2c.553 0 1 .447 1 1s-.447 1-1 1z"/>';
    txt(AX, AYc + 8, "Coding agent", "ka-at", ag, "middle");
    txt(AX, AYc + 28, "𝒜χ · policy π", "ka-as", ag, "middle");
    var roundTxt = txt(AX, AYc + 48, "round 0", "ka-ar", ag, "middle");

    /* ---------------- edges; each label sits on its edge in a pill */
    var pills = [];
    function edge(d, label, t) {
      var p = el("path", {d: d, class: "ka-edge"}, svg);
      if (label) pills.push({path: p, label: label, t: t || 0.5});
      return p;
    }
    function parallel(p, d) {          // the curve of p shifted by d to its right, run backwards, from x = 802 to the ring
      var L = p.getTotalLength(), pts = [], n = 60;
      for (var i = 0; i <= n; i++) {
        var sl = L * i / n, a = p.getPointAtLength(Math.max(0, sl - 1)), b = p.getPointAtLength(Math.min(L, sl + 1)),
            c = p.getPointAtLength(sl), tx = b.x - a.x, ty = b.y - a.y, m = Math.hypot(tx, ty) || 1;
        pts.push([c.x - d * ty / m, c.y + d * tx / m]);
      }
      pts.reverse();
      pts[0][0] = Math.min(pts[0][0], 802);
      while (pts.length > 2 && Math.hypot(pts[pts.length - 1][0] - AX, pts[pts.length - 1][1] - AYc) < CORE) pts.pop();
      var e = pts[pts.length - 1], f = pts[pts.length - 2], ux = e[0] - f[0], uy = e[1] - f[1], um = Math.hypot(ux, uy);
      ux /= um; uy /= um;
      var ox = e[0] - AX, oy = e[1] - AYc, B = ox * ux + oy * uy, disc = B * B - (ox * ox + oy * oy - CORE * CORE);
      if (disc >= 0) { var sx = -B - Math.sqrt(disc); pts.push([e[0] + sx * ux, e[1] + sx * uy]); }
      return "M" + pts.map(function (q) { return q[0].toFixed(1) + " " + q[1].toFixed(1); }).join(" L");
    }
    var candP = edge("M615.1 230.8 C 683 199, 740 90, 802 84", "candidates", 0.75);
    var E = {
      task: edge("M550 134 L 550 192", "reads", 0.45),
      kl: edge("M298 266 L 476 266", "reads notes"),
      ke: edge("M450 470 C 450 420, 472 362, 497.7 318.3", "reads summaries\nand artifacts", 0.5),
      write: edge("M602.3 318.3 C 628 362, 650 420, 650 470", "writes mᵣ\nand artifacts", 0.5),   // mirror of ke about x = 550
      cand: candP,
      pred: edge("M933 134 L 933 223", ""),
      err: edge("M933 285 L 933 " + (KE0 - 96), ""),
      back: edge(parallel(candP, 40), "records", 0.25),     // parallel to the candidates
      conv: edge("M933 " + (KE0 - 34) + " L 933 " + KE0, ""),
      fld: edge("M1064 " + (KE0 - 65) + " L 1100 " + (KE0 - 65), ""),          // incumbent -> its field
      sum: edge("M334 " + (KE0 + 130) + " L 298 " + (KE0 + 130), "")           // experience base -> summaries
    };
    [E.task, E.kl, E.ke, E.back].forEach(function (q) { q.classList.add("ka-in"); q.setAttribute("marker-end", "url(#ka-arr-in)"); });
    [E.cand, E.write].forEach(function (q) { q.classList.add("ka-out"); q.setAttribute("marker-end", "url(#ka-arr-out)"); });
    txt(947, 334, "ε", "ka-el ka-eps", svg);
    (function () {        // the candidates and records labels: staggered, symmetric about the middle of the pair of curves
      var pc = pills.filter(function (q) { return q.path === E.cand; })[0], pb = pills.filter(function (q) { return q.path === E.back; })[0];
      var Lc = E.cand.getTotalLength(), Lb = E.back.getTotalLength(), mc = E.cand.getPointAtLength(Lc / 2), mb = E.back.getPointAtLength(Lb / 2),
          c = E.cand.getPointAtLength(Lc * pc.t), gx = mc.x + mb.x - c.x, gy = mc.y + mb.y - c.y, best = Infinity;
      for (var sl = 0; sl <= Lb; sl += 0.5) {
        var q = E.back.getPointAtLength(sl), dd = Math.hypot(q.x - gx, q.y - gy);
        if (dd < best) { best = dd; pb.t = sl / Lb; }
      }
    })();
    var labs = el("g", {class: "ka-labs"}, svg);
    pills.forEach(function (pl) {
      var L = pl.path.getTotalLength(), pt = pl.path.getPointAtLength(L * pl.t), lines = pl.label.split("\n");
      pl.g = el("g", {class: "ka-pill" + (pl.path.classList.contains("ka-out") ? " out" : "")}, labs);
      pl.rect = el("rect", {rx: 4}, pl.g);
      pl.text = el("text", {class: "ka-el", "text-anchor": "middle"}, pl.g);
      lines.forEach(function (ln, j) {
        var ts = el("tspan", {x: pt.x, y: pt.y + 4 + (j - (lines.length - 1) / 2) * 15}, pl.text);
        ts.textContent = ln;
      });
      pl.pt = pt; pl.n = lines.length;
    });
    function sizePills() {
      pills.forEach(function (pl) {
        var w = 0;
        try { w = pl.text.getBBox().width; } catch (e) { w = 0; }
        if (!w) w = Math.max.apply(null, pl.label.split("\n").map(function (x) { return x.length; })) * 6.6;
        pl.w = w;
      });
      pills.forEach(function (pl) {
        var w = pl.w;
        if (pl.path === E.ke || pl.path === E.write) {               // the mirrored pair: one width, so the boxes mirror too
          pills.forEach(function (o) { if (o.path === E.ke || o.path === E.write) w = Math.max(w, o.w); });
        }
        var h = 18 + (pl.n - 1) * 15;
        pl.rect.setAttribute("x", pl.pt.x - w / 2 - 8); pl.rect.setAttribute("width", w + 16);
        pl.rect.setAttribute("y", pl.pt.y - h / 2); pl.rect.setAttribute("height", h);
      });
    }
    sizePills();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(sizePills);
    var parts = el("g", {}, svg);
    function particle(path, f, cls) {
      if (f <= 0 || f >= 1) return;
      var L = path.getTotalLength(), pt = path.getPointAtLength(L * f);
      el("circle", {cx: pt.x, cy: pt.y, r: 4.2, class: "ka-p " + (cls || "")}, parts);
    }
    function burst(path, a0, b, t, n, cls) {
      for (var j = 0; j < n; j++) particle(path, (t - a0 - j * (b - a0) * 0.12) / ((b - a0) * 0.6), cls);
    }

    var log = document.getElementById("ka-log");
    var shownN = -1;

    function summaryLi(k) {
      var li = document.createElement("li");
      li.innerHTML = "<div class='ka-m'><b>m" + sub(k) + "</b><span>" + esc(D.rounds[k].head) + "</span></div>";
      return li;
    }
    function rebuildLog(n) {                                  // seeking: the window shows the last 3 at once
      log.innerHTML = "";
      for (var k = n - 1; k >= Math.max(0, n - 3); k--) log.appendChild(summaryLi(k));
      fitLog();
    }
    function addSummary(k) {          // playing: the new summary slides in on top, the others move down, the oldest fades out
      [].forEach.call(log.querySelectorAll("li.out"), function (o) { o.remove(); });
      var olds = [].slice.call(log.children), tops = olds.map(function (o) { return o.offsetTop; });
      var li = summaryLi(k);
      li.classList.add("enter");
      log.insertBefore(li, log.firstChild);
      olds.forEach(function (o, i) {
        o.style.visibility = "";
        o.style.transition = "none";
        o.style.transform = "translateY(" + (tops[i] - o.offsetTop) + "px)";
      });
      void log.offsetHeight;
      olds.forEach(function (o) { o.style.transition = ""; o.style.transform = ""; });
      if (olds.length >= 3) olds[olds.length - 1].classList.add("out");   // after the reflow, so it fades
      li.classList.remove("enter");
      setTimeout(function () { [].forEach.call(log.querySelectorAll("li.out"), function (o) { o.remove(); }); fitLog(); }, 650);
    }
    function fitLog() {
      var lim = log.getBoundingClientRect().bottom + 0.5;
      [].forEach.call(log.children, function (li) { li.style.visibility = li.getBoundingClientRect().bottom > lim ? "hidden" : ""; });
    }
    var fieldCard = document.querySelector(".pfh-ka-field"), mwin = document.querySelector(".pfh-ka-mwin"), DS = 0.83;
    function place(card, x, y, w, h) {    // a card on its slot of the diagram, laid out at DS px per unit and scaled with it
      var b = svg.getBoundingClientRect(), g = svg.parentNode.getBoundingClientRect(), sc = b.width / VBW;
      if (!sc) return;
      card.style.left = (b.left - g.left + (x - VBX) * sc) + "px"; card.style.top = (b.top - g.top + (y - VBY) * sc) + "px";
      card.style.width = (w * DS) + "px"; card.style.height = (h * DS) + "px";
      card.style.transform = "scale(" + (sc / DS) + ")";
    }
    function syncSide() {
      place(mwin, 36, KE0, 262, BOT - KE0);                   // below 𝒦ₗ, beside 𝒦ₑ
      place(fieldCard, 1100, 72, 262, KE0 - 106);              // beside the incumbent, as tall as 𝒦ₗ
      var ctlLeft = document.querySelector(".pfh-ka-ctl-left");
      if (window.matchMedia("(min-width: 561px)").matches) {   // the seek bar starts under the left edge of 𝒦ₑ
        var kr = ke.querySelector("rect").getBoundingClientRect(), cr = ctlLeft.getBoundingClientRect();
        ctlLeft.style.width = Math.max(0, kr.left - cr.left - 10) + "px";
      } else ctlLeft.style.width = "";
      fitLog();
    }
    syncSide();
    if ("ResizeObserver" in window) new ResizeObserver(syncSide).observe(svg); else window.addEventListener("resize", syncSide);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(syncSide);

    /* ---------------- the field of the incumbent: vorticity at the last time, one image per round */
    var F = D.fields, fimg = document.getElementById("ka-fimg"), fieldKey = null, preload = [];
    Object.keys(F.img).forEach(function (k) { var im = new Image(); im.src = F.img[k]; preload.push(im); });
    document.querySelector(".pfh-ka-cbar > i").style.background = "linear-gradient(90deg," + F.stops.map(function (st) {
      return st[1] + " " + ((st[0] + 1) * 50) + "%"; }).join(",") + ")";
    document.getElementById("ka-vmax").textContent = F.vmax;
    document.getElementById("ka-vmin").textContent = "−" + F.vmax;
    function drawField(key) {
      if (key === fieldKey) return;
      fieldKey = key;
      fimg.src = F.img[key];
      var w = fimg.parentNode; w.classList.remove("flash"); void w.offsetWidth; w.classList.add("flash");
    }
    function draw(T) {
      var r = Math.min(R, Math.floor(T / ROUND_MS)), f = (T - r * ROUND_MS) / ROUND_MS, done = T >= (R + 1) * ROUND_MS;
      if (done) { r = R; f = 1; }
      while (parts.firstChild) parts.removeChild(parts.firstChild);
      var cited = {}, now = {};
      for (var qq = 1; qq <= r; qq++) D.rounds[qq].cited.forEach(function (n) { if (qq < r || f > 0.08) cited[n] = 1; });
      if (r > 0 && f < 0.5) D.rounds[r].cited.forEach(function (n) { now[n] = 1; });
      dots.forEach(function (d, ii) { d.setAttribute("class", "ka-note" + (now[ii + 1] ? " now" : cited[ii + 1] ? " seen" : "")); });
      tiles.forEach(function (tl, kk) { tl.setAttribute("class", "ka-tile" + (kk < r || (kk === r && f > 0.86) ? " on" : "") + (kk === r && f > 0.86 && !done ? " new" : "")); });
      if (!done) {
        if (r > 0) { burst(E.task, 0, 0.22, f, 1, "k"); burst(E.kl, 0, 0.25, f, 3, "k"); burst(E.ke, 0.03, 0.26, f, 2, "k"); }
        burst(E.cand, 0.26, 0.5, f, 4, "a");
        burst(E.pred, 0.5, 0.6, f, 2, "a");
        burst(E.err, 0.6, 0.7, f, 1, "e");
        burst(E.conv, 0.68, 0.76, f, 1, "e");
        burst(E.back, 0.66, 0.84, f, 3, "e");
        burst(E.fld, 0.70, 0.80, f, 1, "e");
        burst(E.sum, 0.88, 0.98, f, 1, "k");
        burst(E.write, 0.84, 0.98, f, 2, "k");
      }
      ring.style.transform = "rotate(" + (T / 40) % 360 + "deg)";
      ag.classList.toggle("busy", f > 0.2 && f < 0.86 && !done);
      roundTxt.textContent = "round " + r;
      var best = f >= 0.7 || done ? D.rounds[r].best : (r > 0 ? D.rounds[r - 1].best : null);
      incV.textContent = best == null ? "" : fmt(best);
      var fr = f >= 0.7 || done ? r : r - 1;                  // the incumbent's round, once it is scored
      if (fr >= 0) {
        var bk = "b" + String(fr).padStart(2, "0");
        drawField(F.alias[bk] || bk);
      } else {
        drawField("b00");
      }
      // artifacts
      var sr = f > 0.86 || done ? r : r - 1;
      var nC = sr >= 0 ? cumC[sr] : 0, nE = sr >= 0 ? cumE[sr] : 0;
      artP.textContent = nC + " candidate programs";
      artE.textContent = nE + " evaluation records";
      var lit = Math.round(artTiles.length * nC / cumC[R]);
      artTiles.forEach(function (t, ii) { t.setAttribute("class", "ka-art-t" + (ii < lit ? " on" : "")); });
      // convergence: the evaluations of a round appear while its candidates are scored
      var xr = r === 0 ? (f >= 0.6 ? 0 : -1) : (r - 1) + clamp((f - 0.45) / 0.35);
      if (done) xr = R;
      cdots.forEach(function (d, ii) { d.style.opacity = pts[ii][0] <= xr + 1e-9 ? 1 : 0; });
      var cut = bestPath.filter(function (p) { return p[0] <= xr + 1e-9; });
      if (cut.length) {
        var end = [Math.max(xr, cut[cut.length - 1][0]), cut[cut.length - 1][1]];
        cline.setAttribute("d", "M" + cut.concat([end]).map(function (p) { return CX(p[0]).toFixed(1) + "," + CY(p[1]).toFixed(1); }).join("L"));
        chead.setAttribute("cx", CX(end[0])); chead.setAttribute("cy", CY(end[1])); chead.style.opacity = 1;
      } else { cline.setAttribute("d", ""); chead.style.opacity = 0; }
      var nDone = r + (f > 0.86 || done ? 1 : 0);             // summaries written so far; the window shows the last 3
      if (nDone !== shownN) {
        var grow = nDone === shownN + 1 && playing && !scrub;
        shownN = nDone;
        if (grow) addSummary(nDone - 1); else rebuildLog(nDone);
      }
    }

    /* ---------------- detail panels */
    var panel = document.getElementById("ka-panel"), body = document.getElementById("ka-panel-body");
    function tpl(id) { var t = document.getElementById("ka-tpl-" + id); return t ? t.innerHTML : ""; }
    function literaturePanel() {
      var years = {};
      Object.keys(D.notes).forEach(function (n) { var y = D.notes[n].y; years[y] = (years[y] || 0) + 1; });
      var ys = Object.keys(years).sort();
      var mx = Math.max.apply(null, ys.map(function (y) { return years[y]; }));
      var hist = ys.map(function (y) {
        return "<div class='kp-yr' title='" + years[y] + " notes from " + y + "'><b>" + years[y] + "</b><i style='height:" + (100 * years[y] / mx) + "%'></i><span>" + y + "</span></div>";
      }).join("");
      return "<p class='kp-kicker'>Literature base</p><h3 id='ka-panel-title'>130 notes on PINN methods</h3>" +
        "<p>Each note distills one paper for the agent: a summary of the mechanism and the claim, the failure of PINN training that the method addresses, the method with an implementation in JAX that a round can adapt, and the reported results. Every note is checked for its format and for a correct account of the technique.</p>" +
        "<div class='kp-funnel'><div><b>3,490</b><span>papers found</span></div><div><b>143</b><span>after screening</span></div><div><b>130</b><span>after full-text review</span></div></div>" +
        "<div class='kp-hist'>" + hist + "</div><p class='kp-cap'>Notes by year of publication</p>" +
        "<div class='kp-tools'><input type='search' id='kp-q' placeholder='Search the 130 notes' aria-label='Search the notes'>" +
        "<label><input type='checkbox' id='kp-cited'> cited in this run (" + Object.keys(citedAll).length + ")</label></div>" +
        "<ol class='kp-list' id='kp-list'></ol>";
    }
    function fillNotes() {
      var q = (document.getElementById("kp-q").value || "").toLowerCase(), only = document.getElementById("kp-cited").checked;
      var ids = Object.keys(D.notes).map(Number).sort(function (x, y) { return x - y; });
      document.getElementById("kp-list").innerHTML = ids.filter(function (n) {
        var t = D.notes[n]; if (only && !citedAll[n]) return false;
        return !q || (t.t + " " + t.a + " " + t.v + " " + t.s + " " + pad(n)).toLowerCase().indexOf(q) >= 0;
      }).map(function (n) {
        var t = D.notes[n];
        return "<li class='" + (citedAll[n] ? "c" : "") + "'><button type='button'><b>" + pad(n) + "</b><span>" + esc(t.t) + "</span><em>" + t.y + "</em></button>" +
          "<div class='kp-more'><p class='kp-meta'>" + esc(t.a) + ", " + t.y + (t.v ? " · " + esc(t.v) : "") +
          (citedAll[n] ? " · cited in " + citedAll[n] + (citedAll[n] > 1 ? " rounds" : " round") + " of this run" : "") + "</p>" +
          "<p>" + esc(t.s) + "</p>" + (t.g ? "<a href='" + esc(t.g) + "' target='_blank' rel='noopener'>Code repository &#8599;</a>" : "") + "</div></li>";
      }).join("") || "<li class='kp-none'>No note matches.</li>";
    }
    function experiencePanel() {
      var bestR = D.rounds.filter(function (q) { return q.best === D.rounds[R].best; })[0].r;   // the round that set the final incumbent
      return tpl("ke") + "<ol class='kp-rounds'>" + D.rounds.map(function (r) {
        return "<li" + (r.r === bestR ? " class='best'" : "") + "><b>m" + sub(r.r) + "</b><span>" + esc(r.head) + "</span><em>" + fmt(r.best) + "</em>" +
          "<small>" + (r.cand || 0) + " programs · " + (r.evals || 0) + " evaluations" + (r.cited.length ? " · cites " + r.cited.map(pad).join(", ") : "") + "</small></li>";
      }).join("") + "</ol>";
    }
    var current = null;
    function placePanel(g) {             // beside the box that was clicked, a little shorter than the stage
      var wideNow = window.matchMedia("(min-width: 961px)").matches;
      if (!wideNow || !g) {
        panel.style.left = panel.style.right = panel.style.top = panel.style.bottom = panel.style.width = panel.style.maxHeight = "";
        panel.classList.remove("from-left");
        return;
      }
      var sr = stage.getBoundingClientRect(), r = g.getBoundingClientRect(), W = Math.min(440, sr.width - 16), gap = 12;
      var toRight = (r.left + r.right) / 2 < sr.left + sr.width / 2, x;
      x = toRight ? Math.min(r.right - sr.left + gap, sr.width - W - 8) : Math.max(r.left - sr.left - gap - W, 8);
      panel.style.left = x + "px"; panel.style.right = "auto"; panel.style.width = W + "px";
      panel.style.top = "12px"; panel.style.bottom = "auto";        // as tall as its text, at most the stage less 24 px
      panel.style.maxHeight = (sr.height - 24) + "px";
      panel.classList.toggle("from-left", !toRight);
    }
    function open(key, g) {
      if (current === key && !panel.hidden) { close(); return; }
      current = key;
      body.innerHTML = key === "kl" ? literaturePanel() : key === "ke" ? experiencePanel() : tpl(key);
      placePanel(g);
      panel.hidden = false;
      stage.classList.add("ka-open");
      stage.querySelectorAll(".ka-hit, .pfh-ka-hlabel").forEach(function (g) { g.classList.toggle("sel", g.getAttribute("data-panel") === key); });
      katexIn(body);
      if (key === "kl") {
        fillNotes();
        body.querySelector("#kp-q").addEventListener("input", fillNotes);
        body.querySelector("#kp-cited").addEventListener("change", fillNotes);
        body.querySelector("#kp-list").addEventListener("click", function (e) {
          var b = e.target.closest("button"); if (b) b.parentNode.classList.toggle("open");
        });
      }
      body.scrollTop = 0;
    }
    function close() {
      panel.hidden = true; current = null; stage.classList.remove("ka-open");
      stage.querySelectorAll(".ka-hit, .pfh-ka-hlabel").forEach(function (g) { g.classList.remove("sel"); });
    }
    stage.querySelectorAll(".ka-hit, .pfh-ka-hlabel").forEach(function (g) {
      g.addEventListener("click", function (e) { e.stopPropagation(); open(g.getAttribute("data-panel"), g); });
      g.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(g.getAttribute("data-panel"), g); } });
    });
    panel.querySelector(".pfh-ka-close").addEventListener("click", close);
    document.addEventListener("keydown", function (e) { if (e.key === "Escape" && !panel.hidden) close(); });
    document.addEventListener("click", function (e) {         // a click anywhere outside the panel closes it
      if (!panel.hidden && !panel.contains(e.target)) close();
    });

    /* ---------------- the evidence bars and the funnel */
    var bars = document.querySelectorAll("#ka-bars .pfh-bar");
    function drawBars(t) {
      bars.forEach(function (b, ii) {
        var v = +b.getAttribute("data-v"), w = (Math.log10(v) + 2) / 2;
        b.querySelector("i").style.width = (100 * w * clamp((t - ii * 0.18) / 0.9)) + "%";
      });
    }

    /* ---------------- playback: play, pause, restart, and a seek bar over the eleven rounds */
    // the timeline runs over the agent's rounds 1-10; round 0 (the standard PINN) is its starting state
    var TSTART = ROUND_MS, TEND = (R + 1) * ROUND_MS, T = TSTART, playing = !reduce, holdT = 0, scrub = false, last = null, speed = 1;
    var seek = document.getElementById("ka-seek"), playBtn = document.getElementById("ka-play"),
        tLab = document.getElementById("ka-time"), ticks = document.getElementById("ka-ticks");
    seek.min = TSTART; seek.max = TEND;
    for (var kt = 1; kt <= R; kt++) {                         // round numbers in the middle of their rounds
      var tk = document.createElement("span"); tk.textContent = kt; tk.style.left = ((kt - 0.5) / R * 100) + "%";
      ticks.appendChild(tk);
    }
    for (var kb = 0; kb <= R; kb++) {                         // small marks at the boundaries between rounds
      var mk = document.createElement("i"); mk.style.left = (kb / R * 100) + "%";
      ticks.appendChild(mk);
    }
    function render(t) {
      draw(t);
      seek.value = t; seek.style.setProperty("--p", (100 * (t - TSTART) / (TEND - TSTART)).toFixed(2) + "%");
      tLab.textContent = "round " + Math.min(R, Math.floor(t / ROUND_MS)) + "/" + R;
    }
    function setPlaying(on) {
      playing = on;
      playBtn.classList.toggle("paused", !on);
      playBtn.setAttribute("aria-label", on ? "Pause" : "Play");
      if (on && T >= TEND) { T = TSTART; holdT = 0; }
    }
    playBtn.addEventListener("click", function () { setPlaying(!playing); render(T); });
    document.getElementById("ka-restart").addEventListener("click", function () { T = TSTART; holdT = 0; setPlaying(true); render(T); });
    // the download menu (MP4 or GIF) closes on a choice, a click elsewhere or Escape
    var dl = document.getElementById("ka-dl");
    if (dl) {
      dl.addEventListener("click", function (e) { if (e.target.closest("a")) dl.open = false; });
      document.addEventListener("click", function (e) { if (dl.open && !dl.contains(e.target)) dl.open = false; });
      document.addEventListener("keydown", function (e) { if (e.key === "Escape" && dl.open) { dl.open = false; dl.querySelector("summary").focus(); } });
    }
    [].forEach.call(document.querySelectorAll(".pfh-ka-speed button"), function (b, _, all) {
      b.addEventListener("click", function () {
        speed = +b.getAttribute("data-speed");
        [].forEach.call(all, function (o) { o.classList.toggle("on", o === b); o.setAttribute("aria-pressed", o === b ? "true" : "false"); });
      });
    });
    seek.addEventListener("input", function () { scrub = true; T = +seek.value; holdT = 0; render(T); });
    ["change", "pointerup", "touchend", "blur"].forEach(function (ev) { seek.addEventListener(ev, function () { scrub = false; }); });
    setPlaying(playing);
    if (reduce) { T = TEND; drawBars(9); }
    render(T);

    var proofT = null;
    var proof = document.querySelector(".pfh-ka-proof");
    var vis = {stage: false, proof: false};
    function watch(node, key, th) {
      if (!node) return;
      if (!("IntersectionObserver" in window)) { vis[key] = true; return; }
      new IntersectionObserver(function (es) { es.forEach(function (e) { vis[key] = e.isIntersecting; }); }, {threshold: th}).observe(node);
    }
    watch(stage, "stage", 0.25); watch(proof, "proof", 0.3);
    function frame(now) {
      requestAnimationFrame(frame);
      var dt = last === null ? 0 : Math.min(now - last, 100);
      last = now;
      if (vis.stage && playing && !scrub) {                  // the run advances only while it is in view
        if (T < TEND) T = Math.min(TEND, T + dt * speed);
        else if ((holdT += dt) > HOLD_MS) { T = TSTART; holdT = 0; }
        render(T);
      }
      if (reduce) return;
      if (vis.proof) { if (proofT === null) proofT = now; drawBars((now - proofT) / 1000); }
    }
    requestAnimationFrame(frame);
  });
})();
