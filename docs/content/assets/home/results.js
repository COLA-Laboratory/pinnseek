/* The four interactive figures of the Results on the landing page: generalization, applications, inside a
   discovery and the ablations. Data in assets/home/res (written by website/tools/results_web.py from the source
   data of Figs. 2 to 5): charts with ECharts, fields drawn on canvases from uint8 frames. Each figure starts when
   it comes near the viewport and keeps its paper image until then, or when a script fails. */
(function () {
  "use strict";
  var UI = '"Helvetica Web", Helvetica, Arial, sans-serif';
  var INK = "#1d1a16", INK2 = "#4a443c", INK3 = "#7b7368", GRID = "#ece5d9", LINE = "#e6dfd3";
  var EMBER = "#e8743b", EMBER2 = "#c85a26", EMBER3 = "#a4471c", GLOW = "#f2a65a", STEEL = "#7f9cb8", SAND = "#c9a46e",
      PALE = "#f3c49c";
  var REDUCE = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

  /* ---------------------------------------------------------------- helpers */
  function ready(fn) { if (document.readyState !== "loading") fn(); else document.addEventListener("DOMContentLoaded", fn); }
  function h(tag, attrs, kids) {
    var e = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      var v = attrs[k];
      if (k === "text") e.textContent = v; else if (k === "html") e.innerHTML = v; else e.setAttribute(k, v);
    });
    (kids || []).forEach(function (c) { if (c) e.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return e;
  }
  function sup(n) {
    var m = {"-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"};
    return String(n).split("").map(function (c) { return m[c] || c; }).join("");
  }
  function logLabel(v) {
    var e = Math.round(Math.log10(v));
    if (Math.abs(Math.log10(v) - e) > 1e-6) return "";
    return e === 0 ? "1" : e === 1 ? "10" : "10" + sup(e);
  }
  function fmt(v) {
    if (v === null || v === undefined || v !== v) return "–";
    var a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 0.001 && a < 1000) return String(Number(v.toPrecision(3)));
    var e = Math.floor(Math.log10(a));
    return (v / Math.pow(10, e)).toPrecision(3) + "×10" + sup(e);
  }
  function minus(s) { return String(s).replace(/^-/, "−"); }
  function decades(vals) {
    var ok = vals.filter(function (v) { return v > 0; });
    return [Math.pow(10, Math.floor(Math.log10(Math.min.apply(null, ok)))),
            Math.pow(10, Math.ceil(Math.log10(Math.max.apply(null, ok))))];
  }
  var TIP = {backgroundColor: "#fffdf9", borderColor: LINE, borderWidth: 1, padding: [9, 12],
             textStyle: {fontFamily: UI, fontSize: 12.5, color: INK},
             extraCssText: "border-radius: 10px; box-shadow: 0 12px 30px rgba(60, 40, 20, .14);"};
  function dotH(c, ring, sq) {
    return "<span style='display:inline-block;width:8px;height:8px;border-radius:" + (sq ? "2px" : "50%") + ";margin-right:7px;vertical-align:0;" +
           (ring ? "border:2px solid " + c + ";width:5px;height:5px" : "background:" + c) + "'></span>";
  }
  function row(l, r, strong) {
    return "<div style='display:flex;justify-content:space-between;gap:18px" + (strong ? ";font-weight:600" : "") + "'><span>" +
           l + "</span><span>" + r + "</span></div>";
  }
  function muted(s) { return "<span style='color:" + INK3 + "'>" + s + "</span>"; }
  // every axis of the charts shows a solid line (author, 2026-10-23)
  function logAxis(o) {
    return Object.assign({type: "log", logBase: 10, axisLabel: {formatter: logLabel, color: INK3, fontSize: 11.5},
      splitLine: {lineStyle: {color: GRID, type: "dashed"}}, axisLine: {show: true, lineStyle: {color: "#cfc6b9"}}, axisTick: {show: false},
      nameTextStyle: {fontFamily: UI, fontSize: 11.5, color: INK3}}, o || {});
  }
  function valAxis(o) {
    return Object.assign({type: "value", axisLabel: {color: INK3, fontSize: 11.5, formatter: function (v) { return minus(fmt(v)); }},
      splitLine: {lineStyle: {color: GRID, type: "dashed"}}, axisLine: {show: true, lineStyle: {color: "#cfc6b9"}}, axisTick: {show: false},
      nameTextStyle: {fontFamily: UI, fontSize: 11.5, color: INK3}}, o || {});
  }
  var L2 = {rich: {i: {fontStyle: "italic", fontFamily: UI, fontSize: 11.5, color: INK3}}};
  // variables in italic, in HTML (<i>) and in ECharts rich text ({i|...}): Re, α, and t or L before " =" in the names of cases
  var VAR = /\bRe\b|α|\b[tL](?= =)/g;
  function vh(s) { return String(s).replace(VAR, function (m) { return "<i>" + m + "</i>"; }); }
  function vr(s) { return String(s).replace(VAR, function (m) { return "{i|" + m + "}"; }); }
  var V11 = {fontFamily: UI, fontSize: 11, color: INK3, rich: {i: {fontFamily: UI, fontSize: 11, fontStyle: "italic", color: INK3}}};
  var COEF = {fontFamily: UI, fontSize: 11, color: INK3, rich: {     // an axis name "{i|C}{s|l}": an italic C with a subscript
    i: {fontFamily: UI, fontSize: 12, fontStyle: "italic", color: INK3, lineHeight: 16, verticalAlign: "bottom"},
    s: {fontFamily: UI, fontSize: 8.5, color: INK3, lineHeight: 10, verticalAlign: "bottom", padding: [0, 0, -2, 1]}}};

  function seg(labels, onPick, aria, start, glide) {
    var box = h("div", {"class": "pfh-seg" + (glide ? " pfh-seg-glide" : ""), role: "group", "aria-label": aria});
    var thumb = glide ? h("span", {"class": "pfh-seg-thumb", "aria-hidden": "true"}) : null;
    if (thumb) box.appendChild(thumb);
    function slide() {                                         // the white pill glides to the pressed button
      var b = btns.filter(function (x) { return x.getAttribute("aria-pressed") === "true"; })[0];
      if (!b || !b.offsetWidth) return;
      thumb.style.width = b.offsetWidth + "px";
      thumb.style.height = b.offsetHeight + "px";
      thumb.style.transform = "translate(" + b.offsetLeft + "px, " + b.offsetTop + "px)";
    }
    var btns = labels.map(function (l, i) {
      var b = h("button", {type: "button", "aria-pressed": i === (start || 0) ? "true" : "false", html: l});
      b.addEventListener("click", function () {
        if (b.getAttribute("aria-pressed") === "true") return;
        btns.forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
        if (thumb) slide();
        onPick(i);
      });
      box.appendChild(b);
      return b;
    });
    if (thumb) {                                               // placed without motion first, after the fonts and on resizes
      requestAnimationFrame(function () { slide(); requestAnimationFrame(function () { box.classList.add("is-ready"); }); });
      if (window.ResizeObserver) new ResizeObserver(slide).observe(box);
    }
    box.pick = function (i) { btns[i].click(); };
    return box;
  }
  function legend(items) {
    return h("div", {"class": "pfh-fig-legend"}, items.map(function (it) {
      return h("span", {}, [h("i", {"class": it.kind || "", style: "--c:" + it.color}), it.label]);
    }));
  }
  function fade(el) { el.classList.remove("pfh-fade"); void el.offsetWidth; el.classList.add("pfh-fade"); }
  function chartIn(el) {
    var c = echarts.init(el, null, {renderer: "svg"});
    window.addEventListener("resize", function () { c.resize(); });
    return c;
  }

  var cache = {};
  function getJSON(base, f) {
    return cache[f] || (cache[f] = fetch(base + f).then(function (r) { if (!r.ok) throw new Error(f); return r.json(); }));
  }
  function getBin(base, f) {
    return cache[f] || (cache[f] = fetch(base + f).then(function (r) { if (!r.ok) throw new Error(f); return r.arrayBuffer(); })
      .then(function (b) { return new Uint8Array(b); }));
  }

  /* ---------------------------------------------------------------- colour maps and fields */
  function ramp(stops) {
    var n = 1024, lut = new Uint8ClampedArray(n * 3);
    var s = stops.map(function (p) {
      return [p[0], parseInt(p[1].slice(1, 3), 16), parseInt(p[1].slice(3, 5), 16), parseInt(p[1].slice(5, 7), 16)];
    });
    for (var i = 0; i < n; i++) {
      var t = i / (n - 1), k = 1;
      while (k < s.length - 1 && t > s[k][0]) k++;
      var a = s[k - 1], b = s[k], f = Math.min(1, Math.max(0, (t - a[0]) / ((b[0] - a[0]) || 1)));
      for (var c = 1; c < 4; c++) lut[i * 3 + c - 1] = a[c] + (b[c] - a[c]) * f;
    }
    return {lut: lut, css: "linear-gradient(90deg," + stops.map(function (p) { return p[1] + " " + (p[0] * 100) + "%"; }).join(",") + ")"};
  }
  // diverging in the page's two field colours (the steel blue and the ember of the section backgrounds), paper in
  // the middle; sequential from paper to deep ember for errors; the smoke plume in ember over the map
  var DIV = ramp([[0, "#22476f"], [0.18, "#3b6fa3"], [0.36, "#9cbcd8"], [0.5, "#f7f2eb"], [0.64, "#f2b38a"], [0.82, "#d9622f"], [1, "#89290b"]]);
  var SEQ = ramp([[0, "#fbf8f3"], [0.3, "#f6d0b0"], [0.6, "#ec8f55"], [0.85, "#c4521f"], [1, "#7a2a0d"]]);
  var SMOKE = ramp([[0, "#f4a77c"], [0.5, "#d9552a"], [1, "#86240b"]]);
  function lin(lo, hi) { return function (v) { return (v - lo) / (hi - lo); }; }
  function two(lo, c, hi) { return function (v) { return v < c ? 0.5 * (v - lo) / (c - lo) : 0.5 + 0.5 * (v - c) / (hi - c); }; }

  function table(e) {
    if (e.table) return e.table;
    var t = new Float32Array(256);
    for (var q = 0; q < 255; q++) { var f = q / 254; if (e.sqrt) f = f * f; t[q] = e.lo + f * (e.hi - e.lo); }
    t[255] = NaN;
    return (e.table = t);
  }
  function frame(buf, e, k) { var n = e.w * e.h; return buf.subarray(e.off + (k || 0) * n, e.off + ((k || 0) + 1) * n); }
  var W4 = [0, 0, 0, 0], V4 = [0, 0, 0, 0];
  function bil(q, T, a, b, c, d, tx, ty) {
    var va = T[q[a]], vb = T[q[b]], vc = T[q[c]], vd = T[q[d]];
    if (va !== va || vb !== vb || vc !== vc || vd !== vd) {            // next to a hole: the nearest value
      W4[0] = (1 - tx) * (1 - ty); W4[1] = tx * (1 - ty); W4[2] = (1 - tx) * ty; W4[3] = tx * ty;
      V4[0] = va; V4[1] = vb; V4[2] = vc; V4[3] = vd;
      var best = NaN, bw = -1;
      for (var k = 0; k < 4; k++) if (V4[k] === V4[k] && W4[k] > bw) { bw = W4[k]; best = V4[k]; }
      return best;
    }
    return (va * (1 - tx) + vb * tx) * (1 - ty) + (vc * (1 - tx) + vd * tx) * ty;
  }
  /* A field (w x h values, rows from the top, grid points on the edges of the extent) drawn over the whole canvas,
     sampled bilinearly in the values. o: {e, q, cm, norm, diff: {e, q} for |q - diff|, mask: {e, q} a signed
     distance with the solid where > 0, span: data units across, alpha: value -> 0..1, dpr, after(ctx, W, H)} */
  function drawField(canvas, o) {
    var r = canvas.getBoundingClientRect();
    if (!r.width) return;
    var dpr = Math.min(window.devicePixelRatio || 1, o.dpr || 2);
    var W = Math.max(2, Math.round(r.width * dpr)), H = Math.max(2, Math.round(r.height * dpr));
    if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
    var ctx = canvas.getContext("2d"), img = ctx.createImageData(W, H), px = img.data;
    var e = o.e, w = e.w, hh = e.h, q = o.q, T = table(e), lut = o.cm.lut, norm = o.norm, alpha = o.alpha;
    var dq = o.diff && o.diff.q, DT = o.diff && table(o.diff.e);
    var mq = o.mask && o.mask.q, MT = o.mask && table(o.mask.e), edge = o.mask ? 1.3 * o.span / W : 1;
    var sx = (w - 1) / Math.max(1, W - 1), sy = (hh - 1) / Math.max(1, H - 1);
    for (var j = 0; j < H; j++) {
      var fy = j * sy, y0 = fy | 0, y1 = y0 + 1 < hh ? y0 + 1 : y0, ty = fy - y0, r0 = y0 * w, r1 = y1 * w;
      for (var i = 0; i < W; i++) {
        var fx = i * sx, x0 = fx | 0, x1 = x0 + 1 < w ? x0 + 1 : x0, tx = fx - x0, p = (j * W + i) * 4;
        var v = bil(q, T, r0 + x0, r0 + x1, r1 + x0, r1 + x1, tx, ty);
        if (dq) v = Math.abs(v - bil(dq, DT, r0 + x0, r0 + x1, r1 + x0, r1 + x1, tx, ty));
        if (v !== v) { px[p + 3] = 0; continue; }
        var a = 255;
        if (mq) {
          var s = 0.5 - bil(mq, MT, r0 + x0, r0 + x1, r1 + x0, r1 + x1, tx, ty) / edge;
          if (s <= 0) { px[p + 3] = 0; continue; }
          if (s < 1) a = 255 * s;
        }
        if (alpha) a *= alpha(v);
        var t = norm(v), li = ((t < 0 ? 0 : t > 1 ? 1 : t) * 1023 | 0) * 3;
        px[p] = lut[li]; px[p + 1] = lut[li + 1]; px[p + 2] = lut[li + 2]; px[p + 3] = a;
      }
    }
    ctx.putImageData(img, 0, 0);
    if (o.after) o.after(ctx, W, H);
  }
  function mapBox(label, ratio, cls) {
    var cv = h("canvas", {"aria-hidden": "true"});
    var box = h("div", {"class": "pfh-map" + (cls ? " " + cls : ""), style: "aspect-ratio:" + ratio}, [cv, label ? h("span", {"class": "pfh-map-tag", text: label}) : null]);
    box.cv = cv;
    return box;
  }
  function cbar(cm, lo, hi, title, mid) {
    return h("div", {"class": "pfh-cbar"}, [h("span", {"class": "t", html: title}), h("span", {text: lo}),
      h("i", {style: "background:" + cm.css}), mid ? h("span", {"class": "mid", text: mid}) : null, h("span", {text: hi})]);
  }
  function onResize(el, fn) {
    var last = 0;
    if (window.ResizeObserver) new ResizeObserver(function () {
      var w = el.getBoundingClientRect().width;
      if (Math.abs(w - last) > 2) { last = w; fn(); }
    }).observe(el);
  }
  // a time line with play and pause; plays while in view unless the reader prefers less motion
  function player(n, label, onFrame, ms, watch) {
    var playing = false, k = 0, timer = null;
    var btn = h("button", {type: "button", "class": "pfh-play", "aria-label": "Play"});
    var range = h("input", {type: "range", min: "0", max: String(n - 1), step: "1", value: "0", "aria-label": "Time"});
    var out = h("output", {text: label(0)});
    var box = h("div", {"class": "pfh-player"}, [btn, range, out]);
    function set(i) { k = i; range.value = String(i); out.textContent = label(i); onFrame(i); }
    function tick() { set((k + 1) % n); timer = setTimeout(tick, k === n - 1 ? ms * 6 : ms); }
    function play(on) {
      playing = on;
      btn.classList.toggle("on", on);
      btn.setAttribute("aria-label", on ? "Pause" : "Play");
      clearTimeout(timer);
      if (on) timer = setTimeout(tick, ms);
    }
    btn.addEventListener("click", function () { play(!playing); box.user = true; });
    range.addEventListener("input", function () { play(false); box.user = true; set(+range.value); });
    box.set = set;
    box.play = play;
    if (watch && window.IntersectionObserver) {
      new IntersectionObserver(function (es) {
        es.forEach(function (en) {
          if (!en.isIntersecting) play(false);
          else if (!REDUCE && !box.user && !playing) play(true);
        });
      }, {threshold: 0.35}).observe(watch);
    }
    return box;
  }

  /* ---------------------------------------------------------------- generalization */
  // the chart alone, the same size for every problem; faint bands of alternating colour tell the cases apart. On a new
  // problem the markers and ranges glide to their new places, case by case, instead of the chart being redrawn
  function initGen(root, base) {
    var COL = {a1: EMBER, a5: GLOW, a3: EMBER3, a4: INK2};
    return getJSON(base, "gen.json").then(function (D) {
      var cur = 0;
      root.innerHTML = "";
      var segBox = seg(D.rows.map(function (x) { return x.name; }), function (i) { cur = i; draw(); fade(cap); }, "Problem", 0, true);
      root.appendChild(h("div", {"class": "pfh-fig-top"}, [segBox]));
      root.appendChild(legend(D.arms.map(function (a) { return {label: a.label, color: COL[a.key], kind: a.key === "a5" ? "ring" : a.key === "a1" ? "sq" : ""}; })
        .concat([{label: "Original problem", color: INK3, kind: "dash"}])));
      var chartEl = h("div", {"class": "pfh-fig-chart", style: "height:290px"});
      var cap = h("p", {"class": "pfh-fig-cap"});
      root.appendChild(chartEl); root.appendChild(cap);
      var chart = chartIn(chartEl);
      function draw() {
        var P = D.rows[cur], narrow = chartEl.clientWidth < 520, vals = [P.original];
        P.cases.forEach(function (c) {
          D.arms.forEach(function (a) { var v = c.arms[a.key]; if (v) vals = vals.concat(v.seeds.filter(Boolean), [v.mean]); });
        });
        var ext = decades(vals), top = Math.max.apply(null, vals.filter(function (v) { return v > 0; }));
        ext[1] = Math.min(ext[1], Math.pow(10, Math.log10(top) + 0.12));   // just above the largest seed, not the next power of 10
        var series = D.arms.map(function (a, k) {
          return {type: "custom", id: a.key, name: a.label, z: 3 + k, encode: {x: 0, y: 1},
            animationDelayUpdate: function (i) { return REDUCE ? 0 : i * 30 + k * 12; },
            renderItem: function (params, api) {
              var c = P.cases[params.dataIndex], v = c.arms[a.key];
              if (!v) return;
              var band = api.size([1, 0])[0], gap = Math.min(15, band * 0.17), x = api.coord([params.dataIndex, v.mean])[0] + (k - 1.5) * gap;
              var s = v.seeds.filter(Boolean), y = api.coord([params.dataIndex, v.mean])[1];
              var ya = api.coord([params.dataIndex, Math.min.apply(null, s)])[1], yb = api.coord([params.dataIndex, Math.max.apply(null, s)])[1];
              var ring = a.key === "a5", sq = a.key === "a1", rad = band > 46 ? 5 : 4;   // the retrained solver is a small square
              return {type: "group", children: [
                {type: "line", shape: {x1: x, y1: ya, x2: x, y2: yb}, style: {stroke: COL[a.key], lineWidth: 1.6, opacity: 0.5}, transition: "all",
                 enterFrom: {style: {opacity: 0}}, leaveTo: {style: {opacity: 0}}},
                {type: sq ? "rect" : "circle",
                 shape: sq ? {x: x - rad - 0.5, y: y - rad - 0.5, width: 2 * rad + 1, height: 2 * rad + 1, r: 2.5} : {cx: x, cy: y, r: ring ? rad - 0.6 : rad},
                 style: {fill: ring ? "#ffffff" : COL[a.key], stroke: ring ? COL[a.key] : "#ffffff", lineWidth: ring ? 2 : 1}, transition: "all",
                 enterFrom: {style: {opacity: 0}}, leaveTo: {style: {opacity: 0}}}]};
            },
            // an id per problem and case, so that a case of the same name in two problems (Re 1000 of LDC and KF) does not
            // slide from one column to the other; the cases of the old problem fade out and those of the new one fade in
            data: P.cases.map(function (c, i) { var v = c.arms[a.key]; return {id: P.name + "/" + c.title, value: [i, v ? v.mean : ext[0]]}; })};
        });
        series.push({id: "original", type: "line", data: [], silent: true, markLine: {silent: true, symbol: "none", precision: 10,   // not rounded to 2 decimals
          lineStyle: {color: INK3, type: "dashed", width: 1.2}, label: {show: false}, data: [{yAxis: P.original}]}});
        chart.setOption({
          textStyle: {fontFamily: UI, color: INK}, animationDuration: 500,
          animationDurationUpdate: REDUCE ? 0 : 650, animationEasingUpdate: "cubicInOut",
          grid: {left: 6, right: 10, top: 14, bottom: 6, containLabel: true},
          tooltip: Object.assign({}, TIP, {trigger: "axis", axisPointer: {type: "shadow", shadowStyle: {color: "rgba(232, 116, 59, 0.06)"}},
            formatter: function (ps) {
              var c = P.cases[ps[0].dataIndex];
              if (!c) return "";
              var s = "<div style='font-weight:600;margin-bottom:6px'>" + P.name + ", " + vh(c.title) + "</div>";
              D.arms.forEach(function (a) {
                var v = c.arms[a.key];
                if (!v) return;
                var sd = v.seeds.filter(Boolean);
                s += row(dotH(COL[a.key], a.key === "a5", a.key === "a1") + a.label, fmt(v.mean) + " " +
                         muted(fmt(Math.min.apply(null, sd)) + " to " + fmt(Math.max.apply(null, sd))), a.key === "a1");
              });
              return s + "<div style='margin-top:6px;color:" + INK3 + "'>Original problem " + fmt(P.original) + "</div>";
            }}),
          xAxis: {type: "category", data: P.cases.map(function (c) { return c.title; }),
                  axisLabel: {color: INK, fontSize: 11.5, interval: 0, rotate: narrow && P.cases.length > 5 ? 40 : 0, formatter: vr,
                              rich: {i: {fontFamily: UI, fontSize: 11.5, fontStyle: "italic", color: INK}}},
                  axisTick: {show: false}, axisLine: {show: true, lineStyle: {color: "#cfc6b9"}},
                  splitArea: {show: true, interval: 0, areaStyle: {color: ["rgba(243, 238, 229, 0.6)", "rgba(255, 255, 255, 0)"]}}},
          yAxis: logAxis({min: ext[0], max: ext[1], name: "Relative {i|L}² error", nameLocation: "middle", nameGap: 40,
                          nameTextStyle: Object.assign({fontFamily: UI, fontSize: 11.5, color: INK3}, L2),
                          axisLine: {show: true, lineStyle: {color: "#cfc6b9"}}}),
          series: series
        });                                                    // merged, so that the series morph
        cap.innerHTML = "Relative <i>L</i><sup>2</sup> error on the new cases of " + P.name + ", mean over five seeds with the range of the seeds" +
          (P.warm_share ? "; the warm start trains for " + Math.round(P.warm_share * P.time) + " s, " + Math.round(P.warm_share * 100) +
                          "% of the training time of " + P.time + " s" : "") + ".";
      }
      draw();
    });
  }

  /* ---------------------------------------------------------------- applications */
  function initApps(root, base) {
    return Promise.all([getJSON(base, "apps.json"), getBin(base, "apps.bin")]).then(function (r) {
      var D = r[0], B = r[1];
      root.innerHTML = "";
      var views = [smokeView(D.smoke, B, base), cylView(D.cyl, B), wingView(D.wing, B)];
      var segBox = seg(["Smoke", "Cyl", "Wing3D"], function (i) {
        views.forEach(function (v, k) { v.el.hidden = k !== i; if (k !== i && v.stop) v.stop(); });
        fade(views[i].el);
        views[i].show();
      }, "Application");
      root.appendChild(h("div", {"class": "pfh-fig-top"}, [segBox]));
      views.forEach(function (v, k) { v.el.hidden = k !== 0; root.appendChild(v.el); });
      views[0].show();
    });
  }

  function smokeView(S, B, base) {
    var el = h("div", {"class": "pfh-fig-view"});
    var ratio = S.base.w + " / " + S.base.h, lon = S.base.lon, lat = S.base.lat;
    function city(c, named) {
      var x = (c.lon - lon[0]) / (lon[1] - lon[0]) * 100, y = (lat[1] - c.lat) / (lat[1] - lat[0]) * 100;
      var west = c.name === "Ottawa" || c.name === "Toronto";
      return h("span", {"class": "pfh-city" + (west ? " west" : ""), style: "left:" + x + "%;top:" + y + "%"},
               named ? [h("b", {text: c.name})] : []);
    }
    // the graticule every 5° of latitude and 10° of longitude, labelled inside the right and lower edges
    function graticule(m) {
      [40, 45, 50, 55].forEach(function (la) {
        var y = (lat[1] - la) / (lat[1] - lat[0]) * 100;
        m.appendChild(h("i", {"class": "pfh-grat h", style: "top:" + y + "%"}));
        m.appendChild(h("span", {"class": "pfh-grat-l lat", style: "top:" + y + "%", text: la + "°N"}));
      });
      [-85, -75, -65].forEach(function (lo) {
        var x = (lo - lon[0]) / (lon[1] - lon[0]) * 100;
        m.appendChild(h("i", {"class": "pfh-grat v", style: "left:" + x + "%"}));
        m.appendChild(h("span", {"class": "pfh-grat-l lon", style: "left:" + x + "%", text: -lo + "°W"}));
      });
    }
    var maps = [mapBox("Reference", ratio, "pfh-geo"), mapBox("Discovered PINN solver", ratio, "pfh-geo")];
    maps.forEach(function (m, k) {
      m.style.backgroundImage = "url('" + base + "smoke_base.webp')";
      graticule(m);
      S.cities.forEach(function (c) { m.appendChild(city(c, k === 0)); });
    });
    var grid = h("div", {"class": "pfh-maps-2"}, maps);
    var k0 = 0, ci = 0;
    function drawMaps(k) {
      [S.ref, S.pred].forEach(function (e, j) {
        drawField(maps[j].cv, {e: e, q: frame(B, e, k), cm: SMOKE, norm: function (v) { return Math.sqrt(Math.max(0, v) / 0.7); },
          alpha: function (v) { return Math.min(1, Math.sqrt(Math.max(0, v) / 0.7)) * 0.94; }, dpr: 1.5});
      });
    }
    var pl = player(S.hours.length, function (i) { return Math.round(S.hours[i]) + " h"; }, function (i) { k0 = i; drawMaps(i); mark(); }, 170, grid);
    var cityBox = seg(S.cities.map(function (c) { return c.name; }), function (i) { ci = i; series(); }, "City");
    var chartEl = h("div", {"class": "pfh-fig-chart", style: "height:180px"});
    el.appendChild(grid);
    el.appendChild(h("div", {"class": "pfh-fig-row"}, [pl, cbar(SMOKE, "0", "0.7", "Smoke burden")]));
    el.appendChild(h("div", {"class": "pfh-fig-sub"}, [h("b", {text: "Smoke burden over a city"}), cityBox]));
    el.appendChild(chartEl);
    el.appendChild(h("p", {"class": "pfh-fig-cap", html: "Smoke from the wildfires in Quebec, 4 to 7 June 2023, transported by reanalysis winds over 72 h."}));
    var chart = null;
    function series() {
      var c = S.cities[ci];
      chart.setOption({series: [
        {id: "pred", data: S.series_hours.map(function (t, i) { return [t, c.pred[i]]; })},
        {id: "ref", data: S.hours.map(function (t, i) { return [t, c.ref[i]]; })}]});
    }
    function mark() {
      if (chart) chart.setOption({series: [{id: "pred", markLine: {data: [{xAxis: S.hours[k0]}]}}]});
    }
    function init() {
      chart = chartIn(chartEl);
      chart.setOption({
        textStyle: {fontFamily: UI, color: INK}, animationDuration: 400,
        grid: {left: 6, right: 12, top: 12, bottom: 4, containLabel: true},
        tooltip: Object.assign({}, TIP, {trigger: "axis", formatter: function (ps) {
          var s = "<div style='font-weight:600;margin-bottom:4px'>" + S.cities[ci].name + ", " + Math.round(ps[0].value[0]) + " h</div>";
          ps.forEach(function (p) { s += row(dotH(p.seriesId === "pred" ? EMBER : INK) + p.seriesName, fmt(p.value[1])); });
          return s;
        }}),
        xAxis: valAxis({min: 0, max: 72, interval: 12, axisLabel: {color: INK3, fontSize: 11.5, formatter: "{value} h"}}),
        yAxis: valAxis({min: 0, splitNumber: 3}),
        series: [
          {id: "pred", name: "Discovered PINN solver", type: "line", showSymbol: false, smooth: false, lineStyle: {color: EMBER, width: 2.2},
           itemStyle: {color: EMBER}, z: 3, markLine: {silent: true, symbol: "none", animation: false, label: {show: false},
           lineStyle: {color: EMBER2, width: 1, type: "solid", opacity: 0.6}, data: [{xAxis: 0}]}},
          {id: "ref", name: "Reference", type: "scatter", symbolSize: 5, itemStyle: {color: INK}, z: 4}]
      });
      series();
      chart.getZr().on("click", function (ev) {
        var p = chart.convertFromPixel({gridIndex: 0}, [ev.offsetX, ev.offsetY]);
        if (!p) return;
        var best = 0;
        S.hours.forEach(function (t, i) { if (Math.abs(t - p[0]) < Math.abs(S.hours[best] - p[0])) best = i; });
        pl.play(false); pl.user = true; pl.set(best);
      });
      onResize(grid, function () { drawMaps(k0); });
    }
    return {el: el, show: function () { if (!chart) init(); else chart.resize(); drawMaps(k0); mark(); }, stop: function () { pl.play(false); }};
  }

  function cylView(C, B) {
    var el = h("div", {"class": "pfh-fig-view"});
    var ratio = ((C.x[1] - C.x[0]) / (C.y[1] - C.y[0])).toFixed(3);
    var maps = [mapBox("CFD reference", ratio), mapBox("Discovered PINN solver", ratio)];
    var k0 = 0;
    function after(probes) {
      return function (ctx, W, H) {
        var sx = W / (C.x[1] - C.x[0]), sy = H / (C.y[1] - C.y[0]);
        ctx.beginPath();
        ctx.arc((C.circle[0] - C.x[0]) * sx, H - (C.circle[1] - C.y[0]) * sy, C.circle[2] * sx, 0, 2 * Math.PI);
        ctx.fillStyle = "#e9e3d9"; ctx.fill();
        ctx.lineWidth = Math.max(1, W / 700); ctx.strokeStyle = "#8c8378"; ctx.stroke();
        if (!probes) return;
        C.probes.forEach(function (p) {
          var x = (p.x - C.x[0]) * sx, y = H - (p.y - C.y[0]) * sy, d = W / 150;
          ctx.beginPath(); ctx.moveTo(x - d, y - d); ctx.lineTo(x + d, y + d); ctx.moveTo(x + d, y - d); ctx.lineTo(x - d, y + d);
          ctx.lineWidth = Math.max(1.5, W / 450); ctx.strokeStyle = INK2; ctx.stroke();
        });
      };
    }
    function drawMaps(k) {
      [C.ref, C.pred].forEach(function (e, j) {
        drawField(maps[j].cv, {e: e, q: frame(B, e, k), cm: DIV, norm: lin(-45, 45), dpr: 1.5, after: after(j === 0)});
      });
    }
    var grid = h("div", {"class": "pfh-maps-col"}, maps);
    var pl = player(C.t.length, function (i) { return "t = " + C.t[i].toFixed(2); }, function (i) { k0 = i; drawMaps(i); mark(); }, 200, grid);
    var chartEl = h("div", {"class": "pfh-fig-chart", style: "height:300px"});
    el.appendChild(grid);
    el.appendChild(h("div", {"class": "pfh-fig-row"}, [pl, cbar(DIV, "−45", "45", "Vorticity", "0")]));
    var leg = legend([{label: "Discovered PINN solver", color: EMBER, kind: "ln"}, {label: "CFD reference", color: INK},
                      {label: "FEATFLOW benchmark", color: "#cdc5ba", kind: "ln wide"}]);
    leg.style.justifyContent = "center";                       // centred above the plots (author, 2026-10-23)
    el.appendChild(leg);
    el.appendChild(chartEl);
    el.appendChild(h("p", {"class": "pfh-fig-cap", html: "Vortex shedding behind a cylinder in the DFG 2D-3 benchmark, with <i>Re</i> falling from 95 to 71 over 1.2 time units."}));
    var CL = "<i>C</i><sub>l</sub>";                          // the lift coefficient
    var chart = null, lim = [], PLOT = [CL, CL, CL, "<i>v</i> at <i>x</i> = 0.6", "<i>v</i> at <i>x</i> = 0.6", "<i>v</i> at <i>x</i> = 1.2", "<i>v</i> at <i>x</i> = 1.2"];
    function mark() {                                          // one line through the three plots at the time of the maps
      if (!chart) return;
      var x = chart.convertToPixel({xAxisIndex: 0}, C.t[k0]);
      chart.setOption({graphic: [{id: "now", type: "line", silent: true, z: 5, style: {stroke: EMBER2, lineWidth: 1, opacity: 0.6},
        shape: {x1: x, y1: chart.convertToPixel({yAxisIndex: 0}, lim[0]), x2: x, y2: chart.convertToPixel({yAxisIndex: 2}, -lim[2])}}]});
    }
    function init() {
      chart = chartIn(chartEl);
      window.addEventListener("resize", mark);
      var grids = [], xs = [], ys = [], series = [], names = ["<i>C<sub>l</sub></i>", "<i>v</i> at <i>x</i> = 0.6", "<i>v</i> at <i>x</i> = 1.2"];
      var tops = [6, 38, 70], hgt = 22;
      [0, 1, 2].forEach(function (g) {
        var m = g ? Math.max.apply(null, C.probes[g - 1].ref.map(Math.abs)) : 0.5;
        var a = m >= 1 ? 1 : m >= 0.5 ? 0.5 : 0.25;            // labels at 0, on the zero line, and at a round value either side
        lim[g] = 1.3 * m;
        grids.push({left: 58, right: 14, top: tops[g] + "%", height: hgt + "%"});
        xs.push(valAxis({gridIndex: g, min: 0, max: 1.2, interval: 0.2, axisLabel: {color: INK3, fontSize: 11.5},   // a time axis under each plot
                         axisLine: {onZero: false, show: true, lineStyle: {color: "#cfc6b9"}}, name: "{i|t}", nameLocation: "end", nameGap: 6,
                         nameTextStyle: Object.assign({fontFamily: UI, fontSize: 11.5, color: INK3}, L2)}));
        ys.push(valAxis({gridIndex: g, min: -1.3 * m, max: 1.3 * m,
                         axisTick: {show: false, customValues: [-a, a]},   // no ticks, but dashed grid lines at ±a (author, 2026-10-23)
                         splitLine: {show: true, lineStyle: {color: GRID, type: "dashed"}},
                         axisLabel: {color: INK3, fontSize: 11, customValues: [-a, 0, a], formatter: function (v) { return minus(fmt(v)); }},
                         name: ["{i|C}{s|l}", "{i|v}, {i|x} = 0.6", "{i|v}, {i|x} = 1.2"][g], nameLocation: "middle", nameGap: 40,
                         nameTextStyle: g ? V11 : COEF}));
      });
      series.push({type: "line", xAxisIndex: 0, yAxisIndex: 0, name: "FEATFLOW benchmark", showSymbol: false, silent: true,
                   lineStyle: {color: "#d9d2c7", width: 5}, data: C.lift.featflow_t.map(function (t, i) { return [t, C.lift.featflow[i]]; })});
      series.push({type: "line", xAxisIndex: 0, yAxisIndex: 0, name: "Discovered PINN solver", showSymbol: false, lineStyle: {color: EMBER, width: 1.8},
                   itemStyle: {color: EMBER}, data: C.lift.t.map(function (t, i) { return [t, C.lift.pred[i]]; })});
      series.push({type: "scatter", xAxisIndex: 0, yAxisIndex: 0, name: "CFD reference", symbolSize: 4.5, itemStyle: {color: INK},
                   data: C.t.map(function (t, i) { return [t, C.lift.ref[i]]; })});
      C.probes.forEach(function (p, n) {
        series.push({type: "line", xAxisIndex: n + 1, yAxisIndex: n + 1, name: "Discovered PINN solver", showSymbol: false,
                     lineStyle: {color: EMBER, width: 1.8}, itemStyle: {color: EMBER}, data: C.probe_t.map(function (t, i) { return [t, p.pred[i]]; })});
        series.push({type: "scatter", xAxisIndex: n + 1, yAxisIndex: n + 1, name: "CFD reference", symbolSize: 4.5, itemStyle: {color: INK},
                     data: C.t.map(function (t, i) { return [t, p.ref[i]]; })});
      });
      // the zero line of each plot, now that the time axes sit below
      [0, 1, 2].forEach(function (g) {
        series.push({id: "zero" + g, type: "line", xAxisIndex: g, yAxisIndex: g, data: [], silent: true,
                     markLine: {silent: true, symbol: "none", animation: false, label: {show: false},
                                lineStyle: {color: "#cfc6b9", width: 1, type: "dashed"}, data: [{yAxis: 0}]}});
      });
      chart.setOption({textStyle: {fontFamily: UI, color: INK}, animationDuration: 600, grid: grids, xAxis: xs, yAxis: ys, series: series,
        axisPointer: {link: [{xAxisIndex: "all"}], lineStyle: {color: "#cfc6b9"}},
        tooltip: Object.assign({}, TIP, {trigger: "axis", formatter: function (ps) {
          var s = "<div style='font-weight:600;margin-bottom:4px'><i>t</i> = " + ps[0].value[0].toFixed(3) + "</div>";
          ps.forEach(function (p) {
            if (p.seriesId && p.seriesId.indexOf("zero") === 0) return;
            s += row(dotH(p.seriesName === "CFD reference" ? INK : p.seriesName === "FEATFLOW benchmark" ? "#cdc5ba" : EMBER) +
                     PLOT[p.seriesIndex] + ", " + p.seriesName, minus(fmt(p.value[1])));
          });
          return s;
        }})});
      mark();
      chart.getZr().on("click", function (ev) {
        for (var g = 0; g < 3; g++) {
          if (!chart.containPixel({gridIndex: g}, [ev.offsetX, ev.offsetY])) continue;
          var p = chart.convertFromPixel({gridIndex: g}, [ev.offsetX, ev.offsetY]), best = 0;
          C.t.forEach(function (t, i) { if (Math.abs(t - p[0]) < Math.abs(C.t[best] - p[0])) best = i; });
          pl.play(false); pl.user = true; pl.set(best);
        }
      });
      onResize(grid, function () { drawMaps(k0); });
    }
    return {el: el, show: function () { if (!chart) init(); else { chart.resize(); mark(); } drawMaps(k0); }, stop: function () { pl.play(false); }};
  }

  function wingView(Wd, B) {
    var el = h("div", {"class": "pfh-fig-view"});
    var ratio = ((Wd.x[1] - Wd.x[0]) / (Wd.y[1] - Wd.y[0])).toFixed(3), si = 1;
    var maps = [mapBox("CFD reference", ratio), mapBox("Discovered PINN solver", ratio)];
    var secBox = seg(Wd.sections.map(function (s) { return "<i>z</i> = " + s.z; }), function (i) { si = i; drawMaps(); cp(); fade(grid); }, "Span section", si);
    var grid = h("div", {"class": "pfh-maps-col", style: "width:90%;margin:0 auto"}, maps);   // 0.9 times, centred (author, 2026-10-22)
    var chartEl = h("div", {"class": "pfh-fig-chart", style: "height:210px"});
    // above the maps, their colour bar and the span sections, in one row as wide as the maps (author, 2026-10-23)
    var bar = cbar(DIV, "0", "1.2", "Speed", "1");
    bar.style.margin = "0";
    el.appendChild(h("div", {"class": "pfh-fig-sub", style: "width:90%;margin-left:auto;margin-right:auto"}, [bar, secBox]));
    el.appendChild(grid);
    var leg = legend([{label: "Discovered PINN solver", color: EMBER, kind: "ln"}, {label: "CFD reference", color: INK}]);
    leg.style.justifyContent = "center";                       // centred, C_p named on its axis (author, 2026-10-23)
    el.appendChild(leg);
    el.appendChild(chartEl);
    el.appendChild(h("p", {"class": "pfh-fig-cap", html: "Steady flow over the ONERA M6 wing at <i>Re</i> = 1000 and an angle of attack of 7°."}));
    function drawMaps() {
      var S = Wd.sections[si];
      function after(ctx, W, H) {
        var sx = W / (Wd.x[1] - Wd.x[0]), sy = H / (Wd.y[1] - Wd.y[0]);
        ctx.beginPath();
        S.outline.forEach(function (p, i) {
          var x = (p[0] - Wd.x[0]) * sx, y = H - (p[1] - Wd.y[0]) * sy;
          if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
        });
        ctx.closePath();
        ctx.fillStyle = "#e9e3d9"; ctx.fill();
        ctx.lineWidth = Math.max(1, W / 700); ctx.strokeStyle = "#8c8378"; ctx.stroke();
      }
      [S.ref, S.pred].forEach(function (e, j) {
        drawField(maps[j].cv, {e: e, q: frame(B, e), cm: DIV, norm: two(0, 1, 1.2), after: after});
      });
    }
    var chart = null;
    function cp() {
      var S = Wd.sections[si];
      chart.setOption({series: [
        {id: "p", data: S.cp_xc.map(function (x, i) { return [x, S.cp_pred[i]]; })},
        {id: "r", data: S.cp_ref_xc.map(function (x, i) { return [x, S.cp_ref[i]]; })}]});
    }
    function init() {
      chart = chartIn(chartEl);
      chart.setOption({textStyle: {fontFamily: UI, color: INK}, animationDuration: 500,
        grid: {left: 22, right: 28, top: 10, bottom: 4, containLabel: true},   // room for the names C_p left of the labels and x/c at the end
        tooltip: Object.assign({}, TIP, {trigger: "item", formatter: function (p) {
          return row(dotH(p.seriesId === "p" ? EMBER : INK) + p.seriesName, "<i>x</i>/<i>c</i> " + fmt(p.value[0]) + ", <i>C</i><sub>p</sub> " + minus(fmt(p.value[1])));
        }}),
        xAxis: valAxis({min: 0, max: 1, interval: 0.25, name: "{i|x}/{i|c}", nameLocation: "end", nameGap: 6, nameTextStyle: Object.assign({fontFamily: UI, fontSize: 11.5, color: INK3}, L2),
                        axisLine: {onZero: false, show: true, lineStyle: {color: "#cfc6b9"}}}),   // a solid axis at the bottom, at -1
        yAxis: valAxis({min: -1, max: 1, interval: 0.5, name: "{i|C}{s|p}", nameLocation: "middle", nameGap: 36, nameTextStyle: COEF}),
        series: [{id: "p", name: "Discovered PINN solver", type: "line", showSymbol: false, lineStyle: {color: EMBER, width: 2}, itemStyle: {color: EMBER}, z: 3,
                  markLine: {silent: true, symbol: "none", animation: false, label: {show: false},                // and a dashed line at 0
                             lineStyle: {color: "#cfc6b9", width: 1, type: "dashed"}, data: [{yAxis: 0}]}},
                 {id: "r", name: "CFD reference", type: "scatter", symbolSize: 4.5, itemStyle: {color: INK}, z: 4}]});
      cp();
      onResize(grid, drawMaps);
    }
    return {el: el, show: function () { if (!chart) init(); else chart.resize(); drawMaps(); }};
  }

  /* ---------------------------------------------------------------- inside a discovery */
  var CAT = {as_in_note: PALE, adapted: EMBER, no_note: INK2};
  function initDisc(root, base) {
    return getJSON(base, "disc.json").then(function (D) {
      root.innerHTML = "";
      var keys = Object.keys(D.runs), cur = 0;
      var chartEl = h("div", {"class": "pfh-fig-chart", style: "height:400px"});
      var ablEl = h("div", {"class": "pfh-fig-chart", style: "height:300px"});
      var pathEl = h("div", {"class": "pfh-fig-chart", style: "height:330px"});
      var runView = h("div", {"class": "pfh-fig-view"}), pathView = h("div", {"class": "pfh-fig-view"});
      var segBox = seg(keys.concat(["Six runs"]), function (i) {
        runView.hidden = i === keys.length; pathView.hidden = i !== keys.length;
        if (i < keys.length) { cur = i; drawRun(); fade(runView); } else { drawPaths(); fade(pathView); }
      }, "Run");
      root.appendChild(h("div", {"class": "pfh-fig-top"}, [segBox, legend(Object.keys(D.categories).map(function (k) {
        return {label: D.categories[k], color: CAT[k], kind: k === "as_in_note" ? "pale" : ""};
      }))]));
      // legends in place of a caption (author, 2026-10-23)
      runView.appendChild(legend([{label: "Evaluation", color: "#cdc4b7"}, {label: "Lowest error so far", color: EMBER, kind: "ln"},
                                  {label: "Notes first cited", color: EMBER, kind: "sq"}, {label: "Notes cited before", color: SAND, kind: "sq"}]));
      runView.appendChild(chartEl);
      runView.appendChild(h("div", {"class": "pfh-fig-sub"}, [h("b", {text: "Each component removed from the final solver"}),
        h("span", {text: "five seeds"})]));
      runView.appendChild(ablEl);
      pathView.appendChild(pathEl);
      pathView.appendChild(h("p", {"class": "pfh-fig-cap", text: "The components of the six final solvers by the round that introduced them. Large points are the techniques behind the largest drops."}));
      pathView.hidden = true;
      root.appendChild(runView); root.appendChild(pathView);
      var chart = chartIn(chartEl), abl = chartIn(ablEl), paths = null;

      function drawRun() {
        var R = D.runs[keys[cur]];
        var ys = R.evals.map(function (e) { return e[1]; }).filter(function (v) { return v > 0; });
        var lo = Math.pow(10, Math.floor(Math.log10(Math.min.apply(null, ys)))), hi = 2;
        var best = R.best.map(function (b) { return [b[0], b[1]]; });
        best.push([10, best[best.length - 1][1]]);
        chart.setOption({
          textStyle: {fontFamily: UI, color: INK},
          grid: [{left: 54, right: 14, top: 14, height: "62%"}, {left: 54, right: 14, top: "72%", bottom: 26}],   // a gap of 26 px, was 42 (author, 2026-10-23)
          xAxis: [valAxis({gridIndex: 0, min: 0, max: 10, interval: 1, axisLabel: {show: false}}),
                  {gridIndex: 1, type: "category", data: R.notes.map(function (n) { return String(n.round); }), axisTick: {show: false},
                   axisLine: {show: true, lineStyle: {color: "#cfc6b9"}}, axisLabel: {color: INK3, fontSize: 11.5}, name: "Round", nameLocation: "middle",
                   nameGap: 24, nameTextStyle: {fontFamily: UI, fontSize: 11.5, color: INK3}}],
          yAxis: [logAxis({gridIndex: 0, min: lo, max: hi, name: "Relative {i|L}² error", nameLocation: "middle", nameGap: 38,
                           nameTextStyle: Object.assign({fontFamily: UI, fontSize: 11.5, color: INK3}, L2)}),
                  valAxis({gridIndex: 1, splitNumber: 2, name: "Notes", nameLocation: "middle", nameGap: 38, minInterval: 1,
                           axisLine: {show: true, lineStyle: {color: "#cfc6b9"}}})],
          tooltip: Object.assign({}, TIP, {trigger: "item", formatter: function (p) {
            if (p.seriesId === "lab") return "<div style='font-weight:600;margin-bottom:4px'>" + vh(p.data.full) + "</div>" +
              row(dotH(CAT[p.data.cat]) + D.categories[p.data.cat], "") + row("Error", fmt(p.data.value[1])) + muted(p.data.candidate);
            if (p.seriesId === "ev") return "Round " + Math.ceil(p.value[0]) + "<br>" + row("Error", fmt(p.value[1]));
            if (p.seriesId === "best") return row("Lowest so far", fmt(p.value[1]));
            if (p.seriesId === "n1" || p.seriesId === "n2") {
              var n = R.notes[p.dataIndex];
              return "<div style='font-weight:600;margin-bottom:4px'>Round " + n.round + "</div>" + row(dotH(EMBER) + "First cited", n.first) +
                     row(dotH(SAND) + "Cited before", n.cited - n.first) + row("Notes cited", n.cited, true);
            }
            return "";
          }}),
          series: [
            {id: "ev", type: "scatter", xAxisIndex: 0, yAxisIndex: 0, symbolSize: 5, itemStyle: {color: "#cdc4b7", opacity: 0.75}, z: 2,
             data: R.evals.filter(function (e) { return e[1] > 0; }).map(function (e) { return [e[0], Math.min(e[1], hi)]; }),
             animationDelay: function (i) { return REDUCE ? 0 : i * 4; }},
            {id: "best", type: "line", xAxisIndex: 0, yAxisIndex: 0, step: "end", showSymbol: false, z: 3, lineStyle: {color: EMBER, width: 2.6},
             itemStyle: {color: EMBER}, data: best, animationDuration: REDUCE ? 0 : 1800, animationEasing: "cubicOut"},
            {id: "lab", type: "scatter", xAxisIndex: 0, yAxisIndex: 0, z: 5, symbolSize: 13,   // named in the tooltip only (author, 2026-10-23)
             data: R.labels.map(function (l) {
               return {value: [l.x, l.y], full: l.label, cat: l.cat, candidate: l.candidate,
                       itemStyle: {color: CAT[l.cat], borderColor: l.cat === "as_in_note" ? "#d29466" : "#ffffff", borderWidth: 1.5}};
             }),
             animationDelay: REDUCE ? 0 : 1400},
            {id: "n1", type: "bar", xAxisIndex: 1, yAxisIndex: 1, stack: "n", name: "First cited", barWidth: "55%", itemStyle: {color: EMBER, borderRadius: 0},
             data: R.notes.map(function (n) { return n.first; })},
            {id: "n2", type: "bar", xAxisIndex: 1, yAxisIndex: 1, stack: "n", name: "Cited before", itemStyle: {color: SAND, borderRadius: [3, 3, 0, 0]},
             data: R.notes.map(function (n) { return n.cited - n.first; })}
          ]
        }, true);
        drawAbl(R);
      }
      function drawAbl(R) {
        var rows = R.ablation.slice().reverse(), base0 = R.ablation[0].mean;
        var vals = []; rows.forEach(function (r) { vals.push(r.min, r.max); });
        var ext = decades(vals);
        ablEl.style.height = (rows.length * 27 + 40) + "px";
        abl.resize();
        abl.setOption({
          textStyle: {fontFamily: UI, color: INK}, animationDuration: 600,
          grid: {left: 6, right: 16, top: 6, bottom: 4, containLabel: true},
          tooltip: Object.assign({}, TIP, {trigger: "item", formatter: function (p) {
            var r = rows[p.dataIndex];
            return "<div style='font-weight:600;margin-bottom:4px'>" + (r.component ? "Without: " + vh(r.component) : vh(r.label)) + "</div>" +
              (r.cat ? row(dotH(CAT[r.cat]) + D.categories[r.cat], "") : "") + row("Mean", fmt(r.mean), true) +
              row("Seeds", r.seeds.map(fmt).join(", "));
          }}),
          xAxis: logAxis({min: ext[0], max: ext[1], name: "Relative {i|L}² error", nameLocation: "middle", nameGap: 24,
                          nameTextStyle: Object.assign({fontFamily: UI, fontSize: 11.5, color: INK3}, L2),
                          axisLine: {show: true, lineStyle: {color: "#cfc6b9"}}}),
          yAxis: {type: "category", data: rows.map(function (r) { return r.label; }), axisTick: {show: false}, axisLine: {show: true, lineStyle: {color: "#cfc6b9"}},
                  axisLabel: {color: INK, fontSize: 12},
                  splitArea: {show: true, areaStyle: {color: ["rgba(243, 238, 229, 0.6)", "rgba(255, 255, 255, 0)"]}}},   // rows of alternating colour
          series: [{type: "custom", encode: {x: 1, y: 0}, renderItem: function (params, api) {
              var r = rows[params.dataIndex], y = api.coord([r.mean, params.dataIndex])[1];
              var x = api.coord([r.mean, params.dataIndex])[0], xa = api.coord([r.min, params.dataIndex])[0], xb = api.coord([r.max, params.dataIndex])[0];
              var col = r.cat ? CAT[r.cat] : (r.label.indexOf("Standard") === 0 ? "#b5ab9e" : EMBER);
              var main = !r.component && r.label.indexOf("Standard") !== 0;
              return {type: "group", children: [
                {type: "line", shape: {x1: xa, y1: y, x2: xb, y2: y}, style: {stroke: col === PALE ? "#d29466" : col, lineWidth: 1.6, opacity: 0.6}},
                {type: main ? "rect" : "circle", shape: main ? {x: x - 6, y: y - 6, width: 12, height: 12, r: 3} : {cx: x, cy: y, r: 5.5},
                 style: {fill: col, stroke: col === PALE ? "#d29466" : "#ffffff", lineWidth: 1.2}}]};
            }, data: rows.map(function (r, i) { return [i, r.mean]; }),
            markLine: {silent: true, symbol: "none", animation: false, label: {show: false}, lineStyle: {color: INK3, type: "dashed", width: 1.2},
                       precision: 10, data: [{xAxis: base0}]}}]   // the exact mean, not rounded to 2 decimals
        }, true);
      }
      function drawPaths() {
        if (paths) { paths.resize(); return; }
        paths = chartIn(pathEl);
        var names = Object.keys(D.paths), pts = [];
        names.forEach(function (n, ri) {
          var P = D.paths[n], by = {};
          P.dots.forEach(function (d) { (by[d.round] = by[d.round] || []).push({round: d.round, name: d.component, cat: d.category, notes: d.notes, big: false}); });
          P.nodes.forEach(function (d) { (by[d.round] = by[d.round] || []).push({round: d.round, name: d.label, cat: d.categories[0], big: true}); });
          Object.keys(by).forEach(function (r) {
            var a = by[r];
            a.forEach(function (d, i) {
              pts.push({value: [d.round + (i - (a.length - 1) / 2) * 0.17, ri], d: d, run: n, symbolSize: d.big ? 15 : 9,
                        itemStyle: {color: CAT[d.cat], borderColor: d.cat === "as_in_note" ? "#d29466" : "#ffffff", borderWidth: 1.2}});
            });
          });
        });
        paths.setOption({
          textStyle: {fontFamily: UI, color: INK}, animationDuration: 700,
          grid: {left: 6, right: 6, top: 30, bottom: 26, containLabel: true},
          tooltip: Object.assign({}, TIP, {trigger: "item", formatter: function (p) {
            var d = p.data.d;
            return "<div style='font-weight:600;margin-bottom:4px'>" + vh(d.name) + "</div>" + row(dotH(CAT[d.cat]) + D.categories[d.cat], "") +
              row(p.data.run + ", round " + d.round, "") + (d.notes ? muted("Notes " + d.notes.split(" ").join(", ")) : "");
          }}),
          xAxis: valAxis({min: 0, max: 11, interval: 1, name: "Round", nameLocation: "middle", nameGap: 24,
                          axisLabel: {color: INK3, fontSize: 11.5, formatter: function (v) { return v > 0 && v < 11 ? String(v) : ""; }},
                          axisLine: {show: true, lineStyle: {color: "#cfc6b9"}}}),
          yAxis: [{type: "category", data: names, inverse: true, axisTick: {show: false}, axisLine: {show: true, lineStyle: {color: "#cfc6b9"}}, axisLabel: {color: INK, fontSize: 12},
                   splitArea: {show: true, areaStyle: {color: ["rgba(243, 238, 229, 0.6)", "rgba(255, 255, 255, 0)"]}}},
                  {type: "category", data: names, inverse: true, position: "right", axisTick: {show: false}, axisLine: {show: false},
                   name: "Best", nameLocation: "start", nameGap: 12, nameTextStyle: {fontFamily: UI, fontSize: 11.5, color: INK3, align: "right"},
                   axisLabel: {color: EMBER2, fontSize: 12, fontWeight: 700, formatter: function (v) { return D.paths[v].best; }}}],
          series: [{type: "scatter", data: pts, animationDelay: function (i) { return REDUCE ? 0 : i * 12; }}]
        });
        paths.getZr().on("click", function (ev) {
          var p = paths.convertFromPixel({gridIndex: 0}, [ev.offsetX, ev.offsetY]);
          if (!p) return;
          var k = keys.indexOf(names[Math.round(p[1])]);
          if (k >= 0) segBox.pick(k);
        });
      }
      drawRun();
    });
  }

  /* ---------------------------------------------------------------- ablations */
  function initAbl(root, base) {
    var COL = {full: EMBER, nokb1: INK2, pdfkb: STEEL, kbcore: SAND, "1block": STEEL, dr: SAND, era: INK2};
    var NAME = {full: "PINNSeek", nokb1: "Without the literature base", pdfkb: "PDF files instead of notes",
                kbcore: "Without the cited notes", "1block": "A single round with the whole budget",
                dr: "Claude Code with web search", era: "Tree search of ERA"};
    return getJSON(base, "abl.json").then(function (D) {
      root.innerHTML = "";
      var group = "knowledge", task = "ldc";
      var g = seg(["Knowledge", "Agency"], function (i) { group = ["knowledge", "agency"][i]; draw(); }, "Ablation");
      var t = seg(["LDC", "KS"], function (i) { task = ["ldc", "ks"][i]; draw(); }, "Problem");
      var leg = h("div");
      var chartEl = h("div", {"class": "pfh-fig-chart", style: "height:340px"});
      var cap = h("p", {"class": "pfh-fig-cap"});
      root.appendChild(h("div", {"class": "pfh-fig-top"}, [g, t]));
      root.appendChild(leg); root.appendChild(chartEl); root.appendChild(cap);
      var chart = chartIn(chartEl);
      function draw() {
        var P = D.panels[task + "_" + group], H = P.hours, vals = [];
        P.conditions.forEach(function (c) { vals = vals.concat(c.min, c.max); });
        var ext = decades(vals.filter(function (v) { return v > 0; }));
        leg.innerHTML = "";
        var lg = legend(P.conditions.map(function (c) { return {label: NAME[c.key], color: COL[c.key], kind: "ln"}; }));
        lg.style.cssText = "display:grid;grid-template-columns:repeat(2, minmax(0, max-content));justify-content:center;column-gap:28px";   // two rows of two, centred
        leg.appendChild(lg);
        var series = [];
        P.conditions.forEach(function (c, k) {
          var col = COL[c.key];
          series.push({type: "custom", silent: true, z: 1 + k, renderItem: function (params, api) {
            if (params.dataIndex) return;
            var top = [], bot = [];
            for (var i = 0; i < H.length; i++) {                // from the first evaluation on
              if (c.max[i] == null || c.min[i] == null) continue;
              top.push(api.coord([H[i], c.max[i]])); bot.push(api.coord([H[i], c.min[i]]));
            }
            return {type: "polygon", shape: {points: top.concat(bot.reverse())}, style: {fill: col, opacity: c.key === "full" ? 0.16 : 0.11},
                    enterFrom: {style: {opacity: 0}}};
          }, data: [[0, 1]], clip: true});
          series.push({type: "line", name: NAME[c.key], showSymbol: false, z: 10 + (c.key === "full" ? 5 : k),
                       lineStyle: {color: col, width: c.key === "full" ? 3 : 2.2}, itemStyle: {color: col},
                       data: H.map(function (x, i) { return [x, c.mean[i]]; }),
                       endLabel: {show: true, formatter: function () { return fmt(c.final_mean); }, color: col === SAND ? "#a8813f" : col,
                                  fontFamily: UI, fontSize: 12, fontWeight: 700, distance: 6},
                       animationDuration: REDUCE ? 0 : 1600, animationEasing: "cubicOut"});
        });
        chart.setOption({
          textStyle: {fontFamily: UI, color: INK},
          grid: {left: 54, right: 52, top: 12, bottom: 38},   // room for the axis name below the ticks
          tooltip: Object.assign({}, TIP, {trigger: "axis", formatter: function (ps) {
            var s = "<div style='font-weight:600;margin-bottom:4px'>" + ps[0].value[0].toFixed(1) + " GPU hours</div>";
            ps.filter(function (p) { return p.seriesType === "line"; }).sort(function (a, b) { return a.value[1] - b.value[1]; }).forEach(function (p) {
              s += row(dotH(p.color) + p.seriesName, fmt(p.value[1]), p.seriesName === "PINNSeek");
            });
            return s;
          }}),
          xAxis: valAxis({min: 0, max: H[H.length - 1], interval: H[H.length - 1] / 5, name: "GPU hours", nameLocation: "middle", nameGap: 24}),
          yAxis: logAxis({min: ext[0], max: Math.min(ext[1], 10), name: "Relative {i|L}² error", nameLocation: "middle", nameGap: 38,
                          nameTextStyle: Object.assign({fontFamily: UI, fontSize: 11.5, color: INK3}, L2),
                          axisLine: {show: true, lineStyle: {color: "#cfc6b9"}}}),   // a solid axis over the dashed grid line at 0 h
          series: series
        }, true);
        var runs = P.conditions.map(function (c) { return c.runs; });
        cap.innerHTML = "The lowest error found so far against the GPU time, geometric mean over the runs of each condition, with the range of the runs. Same agent model, task and GPU budget; " +
          (runs.every(function (n) { return n === 3; }) ? "three runs per condition." :
            "three runs per condition, " + P.conditions.filter(function (c) { return c.runs !== 3; }).map(function (c) {
              return (c.runs === 1 ? "one" : "two") + " for the " + NAME[c.key].replace(/^Tree search/, "tree search");
            }).join(", ") + ".");
      }
      draw();
    });
  }

  /* ---------------------------------------------------------------- start */
  ready(function () {
    var FIGS = {"fig-gen": initGen, "fig-apps": initApps, "fig-disc": initDisc, "fig-abl": initAbl};
    function start(el) {
      if (el.dataset.started) return;
      el.dataset.started = "1";
      if (!window.echarts || !window.fetch) return;
      var keep = el.innerHTML;
      el.classList.add("pfh-fig-live");
      FIGS[el.id](el, el.getAttribute("data-res")).catch(function (err) {
        el.classList.remove("pfh-fig-live");
        el.innerHTML = keep;                                   // the paper's figure stays
        if (window.console) console.warn("figure", el.id, err);
      });
    }
    var els = Object.keys(FIGS).map(function (id) { return document.getElementById(id); }).filter(Boolean);
    if (!window.IntersectionObserver) { els.forEach(start); return; }
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (en) { if (en.isIntersecting) { io.unobserve(en.target); start(en.target); } });
    }, {rootMargin: "400px 0px"});
    els.forEach(function (el) { io.observe(el); });
  });
})();
