/* PINNSeek review site: equations, charts and the figure lightbox. No storage; visits are counted by GoatCounter,
   without cookies (overrides/partials/integrations/analytics/goatcounter.html). */
(function () {
  "use strict";

  var MACROS = {
    "\\pd": "\\frac{\\partial #1}{\\partial #2}",
    "\\pdn": "\\frac{\\partial^{#1} #2}{\\partial #3^{#1}}"
  };
  // the charts of the case pages in the palette of the landing page (assets/home/results.js; author, 2026-10-23)
  var FONT = 'Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
  var INK = "#1d1a16", INK2 = "#4a443c", INK3 = "#7b7368", GRID = "#ece5d9", AXIS = "#cfc6b9";
  var EMBER = "#e8743b", EMBER2 = "#c85a26", PALE = "#f3c49c", PALE_EDGE = "#d29466";
  var TIP = {backgroundColor: "#fffdf9", borderColor: "#e6dfd3", borderWidth: 1, padding: [9, 12],
             textStyle: {fontFamily: FONT, fontSize: 12.5, color: INK},
             extraCssText: "border-radius: 10px; box-shadow: 0 12px 30px rgba(60, 40, 20, .14);"};
  // axis titles with an italic L (ECharts rich text)
  var L2_RICH = {fontFamily: FONT, fontSize: 11.5, color: INK3, rich: {i: {fontStyle: "italic", fontFamily: FONT, fontSize: 11.5, color: INK3}}};

  function onReady(fn) {
    if (document.readyState !== "loading") fn(); else document.addEventListener("DOMContentLoaded", fn);
  }

  function readJSON(id) {
    var el = document.getElementById(id);
    return el ? JSON.parse(el.textContent) : null;
  }

  function sup(n) {
    var map = {"-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
               "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"};
    return String(n).split("").map(function (c) { return map[c] || c; }).join("");
  }
  function logLabel(v) {
    var e = Math.round(Math.log10(v));
    if (Math.abs(Math.log10(v) - e) > 1e-6) return "";
    if (e === 0) return "1";
    if (e === 1) return "10";
    return "10" + sup(e);
  }
  function fmt(v) {
    if (v === null || v === undefined) return "";
    if (v >= 0.001 && v < 1000) return Number(v.toPrecision(3)).toString();
    var e = Math.floor(Math.log10(v)), m = v / Math.pow(10, e);
    return m.toPrecision(3) + "×10" + sup(e);
  }

  /* ---------------------------------------------------------------- equations */
  function renderMath() {
    if (!window.renderMathInElement) return;
    window.renderMathInElement(document.body, {
      delimiters: [
        {left: "\\[", right: "\\]", display: true},
        {left: "$", right: "$", display: false}
      ],
      macros: MACROS,
      throwOnError: false,
      ignoredClasses: ["chart", "md5"]
    });
  }

  /* ---------------------------------------------------------------- lightbox */
  function lightbox() {
    var box = document.getElementById("lightbox");
    if (!box) return;
    var img = box.querySelector("img");
    document.querySelectorAll("img.zoom").forEach(function (el) {
      var link = el.closest("a");
      (link || el).addEventListener("click", function (ev) {
        ev.preventDefault();
        img.src = link ? link.getAttribute("href") : el.src;
        img.alt = el.alt;
        box.classList.add("open");
      });
    });
    box.addEventListener("click", function () { box.classList.remove("open"); img.src = ""; });
    document.addEventListener("keydown", function (ev) { if (ev.key === "Escape") box.click(); });
  }

  /* ---------------------------------------------------------------- overview chart (home) */
  function overviewChart() {
    var el = document.getElementById("overview-chart");
    var data = readJSON("overview-data");
    if (!el || !data || !window.echarts) return;
    var OURS = "Discovered PINN solver";
    var methods = Object.keys(data.colors), others = methods.slice(1);
    // the palette of the landing page: the discovered solvers in ember, the four methods in muted tones
    var COLOR = {"Discovered PINN solver": "#e8743b", "Scale-PINN": "#4a443c", "APTS": "#7f9cb8", "SPINN": "#cdb08a", "DSGNAR": "#a39a8f"};
    var UI = 'Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
    var INKW = "#1d1a16", INKW2 = "#4a443c", INKW3 = "#7b7368";
    var NOTRUN = {"no-setup": "setup exceeds the training time", "OOM": "out of GPU memory"};
    // each problem with the ratio of the best other method's mean error to that of the discovered solver,
    // the largest ratio on top (a category axis runs from the bottom up)
    var rows = data.problems.map(function (p) {
      var best = null;
      others.forEach(function (m) {
        var c = p.methods[m];
        if (c.status === "ok" && (!best || c.mean < p.methods[best].mean)) best = m;
      });
      return {p: p, best: best, ratio: p.methods[best].mean / p.methods[OURS].mean};
    }).sort(function (a, b) { return a.ratio - b.ratio; });
    var names = rows.map(function (r) { return r.p.name; });
    function ratioLabel(r) {
      return (r < 10 ? r.toFixed(1) : r < 1000 ? String(Math.round(r)) : (Math.round(r / 1000) * 1000).toLocaleString("en")) + "×";
    }
    var narrow = el.clientWidth < 640;
    var chart = echarts.init(el, null, {renderer: "svg"});
    var series = [{
      // the gap between the discovered solver and the best of the four methods
      name: "gap", type: "custom", silent: true, z: 1, encode: {x: [0, 1], y: 2},
      renderItem: function (params, api) {
        var a = api.coord([api.value(0), api.value(2)]), b = api.coord([api.value(1), api.value(2)]);
        return {type: "line", shape: {x1: a[0], y1: a[1], x2: b[0], y2: b[1]},
                style: {stroke: "rgba(232, 116, 59, 0.28)", lineWidth: 4, lineCap: "round"}};
      },
      data: rows.map(function (r, i) { return [r.p.methods[OURS].mean, r.p.methods[r.best].mean, i]; })
    }].concat(methods.map(function (m) {
      var ours = m === OURS;
      return {
        name: m, type: "scatter", symbol: ours ? "roundRect" : "circle", symbolSize: ours ? 12 : 9, z: ours ? 5 : 3,   // a small square, as in the ablations
        itemStyle: {color: COLOR[m], borderColor: "#ffffff", borderWidth: 1.2, opacity: 1},
        emphasis: {scale: 1.35},
        data: rows.map(function (r, i) {
          var c = r.p.methods[m];
          return c.status === "ok" ? {value: [c.mean, i], id: r.p.id, problem: r.p.name} : null;
        }).filter(Boolean)
      };
    }));
    chart.setOption({
      textStyle: {fontFamily: UI, color: INKW},
      animation: false,
      legend: {top: 0, left: "center", icon: "circle", itemWidth: 10, itemHeight: 10, itemGap: 18,
               data: methods.map(function (m) { return m === OURS ? {name: m, icon: "roundRect"} : m; }),
               textStyle: {fontFamily: UI, fontSize: 12.5, color: INKW2},
               formatter: function (n) { return n === OURS ? "PINNSeek, discovered solver" : n; }},
      grid: {left: 4, right: narrow ? 4 : 18, top: narrow ? 116 : 48, bottom: 40, containLabel: true},
      tooltip: {
        trigger: "axis", axisPointer: {type: "shadow", shadowStyle: {color: "rgba(232, 116, 59, 0.07)"}},
        backgroundColor: "#fffdf9", borderColor: "#e6dfd3", borderWidth: 1, padding: [10, 12],
        textStyle: {fontFamily: UI, fontSize: 12.5, color: INKW},
        extraCssText: "border-radius: 10px; box-shadow: 0 12px 30px rgba(60, 40, 20, .14);",
        formatter: function (ps) {
          var r = rows[names.indexOf(ps[0].axisValue)];
          if (!r) return "";
          var dot = function (m) {
            return "<span style='display:inline-block;width:8px;height:8px;border-radius:" + (m === OURS ? "2px" : "50%") +
                   ";margin-right:7px;background:" + COLOR[m] + "'></span>";
          };
          var lines = methods.slice().sort(function (a, b) {
            return (r.p.methods[a].mean || Infinity) - (r.p.methods[b].mean || Infinity);
          }).map(function (m) {
            var c = r.p.methods[m], ours = m === OURS;
            var v = c.status === "ok" ? fmt(c.mean) + " <span style='color:" + INKW3 + "'>± " + fmt(c.std) + "</span>"
                                      : "<span style='color:" + INKW3 + "'>" + NOTRUN[c.status] + "</span>";
            return "<div style='display:flex;justify-content:space-between;gap:18px;" + (ours ? "font-weight:600;color:#c85a26" : "") + "'>" +
                   "<span>" + dot(m) + (ours ? "PINNSeek" : m) + "</span><span>" + v + "</span></div>";
          }).join("");
          return "<div style='font-weight:600;margin-bottom:6px'>" + esc(r.p.name) +
                 " <span style='font-weight:400;color:" + INKW3 + "'>· " + r.p.time + " s of training</span></div>" + lines +
                 "<div style='margin-top:6px;color:" + INKW3 + "'>" + r.best + ", the best of the four, has " + ratioLabel(r.ratio) +
                 " the error</div>";
        }
      },
      xAxis: {
        type: "log", logBase: 10, min: 1e-7, max: 10, name: "Mean relative {i|L}² error over five seeds",
        nameTextStyle: {fontFamily: UI, fontSize: 12, color: INKW2, rich: {i: {fontStyle: "italic", fontFamily: UI, fontSize: 12, color: INKW2}}},
        nameLocation: "middle", nameGap: 26, axisLabel: {color: INKW3, fontSize: 11.5, formatter: function (v) {
          return narrow && Math.round(Math.log10(v)) % 2 ? "" : logLabel(v);   // every other power on a phone
        }},
        splitLine: {lineStyle: {color: "#ece5d9", type: "dashed"}}, axisLine: {show: true, lineStyle: {color: "#cfc6b9"}}, axisTick: {show: false}   // solid axes, as on every chart
      },
      yAxis: [{
        type: "category", data: names, axisTick: {show: false}, axisLine: {show: true, lineStyle: {color: "#cfc6b9"}},
        // two columns, the problem and its training time, under their headings
        name: "{n|Problem}{t|Time}", nameLocation: "end", nameGap: 10,
        nameTextStyle: {align: "right", padding: [0, narrow ? 8 : 12, 0, 0], rich: {
          n: {width: narrow ? 68 : 84, align: "left", fontFamily: UI, fontSize: 11.5, color: INKW3},
          t: {width: narrow ? 40 : 48, align: "right", fontFamily: UI, fontSize: 11.5, color: INKW3}}},
        axisLabel: {margin: narrow ? 8 : 12, formatter: function (v, i) { return "{n|" + v + "}{t|" + rows[i].p.time + " s}"; }, rich: {
          n: {width: narrow ? 68 : 84, align: "left", fontFamily: UI, fontSize: narrow ? 11.5 : 12.5, fontWeight: 500, color: INKW},
          t: {width: narrow ? 40 : 48, align: "right", fontFamily: UI, fontSize: narrow ? 11 : 12, color: INKW3}}},
        splitArea: {show: true, areaStyle: {color: ["rgba(243, 238, 229, 0.6)", "rgba(255, 255, 255, 0)"]}}
      }, {
        type: "category", data: names, position: "right", axisTick: {show: false}, axisLine: {show: false},
        // the ratios right-aligned in a column outside the plot, under their heading
        name: "{r|Ratio}", nameLocation: "end", nameGap: 10,
        nameTextStyle: {align: "left", padding: [0, 0, 0, narrow ? 6 : 10],
                        rich: {r: {width: narrow ? 46 : 58, align: "right", fontFamily: UI, fontSize: 11.5, color: INKW3}}},
        axisLabel: {margin: narrow ? 6 : 10, align: "left", formatter: function (v, i) { return "{r|" + ratioLabel(rows[i].ratio) + "}"; },
                    rich: {r: {width: narrow ? 46 : 58, align: "right", fontFamily: UI, fontSize: narrow ? 11.5 : 12.5, fontWeight: 700, color: "#c85a26"}}}
      }],
      series: series
    });
    // the discovered solver opens its problem page; the four methods open their driver in the code viewer
    chart.on("click", function (p) {
      if (!p.data || !p.data.id) return;
      if (p.seriesName === OURS) { window.location.href = "problems/" + p.data.id + ".html"; return; }
      showDriver("", p.seriesName, p.data.id, p.data.problem, data.dirs[p.seriesName],
                 (data.files[p.data.id] || {})[p.seriesName] || [], "run.py");
    });
    chart.getZr().on("mousemove", function (e) { chart.getZr().setCursorStyle(e.target ? "pointer" : "default"); });
    window.addEventListener("resize", function () { chart.resize(); });
  }

  /* ---------------------------------------------------------------- per-problem chart (Extended Data Fig. 1 panel) */
  function problemChart() {
    var el = document.getElementById("problem-chart");
    var data = readJSON("problem-data");
    if (!el || !data || !window.echarts) return;
    var chart = echarts.init(el, null, {renderer: "svg"});
    var bars = data.bars;
    var LABEL = {"Discovery run": "Discovery\nrun", "Discovered solver, five seeds": "Discovered\nfive seeds"};
    var cats = bars.map(function (b) {
      if (LABEL[b.slot]) return LABEL[b.slot];
      var m = b.slot.split(" ");
      return m[0] + "\n" + (m[1] === "1x" ? data.time : 2 * data.time) + " s";
    });
    // the discovered solver in ember, its discovery run in pale ember, the four methods as in the overview chart
    var COLOR = {"Discovery run": PALE, "Discovered solver, five seeds": EMBER, "Scale-PINN": "#4a443c", "APTS": "#7f9cb8",
                 "SPINN": "#cdb08a", "DSGNAR": "#a39a8f"};
    function color(b) {
      var m = b.slot.replace(/ [12]x$/, "");
      return COLOR[b.slot] || COLOR[m] || data.colors[b.slot] || data.colors[m];
    }
    var ok = bars.filter(function (b) { return b.mean != null; });
    var lo = Math.pow(10, Math.floor(Math.log10(Math.min.apply(null, ok.map(function (b) { return b.min; })))));
    var hi = Math.pow(10, Math.ceil(Math.log10(Math.max.apply(null, ok.map(function (b) { return b.max; })) * 1.05)));
    chart.setOption({
      textStyle: {fontFamily: FONT, color: INK},
      animation: false,
      grid: {left: 8, right: 12, top: 16, bottom: 8, containLabel: true},
      tooltip: Object.assign({}, TIP, {
        trigger: "item",
        formatter: function (p) {
          var b = bars[p.dataIndex];
          var name = cats[p.dataIndex].replace("\n", LABEL[b.slot] ? " " : ", ");
          if (b.mean == null) return name + ": " + (b.not_run === "OOM" ? "out of GPU memory" : "setup exceeds the training time");
          var hint = "<br><span style='color:" + INK3 + "'>Click to read the code</span>";
          if (b.seeds.length === 1) return "<b>" + name + "</b><br>" + fmt(b.mean) + " (single training)" + hint;
          return "<b>" + name + "</b><br>mean " + fmt(b.mean) + "<br>range " + fmt(b.min) + " to " + fmt(b.max) +
                 "<br>seeds: " + b.seeds.map(fmt).join(", ") + hint;
        }
      }),
      xAxis: {type: "category", data: cats, axisLabel: {color: INK, interval: 0, fontSize: 11, lineHeight: 14},
              axisTick: {show: false}, axisLine: {lineStyle: {color: AXIS}}},
      yAxis: {type: "log", logBase: 10, min: lo, max: hi, axisLabel: {formatter: logLabel, color: INK3, fontSize: 11.5},
              name: "Relative {i|L}² error", nameTextStyle: L2_RICH, nameLocation: "middle", nameGap: 40,
              splitLine: {lineStyle: {color: GRID, type: "dashed"}}, axisLine: {show: true, lineStyle: {color: AXIS}},
              axisTick: {show: false}},
      series: [{
        /* bars drawn by hand: ECharts roots log-axis bars at 1, not at the axis minimum */
        type: "custom", z: 10,
        renderItem: function (params, api) {
          var b = bars[params.dataIndex];
          var p0 = api.coord([params.dataIndex, lo]);
          if (b.mean == null) {
            return {type: "text", rotation: Math.PI / 2, x: p0[0], y: p0[1] - 6,
                    style: {text: b.not_run === "OOM" ? "OOM" : "Setup", textAlign: "left", textVerticalAlign: "middle",
                            fill: INK3, fontSize: 11}};
          }
          var x = p0[0], yBase = p0[1];
          var yBar = api.coord([params.dataIndex, b.mean])[1];
          var bw = api.size([1, 0])[0] * 0.66;
          var yLo = api.coord([params.dataIndex, b.min])[1];
          var yHi = api.coord([params.dataIndex, b.max])[1];
          var s = {stroke: INK2, lineWidth: 1};
          var two = / 2x$/.test(b.slot);
          var edge = b.slot === "Discovery run" ? PALE_EDGE : LABEL[b.slot] ? EMBER2 : "rgba(60, 40, 20, 0.18)";
          var kids = [{type: "rect", shape: {x: x - bw / 2, y: yBar, width: bw, height: yBase - yBar},
                       style: {fill: color(b), stroke: edge, lineWidth: 0.6}}];
          if (two) {   // hatching as in the paper: white diagonal lines over the bar
            var n = Math.ceil((yBase - yBar + bw) / 7);
            for (var k = -Math.ceil(bw / 7); k < n; k++) {
              var y0 = yBase - k * 7;
              kids.push({type: "line", silent: true,
                         shape: {x1: x - bw / 2, y1: y0, x2: x + bw / 2, y2: y0 - bw},
                         style: {stroke: "rgba(255,255,255,0.9)", lineWidth: 1.2}, clipPath: {type: "rect",
                         shape: {x: x - bw / 2, y: yBar, width: bw, height: yBase - yBar}}});
            }
          }
          if (b.seeds.length > 1) {
            var w = 5;
            kids.push({type: "line", shape: {x1: x, y1: yLo, x2: x, y2: yHi}, style: s},
                      {type: "line", shape: {x1: x - w, y1: yLo, x2: x + w, y2: yLo}, style: s},
                      {type: "line", shape: {x1: x - w, y1: yHi, x2: x + w, y2: yHi}, style: s});
          }
          return {type: "group", children: kids};
        },
        encode: {x: 0, y: 1},
        data: bars.map(function (b, i) { return [i, b.mean != null ? b.mean : lo]; })
      }]
    });
    // every bar opens its code: the discovered solver, or the driver of the method
    chart.on("click", function (p) {
      var b = bars[p.dataIndex];
      if (!b) return;
      if (LABEL[b.slot]) {
        openFile("../files/solvers/" + data.case + ".py", "Discovered PINN solver for " + data.name + " / " + data.case + ".py",
                 "The solver PINNSeek discovered, as trained for " + data.time + " s per seed.", true);
        return;
      }
      var m = b.slot.replace(/ [12]x$/, "");
      if (!data.dirs[m] || !(data.files[m] || []).length) return;
      showDriver("../", m, data.case, data.name, data.dirs[m], data.files[m], "run.py");
    });
    chart.getZr().on("mousemove", function (e) { chart.getZr().setCursorStyle(e.target ? "pointer" : "default"); });
    window.addEventListener("resize", function () { chart.resize(); });
  }

  /* driver of a comparison method, with links to the other files of its folder */
  function showDriver(root, method, id, problem, dir, names, file) {
    var base = root + "files/sota/" + id + "/" + dir + "/";
    var links = names.map(function (n) {
      return n === file ? "<b>" + esc(n) + "</b>" : "<a href='#' data-f='" + esc(n) + "'>" + esc(n) + "</a>";
    }).join(" · ");
    openFile(base + file, method + " on " + problem + " / " + file,
             "Driver of " + method + " for Table 1. Files: " + links, true);
    document.querySelectorAll("#viewer .viewer-meta [data-f]").forEach(function (a) {
      a.addEventListener("click", function (ev) {
        ev.preventDefault();
        showDriver(root, method, id, problem, dir, names, a.getAttribute("data-f"));
      });
    });
  }

  /* ---------------------------------------------------------------- file viewer */
  var viewer = null;
  function openFile(url, title, meta, isCode) {
    var dlg = document.getElementById("viewer");
    if (!dlg) return;
    var body = dlg.querySelector(".viewer-body"), code = body.querySelector("code");
    dlg.querySelector(".viewer-title").textContent = title;
    dlg.querySelector(".viewer-meta").innerHTML = meta || "";
    body.classList.toggle("prose", !isCode);
    code.className = "";
    code.textContent = "Loading…";
    if (!viewer) {
      viewer = dlg;
      dlg.querySelector(".viewer-close").addEventListener("click", function () { dlg.close(); });
      var copy = dlg.querySelector(".viewer-copy");
      if (copy) copy.addEventListener("click", function () {
        var text = dlg.querySelector(".viewer-body code").textContent;
        var done = function () { copy.textContent = "Copied"; setTimeout(function () { copy.textContent = "Copy"; }, 1500); };
        if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(done);
        else {
          var ta = document.createElement("textarea");
          ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
          dlg.appendChild(ta); ta.select(); document.execCommand("copy"); dlg.removeChild(ta); done();
        }
      });
      dlg.addEventListener("click", function (ev) { if (ev.target === dlg) dlg.close(); });
    }
    if (dlg.showModal) dlg.showModal(); else dlg.setAttribute("open", "");
    body.scrollTop = 0;
    fetch(url).then(function (r) {
      if (!r.ok) throw new Error(r.status === 404 ? "This file is not in the run archive." : "HTTP " + r.status);
      return r.text();
    }).then(function (text) {
      if (isCode && window.hljs) {
        code.innerHTML = window.hljs.highlight(text, {language: "python", ignoreIllegals: true}).value;
        code.className = "hljs language-python";
      } else {
        code.textContent = text;
      }
      body.scrollTop = 0;   // every file opens at its first line
    }).catch(function (err) { code.textContent = String(err.message || err); });
  }

  /* ---------------------------------------------------------------- discovery process */
  var CAT_COLOR = {as_in_note: PALE, adapted: EMBER, no_note: INK2};   // as on the landing page
  var CAT_TEXT = {as_in_note: "as in a note of the literature base", adapted: "adapted from the literature base",
                  no_note: "not from the literature base"};
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]; }); }

  function discovery() {
    var el = document.getElementById("discovery-chart");
    var data = readJSON("discovery-data");
    if (!el || !data || !window.echarts) return;
    // the chart sits in a stage that also holds the band of the round under the cursor and the panel of its fields
    var stage = document.createElement("div");
    stage.className = "disc-stage";
    el.parentNode.insertBefore(stage, el);
    stage.appendChild(el);
    var hl = document.createElement("div"), pop = document.createElement("div");
    hl.className = "disc-hl";
    pop.className = "disc-pop";
    stage.appendChild(hl);
    stage.appendChild(pop);
    var chart = echarts.init(el, null, {renderer: "svg"});
    var current = null, span = null, incR = [], hover = -1;

    // the code of every plotted candidate, as on the review site (files/runs there; a folder named runs is ignored
    // by the repository's .gitignore, so here files/candidates; author, 2026-10-23)
    function base(run) { return "../files/candidates/" + data.case + "/" + run.run + "/"; }
    function meta(p) {
      return "Round " + p.r + " · seed " + p.s + " · relative <i>L</i>² error <b>" + fmt(p.e) + "</b>" +
             (p.n != null ? " · " + p.n.toLocaleString("en") + " steps" : "") +
             (p.t != null ? " · " + p.t + " s of training" : "");
    }
    function showCode(run, p) { openFile(base(run) + p.c, run.run + " / " + p.c, meta(p), true); }

    function draw(k) {
      var run = data.runs[k];
      current = run;
      var pts = run.points;
      var es = pts.map(function (p) { return p.e; });
      // error axis of the paper's figures; evaluations above its top (diverged solvers) are not drawn
      var lo = run.ylim ? run.ylim[0] : Math.pow(10, Math.floor(Math.log10(Math.min.apply(null, es))));
      var hi = run.ylim ? run.ylim[1] : Math.pow(10, Math.ceil(Math.log10(Math.max.apply(null, es) * 1.01)));
      var shown = pts.filter(function (p) { return p.e <= hi; });
      var R = run.rounds;
      // running best along the round axis
      var sorted = pts.slice().sort(function (a, b) { return a.x - b.x; });
      var best = Infinity, line = [];
      sorted.forEach(function (p) { if (p.e < best) { if (line.length) line.push([p.x, best]); best = p.e; line.push([p.x, best]); } });
      line.push([R, best]);
      span = {lo: lo, hi: hi, R: R};
      incR = [];                                              // the incumbent after each round
      for (var q = 0, upTo = Infinity; q <= R; q++) {
        for (var i = 0; i < pts.length; i++) if (pts[i].r === q && pts[i].e < upTo) upTo = pts[i].e;
        incR.push(upTo);
      }
      buildPop(run);
      // odd rounds shaded sand, even rounds left blank
      var bands = [];
      for (var r = 1; r <= R; r += 2) bands.push([{xAxis: r - 1, itemStyle: {color: "rgba(236, 228, 214, 0.6)"}}, {xAxis: r}]);
      var nodeData = run.nodes.map(function (n) {
        var p = pts.filter(function (q) { return q.c === n.c && q.e === n.e; })[0] || {c: n.c, e: n.e, x: n.x};
        var cat = n.categories[0];
        return {value: [n.x, n.e], node: n, p: p,
                itemStyle: {color: CAT_COLOR[cat] || INK2, borderColor: cat === "as_in_note" ? PALE_EDGE : "#ffffff"}};
      });
      chart.setOption({
        textStyle: {fontFamily: FONT, color: INK},
        animation: false,
        grid: {left: 8, right: 16, top: 24, bottom: 40, containLabel: true},
        tooltip: Object.assign({}, TIP, {
          trigger: "item", confine: true,
          extraCssText: TIP.extraCssText + " z-index: 2;",       // under the panel of the round's fields
          formatter: function (q) {
            var p = q.data && q.data.p;
            if (!p) return "";
            var n = q.data.node;
            return (n ? "<b>" + esc(n.label.replace(/\$/g, "")) + "</b><br>" + n.categories.map(function (c) { return CAT_TEXT[c]; }).join("; ") + "<br>" : "") +
                   esc(p.c) + "<br>" + meta(p) + "<br><span style='color:" + INK3 + "'>Click to read the code</span>";
          }
        }),
        // the bands mark the rounds; each round's number sits in the middle of its span
        xAxis: [{type: "value", min: 0, max: R, interval: 1, name: "Round", nameLocation: "middle", nameGap: 26,
                 nameTextStyle: {fontFamily: FONT, fontSize: 11.5, color: INK3},
                 axisLabel: {show: false}, axisLine: {lineStyle: {color: AXIS}}, axisTick: {show: false},
                 splitLine: {show: false}},
                {type: "value", min: 0, max: R, interval: 0.5, position: "bottom", silent: true,
                 axisLine: {show: false}, axisTick: {show: false}, splitLine: {show: false},
                 axisLabel: {color: INK3, fontSize: 11.5, showMinLabel: false, showMaxLabel: false,
                             formatter: function (v) { return Math.abs(v % 1 - 0.5) < 1e-9 ? String(v + 0.5) : ""; }}}],
        yAxis: {type: "log", logBase: 10, min: lo, max: hi, name: "Relative {i|L}² error", nameTextStyle: L2_RICH, nameLocation: "middle", nameGap: 44,
                axisLabel: {formatter: logLabel, color: INK3, fontSize: 11.5}, axisLine: {show: true, lineStyle: {color: AXIS}},
                axisTick: {show: false}, splitLine: {lineStyle: {color: GRID, type: "dashed"}}},
        series: [
          {type: "scatter", name: "evaluated", symbolSize: 6, z: 2, cursor: "pointer",
           itemStyle: {color: "#cdc4b7", opacity: 0.8},
           emphasis: {scale: 1.8, itemStyle: {color: INK3, opacity: 1}},
           data: shown.map(function (p) { return {value: [p.x, p.e], p: p}; }),
           markArea: {silent: true, data: bands, label: {show: false}}},
          {type: "line", name: "best", data: line, step: "end", symbol: "none", silent: true, z: 3,
           lineStyle: {color: EMBER, width: 2.6}},
          {type: "scatter", name: "nodes", symbol: "diamond", symbolSize: 14, z: 5, cursor: "pointer",
           itemStyle: {borderWidth: 1.5},
           label: {show: false},
           data: nodeData}
        ]
      }, true);
      document.getElementById("run-stats").textContent = pts.length + " evaluations in " + R + " rounds · best " + fmt(Math.min.apply(null, es)) +
        (pts.length > shown.length ? " · " + (pts.length - shown.length) + " diverged solver" + (pts.length - shown.length > 1 ? "s" : "") +
         " with errors above " + fmt(hi) + " not drawn (listed in the table)" : "") +
        (run.failed ? " · " + run.failed + (run.failed > 1 ? " runs" : " run") + " that crashed or returned no score " +
         (run.failed > 1 ? "are" : "is") + " not drawn" : "");
      var rn = document.getElementById("round-nodes");
      function cats(n) { return " <span class='muted'>(" + n.categories.map(function (c) { return CAT_TEXT[c]; }).join("; ") + ")</span>"; }
      if (nodeData.length) {
        rn.innerHTML = "<span class='muted'>Main techniques of the discovered solver:</span><ol>" +
          nodeData.map(function (d, i) { var n = d.node;
            return "<li>Round " + d.p.r + ": " + esc(n.label) + cats(n) + " <button type='button' class='linklike' data-node='" + i + "'>" +
                   esc(n.c.split("/").pop()) + "</button></li>"; }).join("") + "</ol>";
        rn.querySelectorAll("[data-node]").forEach(function (b) {
          b.addEventListener("click", function () { showCode(run, nodeData[+b.getAttribute("data-node")].p); });
        });
      } else if (run.round_nodes && run.round_nodes.length) {
        rn.innerHTML = "<span class='muted'>Main techniques of the discovered solver:</span><ol>" +
          run.round_nodes.map(function (n) { return "<li>Round " + n.round + ": " + esc(n.label) + cats(n) + "</li>"; }).join("") + "</ol>";
      } else {
        rn.innerHTML = "";
      }
      if (window.renderMathInElement) {
        window.renderMathInElement(rn, {delimiters: [{left: "$", right: "$", display: false}], throwOnError: false});
      }
      var rows = pts.slice().sort(function (a, b) { return a.x - b.x; });
      var tb = document.getElementById("eval-rows");
      tb.innerHTML = rows.map(function (p, i) {
        return "<tr" + (p.b ? " class='best'" : "") + "><td>" + p.r + "</td><td><button type='button' data-i='" + pts.indexOf(p) + "'>" + esc(p.c) +
               "</button></td><td>" + (p.s == null ? "" : p.s) + "</td><td>" + fmt(p.e) + (p.e > hi ? " <span class='muted'>(not drawn)</span>" : "") +
               "</td><td>" + (p.n == null ? "" : p.n) +
               "</td><td>" + (p.t == null ? "" : p.t) + "</td></tr>";
      }).join("");
      tb.querySelectorAll("button").forEach(function (b) {
        b.addEventListener("click", function () { showCode(run, pts[+b.getAttribute("data-i")]); });
      });
    }
    /* hovering a round lifts its band and shows, at one fixed place over the plot, the predicted field and the
       error field after that round (round_frames.py: the same view and colour scales for every round of a run), so
       moving across the rounds plays them as frames (author, 2026-11) */
    function signed(v) { return v === 0 ? "0" : (v < 0 ? "−" : "") + fmt(Math.abs(v)); }
    function decade(v) { var e = Math.round(Math.log10(v)); return e === 0 ? "1" : "10" + sup(e); }
    function bar(stops, a, b) {
      return "<div class='cb'><span>" + a + "</span><i style='background:linear-gradient(90deg," + stops.join(",") + ")'></i><span>" + b + "</span></div>";
    }
    function buildPop(run) {
      var F = run.frames;
      hover = -1;
      hl.classList.remove("on");
      pop.classList.remove("on");
      pop.innerHTML = "";
      if (!F) return;
      pop.style.width = Math.max(2 * F.w + 10 + 28, 330) + "px";       // tall views (RT) keep room for the text
      var size = "alt='' width='" + F.w + "' height='" + F.h + "' style='width:" + F.w + "px'";        // the images have twice the pixels
      pop.innerHTML = "<div class='dp-head'><b></b><span></span></div><div class='dp-q'>" + esc(F.label) + "</div><div class='dp-imgs'>" +
        "<figure><figcaption>Prediction</figcaption><img " + size + ">" +
        bar(F.pc, signed(F.p[0]), signed(F.p[1])) + "</figure>" +
        "<figure><figcaption>Absolute error</figcaption><img " + size + ">" +
        bar(F.ec, decade(F.e[0]), decade(F.e[1])) + "</figure></div><div class='dp-note'></div>";
      Object.keys(F.img).forEach(function (k) { F.img[k].forEach(function (src) { new Image().src = src; }); });   // ready to play
    }
    function place(r) {
      var x0 = chart.convertToPixel({xAxisIndex: 0}, r === 0 ? 0 : r - 1);
      var x1 = chart.convertToPixel({xAxisIndex: 0}, r === 0 ? 0.08 : r);
      var yt = chart.convertToPixel({yAxisIndex: 0}, span.hi), yb = chart.convertToPixel({yAxisIndex: 0}, span.lo);
      var xr = chart.convertToPixel({xAxisIndex: 0}, span.R);
      hl.style.transform = "translate(" + x0 + "px," + yt + "px)";
      hl.style.width = Math.max(8, x1 - x0) + "px";
      hl.style.height = (yb - yt) + "px";
      pop.style.top = (yt + 10) + "px";
      pop.style.right = (el.clientWidth - xr + 10) + "px";
    }
    function enter(r) {
      var F = current.frames;
      if (!F || r === hover) return;
      hover = r;
      place(r);
      hl.classList.add("on");
      var k = F.of[r], info = F.info[k], imgs = pop.querySelectorAll("img");
      imgs[0].src = F.img[k][0];
      imgs[1].src = F.img[k][1];
      pop.querySelector(".dp-head b").textContent = "Round " + r;
      pop.querySelector(".dp-head span").textContent = "best so far " + fmt(incR[r]);
      var j = +k.slice(1);
      // a later evaluation of the same file can overwrite the weights of the round's best one (KS run 3, round 4)
      var gone = info.e > incR[j] * 1.01 ? "; the weights of the " + fmt(incR[j]) + " evaluation were overwritten by a later one" : "";
      // a round that was not drawn is unchanged since the round drawn, or within 1% of it (round_fields.setting_rounds)
      var since = j === r ? "Fields of " : (incR[r] < incR[j] ? "Within 1% of round " : "Unchanged since round ") + j + ": fields of ";
      pop.querySelector(".dp-note").textContent = since + info.c + ", relative L² error " + fmt(info.e) + gone;
      pop.classList.add("on");
    }
    function leave() {
      hover = -1;
      hl.classList.remove("on");
      pop.classList.remove("on");
    }
    function track(e) {
      if (!current || !current.frames || !chart.containPixel("grid", [e.offsetX, e.offsetY])) { leave(); return; }
      var x = chart.convertFromPixel({xAxisIndex: 0}, e.offsetX);
      enter(Math.max(0, Math.min(span.R, x <= 0.08 ? 0 : Math.ceil(x))));
    }
    chart.getZr().on("mousemove", track);
    chart.getZr().on("click", track);                                   // a tap on a touch screen
    chart.getZr().on("globalout", leave);

    chart.on("click", function (q) { if (q.data && q.data.p) showCode(current, q.data.p); });
    document.querySelectorAll("[data-run]").forEach(function (b) {
      b.addEventListener("click", function () {
        document.querySelectorAll("[data-run]").forEach(function (o) { o.setAttribute("aria-pressed", o === b ? "true" : "false"); });
        draw(+b.getAttribute("data-run"));
      });
    });
    var m = /run=(\d)/.exec(location.hash);
    var k0 = m && data.runs[+m[1]] ? +m[1] : (data.initial || 0);
    document.querySelectorAll("[data-run]").forEach(function (o) { o.setAttribute("aria-pressed", +o.getAttribute("data-run") === k0 ? "true" : "false"); });
    draw(k0);
    window.addEventListener("resize", function () { chart.resize(); if (hover >= 0) place(hover); });
  }

  onReady(function () {
    lightbox();
    renderMath();
    overviewChart();
    problemChart();
    discovery();
  });
})();
