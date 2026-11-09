/* The field of the hero (author, 2026-10-23): the predictions of five discovered PINN solvers (Cyl, GL, KF, GS, RT) at once,
   as a marble drawn in ink on paper. A soft blob is split into five regions by borders that wander like marbled paper,
   with pale veins where two regions meet; each region is a window onto one field, which plays in a loop whose end
   dissolves into its start. The frames come from the cache of the problem-page animations (website/tools/hero_fields.py,
   assets/home/fields/). Hovering a region shows the equation of its problem under the field, and a click opens its page.
   Without WebGL 2 the video of Cyl and its equations stay; with reduced motion one still frame is drawn. */
(function () {
  "use strict";
  function ready(fn) { if (document.readyState !== "loading") fn(); else document.addEventListener("DOMContentLoaded", fn); }

  var ORDER = ["cyl", "gl", "kf", "gs", "rt"];
  var SUBT = {cyl: "cylinder vortex street", gl: "complex Ginzburg–Landau equation", kf: "Kolmogorov flow",
              gs: "Gray–Scott reaction–diffusion", rt: "Rayleigh–Taylor instability"};
  // each field in its region: px of the whole domain, the point of the domain at the centre of the region, periodic
  // axes (1 in x, 2 in both), seconds per pass, first frame, palette (0 two-sided, 1 one-sided) and the span used
  var LOOK = {cyl: {scale: [620, 150], anchor: [0.26, 0.5], per: 0, loop: 6, from: 0, pal: 0, span: 1},
              gl: {scale: [430, 430], anchor: [0.5, 0.5], per: 2, loop: 9, from: 18, pal: 0, span: 0.9},
              kf: {scale: [270, 270], anchor: [0.5, 0.5], per: 2, loop: 8, from: 0, pal: 0, span: 1},
              gs: {scale: [250, 250], anchor: [0.5, 0.5], per: 2, loop: 8, from: 6, pal: 1, span: 0.82},
              rt: {scale: [170, 340], anchor: [0.5, 0.52], per: 1, loop: 8, from: 0, pal: 0, span: 0.62}};

  var VS = "#version 300 es\nin vec2 a; void main(){ gl_Position = vec4(a, 0.0, 1.0); }";
  var FS = [
    "#version 300 es", "precision highp float;",
    "uniform vec2 uRes, uMouse, uBox; uniform float uTime, uRipple, uDpr, uHover, uIdMode;",
    "uniform sampler2D uTex0, uTex1, uTex2, uTex3, uTex4;",
    "uniform vec4 uLay[5], uDom[5]; uniform vec2 uSize[5], uSeed[5], uScale[5], uAnchor[5];",
    "uniform float uN[5], uFrame[5], uFrameB[5], uBlend[5], uPer[5], uPal[5], uSpan[5];",
    "uniform vec3 uCyl[5];",
    "out vec4 o;",
    "float h21(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }",
    "float vn(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);",
    "  return mix(mix(h21(i), h21(i + vec2(1,0)), u.x), mix(h21(i + vec2(0,1)), h21(i + vec2(1,1)), u.x), u.y); }",
    "float fbm(vec2 p){ float s = 0.0, a = 0.5; for (int k = 0; k < 4; k++) { s += a * vn(p); p *= 2.03; a *= 0.5; } return s; }",
    "vec3 hx(float r, float g, float b){ return vec3(r, g, b) / 255.0; }",
    "vec3 palDiv(float t){",
    "  vec3 c0 = hx(36.,73.,110.), c1 = hx(94.,141.,184.), c2 = hx(201.,220.,234.), c3 = hx(251.,244.,234.), c4 = hx(246.,207.,168.), c5 = hx(232.,116.,59.), c6 = hx(143.,58.,21.);",
    "  if (t < 0.2) return mix(c0, c1, t / 0.2); if (t < 0.4) return mix(c1, c2, (t - 0.2) / 0.2); if (t < 0.5) return mix(c2, c3, (t - 0.4) / 0.1);",
    "  if (t < 0.6) return mix(c3, c4, (t - 0.5) / 0.1); if (t < 0.8) return mix(c4, c5, (t - 0.6) / 0.2); return mix(c5, c6, (t - 0.8) / 0.2); }",
    "vec3 palSeq(float t){",
    "  vec3 c0 = hx(143.,58.,21.), c1 = hx(232.,116.,59.), c2 = hx(246.,207.,168.), c3 = hx(251.,244.,234.);",
    "  if (t < 0.3) return mix(c0, c1, t / 0.3); if (t < 0.65) return mix(c1, c2, (t - 0.3) / 0.35); return mix(c2, c3, (t - 0.65) / 0.35); }",
    "vec3 shade(float v, float pal){ return pal > 0.5 ? palSeq(v) : palDiv(v); }",
    "vec2 tilePx(vec4 lay, float k, vec2 uv){",
    "  float col = mod(k, lay.x), row = floor(k / lay.x); vec2 inner = lay.zw - 2.0;",
    "  return vec2(col * lay.z + 1.5 + uv.x * (inner.x - 1.0), row * lay.w + 1.5 + (1.0 - uv.y) * (inner.y - 1.0)); }",
    "vec4 cylDisc(vec2 uv, vec4 dom, vec3 c, float pxw){     // the cylinder, in the colours of the site",
    "  if (c.z <= 0.0) return vec4(0.0); vec2 x = vec2(mix(dom.x, dom.y, uv.x), mix(dom.z, dom.w, uv.y));",
    "  float d = length(x - c.xy) - c.z;",
    "  float fill = 1.0 - smoothstep(-pxw, pxw, d), ring = 1.0 - smoothstep(0.0, 2.0 * pxw, abs(d));",
    "  return vec4(mix(hx(233.,227.,217.), hx(140.,131.,120.), ring), max(fill, ring)); }",
    "vec2 ripple(vec2 p, vec2 m, float amt, float t){",
    "  if (amt <= 0.0) return p; vec2 v = p - m; float d = length(v);",
    "  return p + (d > 0.5 ? v / d : vec2(0.0)) * sin(d * 0.085 - t * 4.0) * 5.0 * exp(-d / 85.0) * amt; }",
    "float blobR(vec2 p, vec2 b, float t){                  // superellipse radius, 1 on the edge, with a slight breathing",
    "  float th = atan(p.y, p.x);",
    "  float wob = 1.0 + 0.012 * sin(3.0 * th + t * 0.6) + 0.008 * sin(5.0 * th - t * 0.45) + 0.012 * (fbm(vec2(th * 1.3, t * 0.15)) - 0.5);",
    "  vec2 q = abs(p) / (b * wob); return pow(pow(q.x, 2.6) + pow(q.y, 2.6), 1.0 / 2.6); }",
    "float fadeA(vec2 p, vec2 b, float r, float t){          // no edge: the field fades out over the outer third, along a wandering line",
    "  float dpx = (r - 1.0) * min(b.x, b.y), F = 0.34 * min(b.x, b.y) + 16.0;",
    "  float e = dpx + (fbm(p / 70.0 + vec2(t * 0.05, -t * 0.04)) - 0.5) * F * 0.7;",
    "  float a = 1.0 - smoothstep(-F, 0.0, e); return a * a * (3.0 - 2.0 * a); }",
    "vec3 inkify(vec3 col, vec2 p){                          // pigment over paper, granulated, pooling at the edges of a stroke",
    "  vec3 cream = hx(251.,244.,234.), paper = hx(250.,245.,236.) * (0.965 + 0.035 * fbm(p * 0.9));",
    "  float amt = clamp(length(col - cream) / 0.75, 0.0, 1.0);",
    "  float d = clamp(amt * (0.8 + 0.4 * vn(p * 0.55)), 0.0, 1.0);",
    "  float edge = clamp(fwidth(amt) * 9.0, 0.0, 1.0);",
    "  return paper * mix(vec3(1.0), col * (1.0 - 0.35 * edge) / cream, d); }",
    "float tx(int i, vec2 t){ if (i == 0) return texture(uTex0, t).r; if (i == 1) return texture(uTex1, t).r;",
    "  if (i == 2) return texture(uTex2, t).r; if (i == 3) return texture(uTex3, t).r; return texture(uTex4, t).r; }",
    "float frameV(int i, vec2 uv, float f){",
    "  float k0 = floor(f), k1 = min(k0 + 1.0, uN[i] - 1.0);",
    "  return mix(tx(i, tilePx(uLay[i], k0, uv) / uSize[i]), tx(i, tilePx(uLay[i], k1, uv) / uSize[i]), f - k0); }",
    "float val(int i, vec2 uv){",
    "  if (uPer[i] > 0.5) uv.x = fract(uv.x); if (uPer[i] > 1.5) uv.y = fract(uv.y); uv = clamp(uv, 0.0, 1.0);",
    "  float v = frameV(i, uv, uFrame[i]);",
    "  if (uBlend[i] > 0.0) v = mix(v, frameV(i, uv, uFrameB[i]), uBlend[i]);   // the end of a pass dissolves into its start",
    "  return v; }",
    "void main(){",
    "  vec2 p = ripple((gl_FragCoord.xy - 0.5 * uRes) / uDpr, (uMouse - 0.5 * uRes) / uDpr, uRipple, uTime);",
    "  vec2 b = uBox; float r = blobR(p, b, uTime);",
    "  if (uIdMode < 0.5) p += (vec2(fbm(p / 28.0), fbm(p / 28.0 + 7.3)) - 0.5) * 9.0;   // the ink bleeds a little",
    "  vec2 w = vec2(fbm(p / 170.0 + vec2(0.0, uTime * 0.03)), fbm(p / 170.0 + vec2(5.2, -uTime * 0.025))) - 0.5;",
    "  vec2 q = p + w * 150.0;                               // the marbling: distances measured in a warped plane",
    "  float d[5]; float d1 = 1e9, d2 = 1e9; int k1 = 0;",
    "  for (int i = 0; i < 5; i++) { d[i] = length(q - uSeed[i]);",
    "    if (d[i] < d1) { d2 = d1; d1 = d[i]; k1 = i; } else if (d[i] < d2) d2 = d[i]; }",
    "  float a = fadeA(p, b, r, uTime);",
    "  if (uIdMode > 0.5) { o = vec4((float(k1) + 1.0) / 255.0 * step(0.12, a), 0.0, 0.0, 1.0); return; }",
    "  vec3 col = vec3(0.0); float ws = 0.0;",
    "  for (int i = 0; i < 5; i++) {",
    "    float wi = exp(-(d[i] - d1) / 9.0);                // soft borders, about 20 px",
    "    if (wi < 0.01) continue;",
    "    vec2 uv = uAnchor[i] + (p - uSeed[i]) / uScale[i];",
    "    vec3 c = shade(0.5 + (val(i, uv) - 0.5) * uSpan[i], uPal[i]);",
    "    vec4 k = cylDisc(uv, uDom[i], uCyl[i], (uDom[i].y - uDom[i].x) / uScale[i].x * 0.9); c = mix(c, k.rgb, k.a);",
    "    if (uHover >= 0.0 && float(i) != uHover) c = mix(c, hx(251.,244.,234.), 0.4);",
    "    col += wi * c; ws += wi; }",
    "  col /= ws;",
    "  col = mix(col, hx(255.,250.,243.), 0.7 * exp(-(d2 - d1) / 2.2));   // pale veins where two regions meet",
    "  col = inkify(mix(hx(251.,244.,234.), col, 0.7 + 0.3 * a), p);",
    "  float e = clamp(a + (fbm(p / 22.0) - 0.5) * 0.5, 0.0, 1.0); e = e * e * (3.0 - 2.0 * e);   // the wash ends irregularly",
    "  col *= 1.0 - 0.07 * exp(-pow((e - 0.22) / 0.1, 2.0));",
    "  col += (h21(gl_FragCoord.xy + uTime) - 0.5) * 0.02;",
    "  o = vec4(col, e);",
    "}"].join("\n");

  ready(function () {
    var fig = document.getElementById("hero-field");
    if (!fig) return;
    var art = fig.querySelector(".pfh-hf-art"), cv = art && art.querySelector("canvas"), video = fig.querySelector("video");
    function fallback() {                                  // the video of Cyl and its equations, which load only now
      fig.classList.remove("is-live", "is-ready");
      if (video && !(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches)) {
        video.preload = "auto"; video.autoplay = true; var p = video.play(); if (p && p.catch) p.catch(function () {});
      }
    }
    var gl = cv && cv.getContext("webgl2", {premultipliedAlpha: false, antialias: false});
    if (!gl) { fallback(); return; }
    var REDUCE = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
    var base = fig.getAttribute("data-base"), pages = fig.getAttribute("data-pages");
    var eqBox = fig.querySelector(".pfh-hf-eqs"), eqEls = {};

    function sh(type, src) { var s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; }
    var prog = gl.createProgram();
    try {
      gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    } catch (err) { fallback(); return; }
    gl.useProgram(prog);
    fig.classList.add("is-live");                          // the place of the video is kept for the field, which fades in once loaded
    var buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    var loc = gl.getAttribLocation(prog, "a"); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    var U = {};
    ["uRes","uTime","uRipple","uDpr","uMouse","uBox","uHover","uIdMode","uTex0","uTex1","uTex2","uTex3","uTex4",
     "uLay","uDom","uSize","uSeed","uScale","uAnchor","uN","uFrame","uFrameB","uBlend","uPer","uPal","uSpan","uCyl"]
      .forEach(function (n) { U[n] = gl.getUniformLocation(prog, n); });

    var M, tex = [];
    function texture(i) {
      return new Promise(function (res, rej) {
        var img = new Image(); img.onerror = rej; img.onload = function () {
          var t = gl.createTexture(); gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, t);
          gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, gl.RED, gl.UNSIGNED_BYTE, img);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
          tex[i] = t; res();
        }; img.src = base + ORDER[i] + ".png";
      });
    }
    fetch(base + "fields.json").then(function (r) { if (!r.ok) throw new Error("fields"); return r.json(); }).then(function (meta) {
      M = meta;
      return Promise.all(ORDER.map(function (c, i) { return texture(i); }));   // about 0.55 MB in all
    }).then(function () {
      fig.classList.add("is-ready");
      // the video of Cyl stays unloaded
      start();
    }).catch(fallback);

    function equations() {                                 // the equation of each problem, shown under the field on hover
      if (!window.katex) return setTimeout(equations, 150);
      ORDER.forEach(function (c) {
        var d = document.createElement("div"); d.className = "pfh-hf-eq";
        katex.render(M[c].tex, d, {displayMode: false, throwOnError: false});
        eqBox.appendChild(d); eqEls[c] = d;
      });
      fit(); window.addEventListener("resize", fit);
    }
    function fit() {                                       // a long equation shrinks to one line
      ORDER.forEach(function (c) { var d = eqEls[c], k = d && d.firstChild; if (!k) return; d.style.fontSize = "";
        for (var f = 15; f > 9 && k.offsetWidth > d.clientWidth; f -= 0.5) d.style.fontSize = f + "px"; });
    }
    var shown = null;
    function show(i) {                                     // the problem under the cursor: its equation, and a link
      var c = i >= 0 ? ORDER[i] : null;
      if (c === shown) return;
      shown = c;
      ORDER.forEach(function (x) { if (eqEls[x]) eqEls[x].classList.toggle("on", x === c); });
      cv.classList.toggle("link", !!c);
      cv.setAttribute("aria-label", c ? M[c].name + ", " + SUBT[c] + ", predicted by its discovered PINN solver" : LABEL);
    }
    var LABEL = cv.getAttribute("aria-label");

    function uniforms(t, W, H) {
      // the blob, and with it each window onto a field, scales with the canvas (215 px: the half width they were set for)
      var b = [W * 0.4, H * 0.4], z = b[0] / 215, seeds = [], lay = [], dom = [], size = [], scale = [], anchor = [], n = [],
          fr = [], frB = [], bl = [], per = [], pal = [], span = [], cyl = [];
      ORDER.forEach(function (c, i) {
        var m = M[c], L = LOOK[c];
        // the regions turn about the centre (spin: slowly, faster after a drag), each breathing in and out a little
        var th = 2 * Math.PI * i / ORDER.length + spin + 0.25 * Math.sin(t * 0.07 + i * 1.7), rr = 0.5 + 0.08 * Math.sin(t * 0.11 + i);
        seeds.push(Math.cos(th) * b[0] * rr, Math.sin(th) * b[1] * rr);
        lay.push(m.cols, m.rows, m.tile[0], m.tile[1]); size.push(m.size[0], m.size[1]);
        dom.push(m.domain[0], m.domain[1], m.domain[2], m.domain[3]);
        scale.push(L.scale[0] * z, L.scale[1] * z); anchor.push(L.anchor[0], L.anchor[1]);
        // a pass from the first frame to the last, the last fifth dissolving into the first, so it never plays backwards
        var f0 = L.from, last = m.frames - 1, u = (t / L.loop + i * 0.37) % 1;
        n.push(m.frames); fr.push(f0 + u * (last - f0)); frB.push(f0); bl.push(u > 0.8 ? (u - 0.8) / 0.2 : 0);
        per.push(L.per); pal.push(L.pal); span.push(L.span);
        var k = m.cylinder || [0, 0, 0]; cyl.push(k[0], k[1], k[2]);
        gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, tex[i]); gl.uniform1i(U["uTex" + i], i);
      });
      gl.uniform2fv(U.uSeed, seeds); gl.uniform4fv(U.uLay, lay); gl.uniform4fv(U.uDom, dom); gl.uniform2fv(U.uSize, size);
      gl.uniform2fv(U.uScale, scale); gl.uniform2fv(U.uAnchor, anchor); gl.uniform1fv(U.uN, n); gl.uniform1fv(U.uFrame, fr);
      gl.uniform1fv(U.uFrameB, frB); gl.uniform1fv(U.uBlend, bl); gl.uniform1fv(U.uPer, per); gl.uniform1fv(U.uPal, pal);
      gl.uniform1fv(U.uSpan, span); gl.uniform3fv(U.uCyl, cyl); gl.uniform2f(U.uBox, b[0], b[1]);
    }

    var fbo = null, fboTex = null, fboW = 0, fboH = 0, px = new Uint8Array(4);
    function ensureFbo(w, h) {                             // a target for the pass that tells the region under the cursor
      if (fbo && fboW === w && fboH === h) return;
      if (!fbo) { fbo = gl.createFramebuffer(); fboTex = gl.createTexture(); }
      gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D, fboTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, fboTex, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null); fboW = w; fboH = h;
    }

    var clock = 0, last = 0, mouse = [-9999, -9999], rip = 0, raf = 0, visible = true, hover = -1;
    // dragging spins the field (author, 2026-11-02): while held it turns with the pointer about the centre, and on
    // release it keeps the speed of the drag, which decays back to the slow turn of the regions; a drag is no click
    var TURN = 0.035, spin = 0, omega = 0, drag = null, dragged = false;
    function angleAt(e) {                                  // the angle of the pointer about the centre, y up as in the shader
      var r = cv.getBoundingClientRect(), x = e.clientX - (r.left + r.width / 2), y = (r.top + r.height / 2) - e.clientY;
      return {a: Math.atan2(y, x), r: Math.sqrt(x * x + y * y)};
    }
    function start() {
      equations();
      cv.addEventListener("pointerdown", function (e) {
        if (e.button > 0) return;
        var g = angleAt(e);
        drag = {a: g.a, t: performance.now(), x: e.clientX, y: e.clientY, path: 0, v: 0};
        dragged = false;
        try { cv.setPointerCapture(e.pointerId); } catch (x) { /* no capture */ }
      });
      cv.addEventListener("pointermove", function (e) { var r = cv.getBoundingClientRect(), k = cv.width / r.width;
        mouse = [(e.clientX - r.left) * k, (r.bottom - e.clientY) * k];
        if (drag) {
          var g = angleAt(e), now = performance.now(), da = g.a - drag.a;
          da -= 2 * Math.PI * Math.round(da / (2 * Math.PI));        // across the cut at plus and minus pi
          drag.path += Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y);
          drag.x = e.clientX; drag.y = e.clientY; drag.a = g.a;
          if (g.r > 24) {                                   // near the centre the angle jumps: no turn there
            spin += da;
            drag.v = 0.6 * drag.v + 0.4 * da / Math.max(0.008, (now - drag.t) / 1000);
          }
          drag.t = now;
          if (drag.path > 6 && !dragged) { dragged = true; cv.classList.add("dragging"); }
        }
        run(); });
      function release() {
        if (!drag) return;
        var still = performance.now() - drag.t > 120;      // held still before letting go: no throw
        omega = REDUCE || still || !dragged ? 0 : Math.max(-9, Math.min(9, drag.v - TURN));
        drag = null; cv.classList.remove("dragging"); run();
      }
      cv.addEventListener("pointerup", release);
      cv.addEventListener("pointercancel", release);
      cv.addEventListener("pointerleave", function () { if (!drag) { mouse = [-9999, -9999]; run(); } });
      cv.addEventListener("click", function () {
        if (dragged) { dragged = false; return; }
        if (shown) window.location.href = pages + shown + ".html";
      });
      if (window.IntersectionObserver) new IntersectionObserver(function (es) { visible = es[0].isIntersecting; if (visible) run(); }).observe(cv);
      document.addEventListener("visibilitychange", function () { if (!document.hidden) run(); });
      window.addEventListener("resize", run);
      clock = REDUCE ? 20 : 0;                             // with reduced motion, one moment, drawn again only when needed
      spin = clock * TURN;
      run();
    }
    function run() { if (!raf) raf = requestAnimationFrame(frame); }
    function frame(now) {
      raf = 0;
      if (!visible || document.hidden) { last = 0; return; }        // paused: the clock does not run on
      if (!REDUCE) {
        var dt = last ? Math.min(0.1, (now - last) / 1000) : 0;
        clock += dt; last = now;
        if (!drag) { spin += (TURN + omega) * dt; omega *= Math.exp(-dt / 1.6); if (Math.abs(omega) < 0.002) omega = 0; }
      }
      var dpr = Math.min(2, window.devicePixelRatio || 1), W = cv.clientWidth, H = cv.clientHeight;
      if (!W || !H) { raf = requestAnimationFrame(frame); return; }
      if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
      rip += ((!REDUCE && mouse[0] > -999 ? 1 : 0) - rip) * 0.08;
      gl.uniform2f(U.uRes, cv.width, cv.height); gl.uniform1f(U.uTime, clock); gl.uniform1f(U.uDpr, dpr);
      gl.uniform1f(U.uRipple, rip); gl.uniform2f(U.uMouse, mouse[0], mouse[1]);
      uniforms(clock, W, H);
      hover = -1;                                          // the region under the cursor: one pixel of an id pass
      if (mouse[0] > -999) {
        ensureFbo(cv.width, cv.height);
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo); gl.viewport(0, 0, cv.width, cv.height);
        var mx = Math.max(0, Math.min(cv.width - 1, Math.round(mouse[0]))), my = Math.max(0, Math.min(cv.height - 1, Math.round(mouse[1])));
        gl.enable(gl.SCISSOR_TEST); gl.scissor(mx, my, 1, 1);
        gl.uniform1f(U.uIdMode, 1); gl.uniform1f(U.uHover, -1);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        gl.readPixels(mx, my, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
        gl.disable(gl.SCISSOR_TEST);
        hover = px[0] - 1;
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, cv.width, cv.height); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform1f(U.uIdMode, 0); gl.uniform1f(U.uHover, hover);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      show(hover);
      if (!REDUCE || rip > 0.01) raf = requestAnimationFrame(frame);
    }
  });
})();
