/* The benchmark gallery of the landing page as a ring of cards with some thickness, turning slowly in 3D. The card
   passing the centre grows and lights up (a glowing rim and edge, a sweep of light and rising sparks); hovering slows the
   ring to a stop, dragging turns it and it settles on the nearest card like a ratchet, a click opens the problem, and
   focusing a card brings it to the front; the arrow keys move between cards. Without 3D transforms the plain grid of
   cards stays. */
(function () {
  "use strict";
  function ready(fn) { if (document.readyState !== "loading") fn(); else document.addEventListener("DOMContentLoaded", fn); }

  ready(function () {
    var grid = document.querySelector(".pfh-cases");
    if (!grid || !(window.CSS && CSS.supports && CSS.supports("transform-style", "preserve-3d"))) return;
    var cards = [].slice.call(grid.querySelectorAll(".pfh-case")), N = cards.length;
    if (N < 6) return;
    var REDUCE = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
    var STEP = 360 / N, TILT = -10, SPEED = REDUCE ? 0 : -6;     // degrees; a turn a minute

    var stage = document.createElement("div");
    stage.className = "pfh-ring-stage";
    stage.setAttribute("role", "region");
    stage.setAttribute("aria-label", N + " problems on a turning ring. Drag to turn it, use the arrow keys to move between problems.");
    var ring = document.createElement("div");
    ring.className = "pfh-ring";
    var floor = document.createElement("div");
    floor.className = "pfh-ring-floor";
    floor.setAttribute("aria-hidden", "true");
    ring.appendChild(floor);
    cards.forEach(function (a) {
      a.classList.add("pfh-ring-card");
      a.setAttribute("draggable", "false");
      var face = document.createElement("span"), back = document.createElement("span");
      face.className = "pfh-ring-face";
      back.className = "pfh-ring-back";
      back.setAttribute("aria-hidden", "true");
      while (a.firstChild) face.appendChild(a.firstChild);
      var halo = document.createElement("span");               // the glow, ripple and sparks behind the centre card
      halo.className = "pfh-ring-halo";
      halo.setAttribute("aria-hidden", "true");
      for (var s = 0; s < 9; s++) {
        var sp = document.createElement("i"), side = s % 2 ? 1 : -1;
        sp.style.left = (s % 3 === 2 ? 20 + Math.random() * 60 : side > 0 ? 88 + Math.random() * 10 : 2 + Math.random() * 10) + "%";
        sp.style.animationDelay = (s * 0.27 + Math.random() * 0.2).toFixed(2) + "s";
        sp.style.setProperty("--dx", (side * (6 + Math.random() * 16)).toFixed(0) + "px");
        halo.appendChild(sp);
      }
      a.appendChild(halo);
      var shadow = document.createElement("span");             // its soft shadow on the floor
      shadow.className = "pfh-ring-shadow";
      shadow.setAttribute("aria-hidden", "true");
      a.appendChild(shadow);
      for (var k = 1; k <= 8; k++) {                           // the edge of the card, slice by slice
        var sl = document.createElement("span");
        sl.className = "pfh-ring-slab";
        sl.style.setProperty("--k", k);
        a.appendChild(sl);
      }
      var img = face.querySelector("img"), name = face.querySelector(".pfh-case-name");
      if (img) {
        img.loading = "eager";
        img.setAttribute("draggable", "false");
        back.style.backgroundImage = "url('" + img.getAttribute("src") + "')";
      }
      var b = document.createElement("b");
      b.textContent = name && name.firstChild ? name.firstChild.textContent : "";
      back.appendChild(b);
      a.appendChild(face);
      a.appendChild(back);
      ring.appendChild(a);
    });
    stage.appendChild(ring);
    grid.parentNode.insertBefore(stage, grid);
    grid.parentNode.removeChild(grid);

    // sizes from the width: the card width, the radius that seats the cards side by side, the stage around them
    var R = 600, cw = 200, ch = 210, cy = 200;
    function layout() {
      var W = stage.clientWidth || 1000;
      cw = Math.max(150, Math.min(200, W * 0.165));
      ch = Math.round(cw * 9 / 16 + 98);
      R = (cw + 32) / (2 * Math.tan(Math.PI / N));            // a gap of 32 px between neighbouring cards
      var p = 2.6 * R, sTilt = Math.sin(-TILT * Math.PI / 180), back = p / (p + 2 * R);
      cy = Math.round(28 + sTilt * R * back + back * ch / 2);
      stage.style.height = Math.round(cy + sTilt * R + ch / 2 + 76) + "px";
      stage.style.perspective = Math.round(p) + "px";
      stage.style.perspectiveOrigin = "50% " + cy + "px";
      ring.style.top = cy + "px";
      stage.style.setProperty("--cw", Math.round(cw) + "px");
      stage.style.setProperty("--ch", ch + "px");
      stage.style.setProperty("--R", Math.round(R) + "px");
      cards.forEach(function (a, i) { a.style.transform = "rotateY(" + (i * STEP) + "deg) translateZ(" + R.toFixed(1) + "px)"; });
    }
    layout();
    window.addEventListener("resize", layout);

    var theta = 0, v = SPEED, hover = false, goal = null, drag = null, last = 0, raf = 0, visible = true, mute = 0;
    var lift = cards.map(function () { return 0; }), over = -1, front = -1;
    // on its own the ring turns slowly and evenly; a drag ends like a ratchet, carried on by the throw to the nearest
    // card, fast at first, a little past it and back, before the ring picks up its pace again (author, 2026-10-23)
    var move = null;
    function wrap(d) { return ((d % 360) + 540) % 360 - 180; }
    function click(p) { var q = p - 1; return 1 + 2.3 * q * q * q + 1.3 * q * q; }   // ease out with an overshoot of about 5%
    function go(to, t, s) { move = {from: theta, to: to, t0: t, s: s}; }
    function frame(t) {
      var dt = last ? Math.min(0.05, (t - last) / 1000) : 0;
      last = t;
      if (goal !== null && !drag) {                            // a focused card comes to the front
        var d = wrap(goal - theta);
        theta += d * Math.min(1, dt * 7);
        move = null;
        v = 0;
      } else if (move && !drag) {                              // the settling after a drag
        var p = Math.min(1, (t - move.t0) / (move.s * 1000));
        theta = move.from + (move.to - move.from) * click(p);
        if (p >= 1) { theta = move.to; move = null; v = 0; }
      } else if (!drag) {                                      // eases to a stop under the mouse, back to its pace after
        v += ((hover ? 0 : SPEED) - v) * Math.min(1, dt * (hover ? 4 : 1.2));
        theta += v * dt;
      }
      ring.style.transform = "translateZ(" + (-R).toFixed(1) + "px) rotateX(" + TILT + "deg) rotateY(" + theta.toFixed(3) + "deg)";
      var best = 0, bestC = -1;
      for (var i = 0; i < N; i++) {
        var ang = wrap(i * STEP + theta), f = (Math.cos(ang * Math.PI / 180) + 1) / 2;   // 1 at the front, 0 at the back
        var c = Math.max(0, 1 - Math.abs(ang) / STEP);        // 1 at the centre, 0 one place away
        c = c * c * (3 - 2 * c);
        lift[i] += ((i === over ? 1 : 0) - lift[i]) * Math.min(1, dt * 12);
        cards[i].style.setProperty("--f", f.toFixed(3));
        cards[i].style.setProperty("--c", c.toFixed(3));
        cards[i].style.setProperty("--h", lift[i].toFixed(3));
        if (c > bestC) { bestC = c; best = i; }
      }
      if (best !== front) {                                    // the centre card's effects start as it arrives
        if (front >= 0) cards[front].classList.remove("is-front");
        cards[best].classList.add("is-front");
        front = best;
      }
      raf = visible ? requestAnimationFrame(frame) : 0;
    }
    function run() { if (!raf) { last = 0; raf = requestAnimationFrame(frame); } }
    if (window.IntersectionObserver) {
      new IntersectionObserver(function (es) {
        visible = es[0].isIntersecting;
        if (visible) run();
      }).observe(stage);
    }
    run();

    stage.addEventListener("pointerenter", function (e) { if (e.pointerType === "mouse") hover = true; });
    stage.addEventListener("pointerleave", function (e) { if (e.pointerType === "mouse") { hover = false; over = -1; } });
    stage.addEventListener("pointerover", function (e) {
      if (e.pointerType === "mouse") over = cards.indexOf(e.target.closest ? e.target.closest(".pfh-ring-card") : null);
    });
    stage.addEventListener("pointerdown", function (e) {
      if (e.button !== 0) return;
      drag = {x: e.clientX, th: theta, moved: false, lx: e.clientX, lt: performance.now(), vel: 0};
    });
    stage.addEventListener("pointermove", function (e) {
      if (!drag) return;
      var dx = e.clientX - drag.x;
      if (!drag.moved && Math.abs(dx) > 6) {                   // a drag, not a click: the stage takes the pointer
        drag.moved = true;
        goal = null;
        move = null;                                           // a step under way stops where it is
        drag.th = theta; drag.x = e.clientX; dx = 0;
        try { stage.setPointerCapture(e.pointerId); } catch (err) { /* not capturable */ }
        stage.classList.add("dragging");
      }
      if (drag.moved) {
        var now = performance.now(), k = 180 / (Math.PI * R);  // the front card follows the pointer
        theta = drag.th + dx * k;
        drag.vel = 0.7 * drag.vel + 0.3 * (e.clientX - drag.lx) * k / Math.max(1, now - drag.lt) * 1000;
        drag.lx = e.clientX;
        drag.lt = now;
      }
    });
    function end() {
      if (!drag) return;
      if (drag.moved) {                                        // carried on by the throw, it settles on the nearest card
        var vel = Math.max(-160, Math.min(160, drag.vel)), to = Math.round((theta + vel * 0.3) / STEP) * STEP;
        go(to, performance.now(), Math.max(0.45, Math.min(1.1, 0.35 + Math.abs(to - theta) / 160)));
        mute = performance.now();
      }
      drag = null;
      stage.classList.remove("dragging");
    }
    stage.addEventListener("pointerup", end);
    stage.addEventListener("pointercancel", end);
    stage.addEventListener("click", function (e) {           // the click that ends a drag opens nothing
      if (performance.now() - mute < 350) { e.preventDefault(); e.stopPropagation(); }
    }, true);
    stage.addEventListener("focusin", function (e) {
      var i = cards.indexOf(e.target.closest ? e.target.closest(".pfh-ring-card") : null);
      if (i >= 0) { goal = theta + wrap(-i * STEP - theta); over = i; }
    });
    stage.addEventListener("focusout", function (e) { if (!stage.contains(e.relatedTarget)) { goal = null; over = -1; } });
    stage.addEventListener("keydown", function (e) {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      var i = cards.indexOf(document.activeElement);
      if (i < 0) return;
      e.preventDefault();
      cards[(i + (e.key === "ArrowRight" ? 1 : N - 1)) % N].focus();
    });
  });
})();
