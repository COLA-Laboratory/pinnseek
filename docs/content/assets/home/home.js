/* Landing page: the cards of the four numbers under the hero, figures that open in the lightbox of pf.js, and the
   button back to the top. */
(function () {
  function ready(fn) { if (document.readyState !== "loading") fn(); else document.addEventListener("DOMContentLoaded", fn); }
  ready(function () {
    // back to top (author, 2026-11-02): shown once the hero is out of view, a smooth scroll unless motion is reduced
    var top = document.querySelector(".pfh-top");
    if (top) {
      var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches, on = false;
      var foot = document.querySelector(".md-footer"), gap = parseFloat(getComputedStyle(top).bottom) || 24;
      var check = function () {
        // once the footer comes into view the button rides up with it, so it never covers the lab's logo
        var over = foot ? Math.max(0, window.innerHeight - foot.getBoundingClientRect().top) : 0;
        top.style.bottom = (gap + over) + "px";
        var want = window.scrollY > window.innerHeight * 0.8;
        if (want === on) return;
        on = want;
        top.classList.toggle("on", on);
        top.tabIndex = on ? 0 : -1;
      };
      window.addEventListener("scroll", check, {passive: true});
      window.addEventListener("resize", check);
      if (window.ResizeObserver) new ResizeObserver(check).observe(document.body);   // images that load later move the footer
      check();
      top.addEventListener("click", function () {
        window.scrollTo({top: 0, behavior: reduce ? "auto" : "smooth"});
        top.blur();
      });
    }
    // the four numbers under the hero: a card glides to the one under the cursor, the others fade a little
    var stats = document.querySelector(".pfh-stats");
    if (stats && window.matchMedia && window.matchMedia("(hover: hover)").matches) {
      var glow = document.createElement("span");
      glow.className = "pfh-stats-glow";
      glow.setAttribute("aria-hidden", "true");
      stats.insertBefore(glow, stats.firstChild);
      var place = function (cell) {
        glow.style.width = (cell.offsetWidth + 10) + "px";       // 14 px around the text on each side, 14 px clear of the next
        glow.style.height = (cell.offsetHeight + 10) + "px";     // column's text (gap 10 + padding 18); 16 px above and below
        glow.style.transform = "translate(" + (cell.offsetLeft - 14) + "px, " + (cell.offsetTop + 6) + "px)";
      };
      [].slice.call(stats.children).forEach(function (cell) {
        if (cell === glow) return;
        cell.addEventListener("mouseenter", function () {
          if (!stats.classList.contains("is-hot")) {              // from outside: appear in place, then glide between cells
            glow.style.transition = "none";
            place(cell);
            void glow.offsetWidth;
            glow.style.transition = "";
          } else place(cell);
          stats.classList.add("is-hot");
          [].slice.call(stats.children).forEach(function (c) { c.classList.toggle("is-on", c === cell); });
        });
      });
      stats.addEventListener("mouseleave", function () {
        stats.classList.remove("is-hot");
        [].slice.call(stats.children).forEach(function (c) { c.classList.remove("is-on"); });
      });
    }
    var box = document.getElementById("lightbox");
    if (box) {
      document.querySelectorAll(".pfh-zoom img").forEach(function (img) {
        img.addEventListener("click", function () { box.querySelector("img").src = img.src; box.classList.add("open"); });
      });
    }
  });
})();
