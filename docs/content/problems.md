---
title: Benchmark
template: pf.html
hide:
  - navigation
  - toc
pf_libs: [math]
---

<div class="pf case">
<section class="hero"><div class="wrap">
<h1>18 PDE problems</h1>
<p class="lede">From incompressible flows and nonlinear waves to chaotic dynamics, pattern formation and atmospheric transport. Each card opens the definition of the problem, the fields of the discovered PINN solver against the reference solution, and its error against four state-of-the-art methods.</p>
</div></section>
<section><div class="wrap">
<div class="grid"><a class="card" href="problems/poissoncg.html">
<div class="thumb"><img src="assets/img/covers/poissoncg.webp?v=cb663bb8" alt="Poisson, complex geometry" width="720" height="405" loading="lazy"></div>
<div class="body"><div class="name">Poisson-CG <span class="pill time">60 s</span> </div><div class="long">Poisson, complex geometry</div>
<dl class="mini"><dt>Discovered</dt><dd><b>(9.96 ± 0.11) × 10<sup>−4</sup></b></dd><dt>SPINN</dt><dd>0.0215 ± 0.0022</dd></dl>
</div></a><a class="card" href="problems/ib.html">
<div class="thumb"><img src="assets/img/covers/ib.webp?v=3ecfbbbc" alt="Inviscid Burgers" width="720" height="405" loading="lazy"></div>
<div class="body"><div class="name">IB <span class="pill time">100 s</span> <span class="pill anim">animation</span></div><div class="long">Inviscid Burgers</div>
<dl class="mini"><dt>Discovered</dt><dd><b>(1.79 ± 0.48) × 10<sup>−5</sup></b></dd><dt>Scale-PINN</dt><dd>0.192 ± 0.025</dd></dl>
</div></a><a class="card" href="problems/kdv.html">
<div class="thumb"><img src="assets/img/covers/kdv.webp?v=b459ee54" alt="Korteweg–de Vries" width="720" height="405" loading="lazy"></div>
<div class="body"><div class="name">KdV <span class="pill time">100 s</span> <span class="pill anim">animation</span></div><div class="long">Korteweg–de Vries</div>
<dl class="mini"><dt>Discovered</dt><dd><b>0.00204 ± 0.00025</b></dd><dt>Scale-PINN</dt><dd>0.00707 ± 0.00089</dd></dl>
</div></a><a class="card" href="problems/wavems.html">
<div class="thumb"><img src="assets/img/covers/wavems.webp?v=4f3d0a09" alt="Wave, heterogeneous media" width="720" height="405" loading="lazy"></div>
<div class="body"><div class="name">Wave-HM <span class="pill time">100 s</span> <span class="pill anim">animation</span></div><div class="long">Wave, heterogeneous media</div>
<dl class="mini"><dt>Discovered</dt><dd><b>0.0992 ± 0.0756</b></dd><dt>Scale-PINN</dt><dd>0.903 ± 0.085</dd></dl>
</div></a><a class="card" href="problems/ldc.html">
<div class="thumb"><img src="assets/img/covers/ldc.webp?v=36a6841c" alt="Lid-driven cavity flow" width="720" height="405" loading="lazy"></div>
<div class="body"><div class="name">LDC <span class="pill time">150 s</span> </div><div class="long">Lid-driven cavity flow</div>
<dl class="mini"><dt>Discovered</dt><dd><b>0.0269 ± 0.0109</b></dd><dt>Scale-PINN</dt><dd>0.0658 ± 0.0171</dd></dl>
</div></a><a class="card" href="problems/bfs.html">
<div class="thumb"><img src="assets/img/covers/bfs.webp?v=587b2174" alt="Backward-facing step flow" width="720" height="405" loading="lazy"></div>
<div class="body"><div class="name">BFS <span class="pill time">180 s</span> </div><div class="long">Backward-facing step flow</div>
<dl class="mini"><dt>Discovered</dt><dd><b>0.0229 ± 0.0061</b></dd><dt>DSGNAR</dt><dd>0.534 ± 0.092</dd></dl>
</div></a><a class="card" href="problems/heatlt.html">
<div class="thumb"><img src="assets/img/covers/heatlt.webp?v=1cba9f35" alt="Long-time heat equation" width="720" height="405" loading="lazy"></div>
<div class="body"><div class="name">Heat-LT <span class="pill time">180 s</span> <span class="pill anim">animation</span></div><div class="long">Long-time heat equation</div>
<dl class="mini"><dt>Discovered</dt><dd><b>0.220 ± 0.252</b></dd><dt>Scale-PINN</dt><dd>0.268 ± 0.012</dd></dl>
</div></a><a class="card" href="problems/naca.html">
<div class="thumb"><img src="assets/img/covers/naca.webp?v=2963f3ea" alt="NACA0012 airfoil flow" width="720" height="405" loading="lazy"></div>
<div class="body"><div class="name">NACA <span class="pill time">180 s</span> </div><div class="long">NACA0012 airfoil flow</div>
<dl class="mini"><dt>Discovered</dt><dd><b>0.0129 ± 0.0012</b></dd><dt>Scale-PINN</dt><dd>0.0434 ± 0.0168</dd></dl>
</div></a><a class="card" href="problems/smoke.html">
<div class="thumb"><img src="assets/img/covers/smoke.webp?v=85125fa0" alt="Wildfire smoke transport" width="720" height="405" loading="lazy"></div>
<div class="body"><div class="name">Smoke <span class="pill time">180 s</span> <span class="pill anim">animation</span></div><div class="long">Wildfire smoke transport</div>
<dl class="mini"><dt>Discovered</dt><dd><b>0.0637 ± 0.0002</b></dd><dt>Scale-PINN</dt><dd>0.316 ± 0.013</dd></dl>
</div></a><a class="card" href="problems/cyl.html">
<div class="thumb"><img src="assets/img/covers/cyl.webp?v=4821afa5" alt="Cylinder vortex street" width="720" height="405" loading="lazy"></div>
<div class="body"><div class="name">Cyl <span class="pill time">300 s</span> <span class="pill anim">animation</span></div><div class="long">Cylinder vortex street</div>
<dl class="mini"><dt>Discovered</dt><dd><b>0.0226 ± 0.0029</b></dd><dt>Scale-PINN</dt><dd>0.476 ± 0.082</dd></dl>
</div></a><a class="card" href="problems/gl.html">
<div class="thumb"><img src="assets/img/covers/gl.webp?v=4123e2dc" alt="Ginzburg–Landau" width="720" height="405" loading="lazy"></div>
<div class="body"><div class="name">GL <span class="pill time">300 s</span> <span class="pill anim">animation</span></div><div class="long">Ginzburg–Landau</div>
<dl class="mini"><dt>Discovered</dt><dd><b>0.00702 ± 0.00124</b></dd><dt>APTS</dt><dd>0.995 ± 0.001</dd></dl>
</div></a><a class="card" href="problems/gs.html">
<div class="thumb"><img src="assets/img/covers/gs.webp?v=9052a5e1" alt="Gray–Scott" width="720" height="405" loading="lazy"></div>
<div class="body"><div class="name">GS <span class="pill time">300 s</span> <span class="pill anim">animation</span></div><div class="long">Gray–Scott</div>
<dl class="mini"><dt>Discovered</dt><dd><b>0.00212 ± 0.00088</b></dd><dt>Scale-PINN</dt><dd>0.0246 ± 0.0046</dd></dl>
</div></a><a class="card" href="problems/kf.html">
<div class="thumb"><img src="assets/img/covers/kf.webp?v=931f5e8a" alt="Kolmogorov flow" width="720" height="405" loading="lazy"></div>
<div class="body"><div class="name">KF <span class="pill time">300 s</span> <span class="pill anim">animation</span></div><div class="long">Kolmogorov flow</div>
<dl class="mini"><dt>Discovered</dt><dd><b>0.0152 ± 0.0002</b></dd><dt>Scale-PINN</dt><dd>0.373 ± 0.010</dd></dl>
</div></a><a class="card" href="problems/ks.html">
<div class="thumb"><img src="assets/img/covers/ks.webp?v=ea3b69f4" alt="Kuramoto–Sivashinsky" width="720" height="405" loading="lazy"></div>
<div class="body"><div class="name">KS <span class="pill time">300 s</span> <span class="pill anim">animation</span></div><div class="long">Kuramoto–Sivashinsky</div>
<dl class="mini"><dt>Discovered</dt><dd><b>0.00854 ± 0.00450</b></dd><dt>Scale-PINN</dt><dd>0.0785 ± 0.0172</dd></dl>
</div></a><a class="card" href="problems/ns4d.html">
<div class="thumb"><img src="assets/img/covers/ns4d.webp?v=0270dd7a" alt="3D unsteady Navier–Stokes" width="720" height="405" loading="lazy"></div>
<div class="body"><div class="name">NS4D <span class="pill time">300 s</span> <span class="pill anim">animation</span></div><div class="long">3D unsteady Navier–Stokes</div>
<dl class="mini"><dt>Discovered</dt><dd><b>(2.82 ± 0.11) × 10<sup>−7</sup></b></dd><dt>SPINN</dt><dd>0.0173 ± 0.0072</dd></dl>
</div></a><a class="card" href="problems/cyl2.html">
<div class="thumb"><img src="assets/img/covers/cyl2.webp?v=6425c9ff" alt="Long-time vortex street" width="720" height="405" loading="lazy"></div>
<div class="body"><div class="name">Cyl-LT <span class="pill time">480 s</span> <span class="pill anim">animation</span></div><div class="long">Long-time vortex street</div>
<dl class="mini"><dt>Discovered</dt><dd><b>0.101 ± 0.001</b></dd><dt>Scale-PINN</dt><dd>0.373 ± 0.062</dd></dl>
</div></a><a class="card" href="problems/rt.html">
<div class="thumb"><img src="assets/img/covers/rt.webp?v=d0923a79" alt="Rayleigh–Taylor instability" width="720" height="405" loading="lazy"></div>
<div class="body"><div class="name">RT <span class="pill time">480 s</span> <span class="pill anim">animation</span></div><div class="long">Rayleigh–Taylor instability</div>
<dl class="mini"><dt>Discovered</dt><dd><b>0.00618 ± 0.00119</b></dd><dt>SPINN</dt><dd>0.466 ± 0.008</dd></dl>
</div></a><a class="card" href="problems/wing3d.html">
<div class="thumb"><img src="assets/img/covers/wing3d.webp?v=25dd621b" alt="Flow over a 3D wing" width="720" height="405" loading="lazy"></div>
<div class="body"><div class="name">Wing3D <span class="pill time">480 s</span> </div><div class="long">Flow over a 3D wing</div>
<dl class="mini"><dt>Discovered</dt><dd><b>0.0321 ± 0.0044</b></dd><dt>Scale-PINN</dt><dd>0.0472 ± 0.0049</dd></dl>
</div></a></div>
</div></section>
</div>
