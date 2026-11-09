/* A service worker for the pages of this site that handles no requests.
   colalab.ai registers its own service worker (/sw.js, scope "/", from the lab homepage), which answers every request
   of the domain: it stores each response in the Cache API, which rejects the partial (206) responses of ranged video
   requests, and then has nothing to answer with, so videos never load once a visitor has seen the homepage. This
   registration, with the narrower scope of this site, takes over its pages; having no fetch handler, it lets their
   requests go straight to the network. Registered by assets/js/sw-register.js. */
self.addEventListener("install", function () { self.skipWaiting(); });
self.addEventListener("activate", function (event) { event.waitUntil(self.clients.claim()); });
