/* Registers sw.js at the root of this site (see there why), and loads again the videos that failed while the
   service worker of colalab.ai still answered this page's requests (the first visit after the lab homepage). */
(function () {
  if (!("serviceWorker" in navigator) || !document.currentScript) return;
  var root = new URL("../../", document.currentScript.src);          // this script sits in <root>/assets/js/
  function retry() {
    document.querySelectorAll("video").forEach(function (v) {
      if (v.networkState === HTMLMediaElement.NETWORK_NO_SOURCE || v.error) v.load();
    });
  }
  navigator.serviceWorker.addEventListener("controllerchange", retry);
  navigator.serviceWorker.register(new URL("sw.js", root).href, {scope: root.pathname}).catch(function () {});
})();
