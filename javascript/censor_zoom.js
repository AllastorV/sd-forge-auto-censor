// Wheel zoom-to-cursor + pan + per-panel "fit / center" for the Auto-Censor images.
//   • Detected / Result : wheel zoom to cursor, drag to pan.
//   • Input editor      : wheel zoom to cursor (zoom while painting). No drag-pan
//                         there (mousedown = paint) — use the ⛶ button to reset.
//   • Each panel has its own ⛶ Fit / center button (resets only that panel).
// Wheel is handled in the CAPTURE phase because the theme/Gradio swallow it in
// the bubbling phase (otherwise the whole page just scrolls).
(function () {
  "use strict";
  const MAX = 8;
  const PANELS = [
    { host: "ac_preview", sel: "img", pan: true },
    { host: "ac_result", sel: "img", pan: true },
    { host: "auto_censor_input", sel: ".stage-wrap", pan: false },
  ];

  function panelOf(node) {
    if (!node || !node.closest) return null;
    for (let i = 0; i < PANELS.length; i++) {
      if (node.closest("#" + PANELS[i].host)) return PANELS[i];
    }
    return null;
  }
  function elOf(p) {
    const host = document.querySelector("#" + p.host);
    return host ? host.querySelector(p.sel) : null;
  }
  function st(el) { if (!el._acz) el._acz = { s: 1, x: 0, y: 0 }; return el._acz; }
  function draw(el, z) {
    if (el.parentElement) el.parentElement.style.overflow = "hidden";
    // flatten the wrapper's 5px rounded corner while zoomed (it shows as a stray frame)
    el.style.borderRadius = "0";
    el.style.transformOrigin = "0 0";
    el.style.transform = "translate(" + z.x + "px," + z.y + "px) scale(" + z.s + ")";
    el.style.willChange = "transform";
  }
  // Fully remove every inline style we added — used on zoom-out-to-1 and reset, so
  // no leftover transform/overflow lingers (that left a faint frame at rest).
  function clear(el) {
    if (!el) return;
    const z = st(el); z.s = 1; z.x = 0; z.y = 0;
    el.style.transform = "";
    el.style.transformOrigin = "";
    el.style.willChange = "";
    el.style.borderRadius = "";
    el.style.cursor = "";
    if (el.parentElement) el.parentElement.style.overflow = "";
  }

  document.addEventListener("wheel", function (e) {
    const p = panelOf(e.target);
    if (!p) return;
    const el = elOf(p);
    if (!el || !el.parentElement) return;
    e.preventDefault();
    e.stopPropagation();
    const z = st(el);
    const pr = el.parentElement.getBoundingClientRect();
    const cx = e.clientX - pr.left, cy = e.clientY - pr.top;
    const ix = (cx - z.x) / z.s, iy = (cy - z.y) / z.s;
    const ns = Math.max(1, Math.min(MAX, z.s * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
    if (ns <= 1.001) { clear(el); return; }   // back to 1x → strip every leftover style
    z.s = ns; z.x = cx - ix * ns; z.y = cy - iy * ns;
    draw(el, z);
    if (p.pan) el.style.cursor = "grab";
  }, { passive: false, capture: true });

  // drag-pan (Detected / Result only — the editor's mousedown is for painting)
  let drag = null;
  document.addEventListener("mousedown", function (e) {
    const p = panelOf(e.target);
    if (!p || !p.pan) return;
    const el = elOf(p);
    if (!el) return;
    const z = st(el);
    if (z.s <= 1) return;
    drag = { el: el, mx: e.clientX, my: e.clientY, x: z.x, y: z.y };
    el.style.cursor = "grabbing";
    e.preventDefault();
  }, true);
  document.addEventListener("mousemove", function (e) {
    if (!drag) return;
    const z = st(drag.el);
    z.x = drag.x + (e.clientX - drag.mx);
    z.y = drag.y + (e.clientY - drag.my);
    draw(drag.el, z);
  });
  document.addEventListener("mouseup", function () {
    if (!drag) return;
    drag.el.style.cursor = st(drag.el).s > 1 ? "grab" : "";
    drag = null;
  });

  // Inject a per-panel ⛶ fit / center button (top-left corner of each panel).
  function inject() {
    for (let i = 0; i < PANELS.length; i++) {
      const p = PANELS[i];
      const host = document.querySelector("#" + p.host);
      if (!host || host.querySelector(".ac-fit")) continue;
      const b = document.createElement("button");
      b.className = "ac-fit";
      b.type = "button";
      b.textContent = "⛶";
      b.title = "Fit / center this image (reset zoom)";
      b.style.cssText =
        "position:absolute;bottom:6px;right:6px;z-index:40;width:24px;height:24px;padding:0;" +
        "border:none;border-radius:5px;background:rgba(0,0,0,.55);color:#fff;font-size:15px;" +
        "line-height:24px;text-align:center;cursor:pointer;";
      b.addEventListener("click", function (e) {
        e.preventDefault();
        e.stopPropagation();
        clear(elOf(p));
      });
      if (getComputedStyle(host).position === "static") host.style.position = "relative";
      host.appendChild(b);
    }
  }
  if (typeof onUiLoaded === "function") onUiLoaded(inject);
  if (typeof onUiUpdate === "function") onUiUpdate(inject);
})();
