// Mouse-wheel zoom-to-cursor + drag-pan for the Auto-Censor display images
// (the "Detected" and "Result" panels). The input ImageEditor is excluded — it
// has its own zoom in its toolbar and mousedown there means painting.
//
//   wheel        : zoom in/out centred on the cursor
//   drag (zoomed): pan
//   double-click : reset
(function () {
  "use strict";
  const MAX = 8;

  // Scope to the two display panels by their stable elem_ids (the tab-id selector
  // was unreliable). Find the <img> inside whichever panel the cursor is over.
  function targetImg(node) {
    if (!node || !node.closest) return null;
    const host = node.closest("#ac_preview, #ac_result");
    if (!host) return null;
    return host.querySelector("img");
  }

  function state(img) {
    if (!img._acz) img._acz = { s: 1, x: 0, y: 0 };
    return img._acz;
  }

  function draw(img, z) {
    const p = img.parentElement;
    if (p) { p.style.overflow = "hidden"; }
    img.style.transformOrigin = "0 0";
    img.style.transform = "translate(" + z.x + "px," + z.y + "px) scale(" + z.s + ")";
    img.style.cursor = z.s > 1 ? "grab" : "";
    img.style.willChange = "transform";
  }

  document.addEventListener("wheel", function (e) {
    const img = targetImg(e.target);
    if (!img || !img.parentElement) return;
    e.preventDefault();
    const z = state(img);
    const p = img.parentElement.getBoundingClientRect();
    const cx = e.clientX - p.left;
    const cy = e.clientY - p.top;
    const ix = (cx - z.x) / z.s;          // image-space point under the cursor
    const iy = (cy - z.y) / z.s;
    const ns = Math.max(1, Math.min(MAX, z.s * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
    z.s = ns;
    z.x = cx - ix * ns;                   // keep that point under the cursor
    z.y = cy - iy * ns;
    if (z.s <= 1.001) { z.s = 1; z.x = 0; z.y = 0; }
    draw(img, z);
    // capture phase: LobeTheme/Gradio swallow wheel during bubbling, so run first.
  }, { passive: false, capture: true });

  let drag = null;
  document.addEventListener("mousedown", function (e) {
    const img = targetImg(e.target);
    if (!img) return;
    const z = state(img);
    if (z.s <= 1) return;                 // only pan when zoomed in
    drag = { img: img, mx: e.clientX, my: e.clientY, x: z.x, y: z.y };
    img.style.cursor = "grabbing";
    e.preventDefault();
  });
  document.addEventListener("mousemove", function (e) {
    if (!drag) return;
    const z = state(drag.img);
    z.x = drag.x + (e.clientX - drag.mx);
    z.y = drag.y + (e.clientY - drag.my);
    draw(drag.img, z);
  });
  document.addEventListener("mouseup", function () {
    if (!drag) return;
    drag.img.style.cursor = state(drag.img).s > 1 ? "grab" : "";
    drag = null;
  });

  document.addEventListener("dblclick", function (e) {
    const img = targetImg(e.target);
    if (!img) return;
    const z = state(img);
    z.s = 1; z.x = 0; z.y = 0;
    draw(img, z);
  });

  // "Fit / center" button — reset both display panels to a centred 1x fit.
  function resetAll() {
    document.querySelectorAll("#ac_preview img, #ac_result img").forEach(function (img) {
      const z = state(img);
      z.s = 1; z.x = 0; z.y = 0;
      img.style.transform = "";
      img.style.cursor = "";
    });
  }
  // Inject a compact ⊙ "fit/center" button into the editor's top toolbar, next to Undo.
  function injectFitButton() {
    const undo = document.querySelector("#auto_censor_input button[aria-label='Undo']");
    if (!undo || !undo.parentElement) return;
    if (document.querySelector("#ac_fit_btn")) return;
    const b = document.createElement("button");
    b.id = "ac_fit_btn";
    b.className = undo.className;       // match the toolbar button style
    b.title = "Fit / center the Detected & Result images (reset zoom)";
    b.textContent = "⊙";
    b.addEventListener("click", function (e) { e.preventDefault(); resetAll(); });
    undo.parentElement.insertBefore(b, undo);
  }
  if (typeof onUiLoaded === "function") onUiLoaded(injectFitButton);
  if (typeof onUiUpdate === "function") onUiUpdate(injectFitButton);
})();
