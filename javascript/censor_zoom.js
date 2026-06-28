// Mouse-wheel zoom-to-cursor + drag-pan for the Auto-Censor display images
// (the "Detected" and "Result" panels). The input ImageEditor is excluded — it
// has its own zoom in its toolbar and mousedown there means painting.
//
//   wheel        : zoom in/out centred on the cursor
//   drag (zoomed): pan
//   double-click : reset
(function () {
  "use strict";
  const TAB = "auto_censor_tab";
  const EDITOR = "auto_censor_input";   // exclude the paint editor
  const MAX = 8;

  function targetImg(node) {
    if (!node || !node.closest) return null;
    if (!node.closest("#" + TAB)) return null;
    if (node.closest("#" + EDITOR)) return null;
    return node.closest("img");
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
  }, { passive: false });

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
})();
