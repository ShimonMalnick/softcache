/* SoftCache project page: theme, race, gallery wipes, schedule animation, compare viewer, tables. */
(() => {
  "use strict";
  const D = window.PAGE_DATA;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const LABEL = D.labels;
  const fmt = (x, d) => x == null ? "—" : x.toFixed(d);

  function el(tag, attrs = {}, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === false || v == null) continue;
      if (k === "class") n.className = v; else if (k === "html") n.innerHTML = v;
      else n.setAttribute(k, v === true ? "" : v);
    }
    n.append(...kids);
    return n;
  }

  /* ---------------------------------------------------------------- theme */
  const root = document.documentElement;
  try { const t = localStorage.getItem("softcache-theme"); if (t) root.dataset.theme = t; } catch (e) {}
  $(".theme")?.addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : !matchMedia("(prefers-color-scheme: light)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("softcache-theme", root.dataset.theme); } catch (e) {}
  });

  /* ---------------------------------------------------------------- nav highlight */
  const navLinks = $$(".top nav a");
  const io = new IntersectionObserver(es => {
    for (const e of es) if (e.isIntersecting) navLinks.forEach(a => a.setAttribute("aria-current", a.hash === "#" + e.target.id));
  }, { rootMargin: "-40% 0px -55% 0px" });
  $$("section[id]").forEach(s => io.observe(s));

  /* ---------------------------------------------------------------- clips: blob URLs (the artifact
     server ignores Range requests, so a streamed mp4 cannot loop or seek) */
  const blobs = new Map(), progress = new Map();   // progress: src -> [bytes read, total], for loading bars
  async function fetchBlob(src) {
    const r = await fetch(src), p = [0, +r.headers.get("content-length") || 0], reader = r.body.getReader(), parts = [];
    progress.set(src, p);
    for (let c; !(c = await reader.read()).done;) { parts.push(c.value); p[0] += c.value.length; }
    return new Blob(parts, { type: r.headers.get("content-type") || "video/mp4" });
  }
  async function clip(src) {
    if (!blobs.has(src)) blobs.set(src, fetchBlob(src).then(b => URL.createObjectURL(b)).catch(() => src));
    return blobs.get(src);
  }
  async function loadVideo(v) {
    if (v.dataset.loaded) return v;
    v.src = await clip(v.dataset.src); v.dataset.loaded = "1";
    await new Promise(res => { v.addEventListener("loadeddata", res, { once: true }); v.load(); });
    return v;
  }

  /* ---------------------------------------------------------------- lockstep: clips advance frame by frame
     together by seeking (trimmed from the supp site's FrameClock), so side-by-side clips never drift */
  class Lockstep {
    constructor(videos, frames = 65, fps = 8) { Object.assign(this, { videos, frames, fps, frame: 0, playing: false }); videos.forEach(v => v.muted = true); }
    load() {   // the clip's own length sets the loop (Wan: 81 frames at 16 fps)
      return Promise.all(this.videos.map(loadVideo)).then(vs => {
        const d = vs[0]?.duration; if (isFinite(d)) this.frames = Math.round(d * this.fps);
        return vs;
      });
    }
    seek(f) {
      this.frame = ((f % this.frames) + this.frames) % this.frames;
      for (const v of this.videos) {
        if (v.readyState < 1) continue;
        const t = (this.frame + 0.5) / this.fps;
        v.currentTime = Math.min(t, (v.duration || t + 1) - 0.001);
      }
    }
    ready() { return this.videos.every(v => v.readyState >= 2 && !v.seeking); }
    play() {
      if (this.playing) return;
      this.playing = true;
      let last = performance.now(), acc = 0;
      const loop = now => {
        if (!this.playing) return;
        acc += now - last; last = now;
        const dt = 1000 / this.fps;
        if (acc >= dt && this.ready()) { acc = Math.min(acc - dt, dt); this.seek(this.frame + 1); }
        this.raf = requestAnimationFrame(loop);
      };
      this.raf = requestAnimationFrame(loop);
    }
    pause() { this.playing = false; cancelAnimationFrame(this.raf); }
  }
  // play() once half of node is on screen (or it covers half the screen, if taller), reset() once it has
  // fully left, so every visit replays. rootBounds, not innerHeight: iOS's toolbar changes the latter
  function eachEntry(node, play, reset) {
    let shown = false;
    new IntersectionObserver(([e]) => {
      const vh = e.rootBounds?.height || innerHeight;
      if (!shown && e.intersectionRect.height >= 0.5 * Math.min(e.boundingClientRect.height, vh)) { shown = true; play(); }
      else if (shown && !e.isIntersecting) { shown = false; reset(); }
    }, { threshold: Array.from({ length: 21 }, (_, i) => i / 20) }).observe(node);
  }
  // play a Lockstep while its element is on screen
  function playInView(node, lock, onFirst) {
    let first = true;
    new IntersectionObserver(async ([e]) => {
      if (!e.isIntersecting) return lock.pause();
      await lock.load();
      if (first) { first = false; lock.seek(0); onFirst?.(); }
      if (!REDUCED) lock.play();
    }, { threshold: 0.4 }).observe(node);
  }

  /* ---------------------------------------------------------------- hero: reference vs SoftCache "generating".
     Illustrative: blur + grain decay to sharp, SoftCache in 1.15 s of screen time, the reference in
     speedup x 1.15 s (the clip's own GPU seconds at one rate); a stopwatch's hand spins while its lane generates. */
  function denoise() {
    const H = D.hero; const stage = $("#hero-stage"); if (!H || !stage || H.strip) return;
    const tiles = $$(".gen", stage).map(f => {
      const c = $("canvas", f); c.width = 240; c.height = 180;
      const ctx = c.getContext("2d");
      return { f, c, ctx, img: ctx.createImageData(c.width, c.height), v: $("video", f), w: $(".watch", f), secs: H[f.dataset.lane].seconds };
    });
    const rate = H.ours.seconds / 1.15;   // GPU seconds per screen second (SoftCache in 1.15 s; lanes keep the measured ratio)
    const lock = new Lockstep(tiles.map(t => t.v));
    let raf = 0, start = 0;
    function grain(t) {
      const d = t.img.data;
      for (let i = 0; i < d.length; i += 4) { const g = Math.random() * 255; d[i] = d[i + 1] = d[i + 2] = g; d[i + 3] = 255; }
      t.ctx.putImageData(t.img, 0, 0);
    }
    function paint(t, p) {
      const q = 1 - p;
      t.v.style.filter = q > 0 ? `blur(${(20 * q * q).toFixed(1)}px) saturate(${(0.4 + 0.6 * p).toFixed(2)})` : "";
      t.v.style.transform = q > 0 ? `scale(${(1 + 0.04 * q).toFixed(4)})` : "";   // hides the blur's soft edges, easing to 1 so the clean frame doesn't jump
      t.c.style.opacity = Math.sqrt(q).toFixed(3);   // stays noisy most of the run, so the slow lane reads as slow
      t.w.classList.toggle("running", p < 1);
      t.f.classList.toggle("done", p >= 1);
    }
    function frame(now) {
      if (!start) start = now;
      const gpu = (now - start) / 1000 * rate; let pending = 0;
      for (const t of tiles) { const p = Math.min(1, gpu / t.secs); if (p < 1) { pending++; grain(t); } paint(t, p); }
      if (pending) raf = requestAnimationFrame(frame);
    }
    function run() {
      cancelAnimationFrame(raf); start = 0;
      if (REDUCED) return tiles.forEach(t => paint(t, 1));
      tiles.forEach(t => paint(t, 0));
      lock.seek(0); raf = requestAnimationFrame(frame);
    }
    tiles.forEach(t => paint(t, REDUCED ? 1 : 0));
    $("#hero-replay").addEventListener("click", () => { lock.load().then(run); lock.play(); });
    playInView(stage, lock, run);
  }

  /* ---------------------------------------------------------------- lightbox: opens large over a dimmed,
     blurred page; fill(stage) builds the content, whose clips start in lockstep at time t (seconds) */
  const LB = (() => {
    const dlg = $("#lightbox"); if (!dlg) return null;
    const stage = $(".lb-stage", dlg); let lock = null;
    $(".lb-close", dlg).addEventListener("click", () => dlg.close());
    dlg.addEventListener("click", e => { if (e.target === dlg) dlg.close(); });   // click on the dimmed backdrop
    dlg.addEventListener("close", () => { lock?.pause(); lock = null; stage.innerHTML = ""; dlg.classList.remove("lanes"); });
    return {
      open(fill, prompt, t) {
        stage.removeAttribute("style"); stage.className = "lb-stage"; fill(stage);
        $(".lb-prompt", dlg).textContent = `“${prompt}”`;
        dlg.showModal();
        const mine = lock = new Lockstep($$("video", stage));
        mine.load().then(() => { if (lock !== mine) return; mine.seek(Math.round(t * mine.fps)); if (!REDUCED) mine.play(); });
      },
    };
  })();
  // a clip (a) wiped against its reference, in the lightbox
  function openWipe(a, ref, prompt, t, w = 640, h = 480) {
    LB?.open(s => { s.classList.add("cmp-stage"); wipe(s, { kind: "video", w, h, title: prompt }, a, ref, 50, () => {}); }, prompt, t);
  }
  // make node open the lightbox on click, Enter or Space; target(e) picks the element (or null)
  function clickable(node, target, open) {
    node.addEventListener("click", e => { const f = target(e); if (f) open(f); });
    node.addEventListener("keydown", e => { const f = target(e); if (f && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); open(f); } });
  }

  /* ---------------------------------------------------------------- the four lanes large, one shared zoom/pan.
     Ported from the supplementary viewer: V.zoom (1 = fit) and (cx, cy), the image point at each panel's centre */
  function openLanes(it, cells, t) {
    const V = { zoom: 1, cx: 0.5, cy: 0.5 }, GAP = 8, ptrs = new Map(), TOUCH = matchMedia("(pointer: coarse)").matches;
    let tap = null, lastToggle = 0;   // the last touch tap {t, x, y}: phones send no dblclick, so a double tap is detected here
    let stage, panels = [], g = null;
    const fit = () => panels[0].vp.clientWidth / it.w;
    const maxZoom = () => Math.max(4, 4 / fit());
    function apply() {
      const f = fit(); if (!f) return;
      V.zoom = Math.min(maxZoom(), Math.max(1, V.zoom));
      const h = 0.5 / V.zoom;   // half the visible share of the image (panels have the image's aspect)
      V.cx = Math.min(1 - h, Math.max(h, V.cx)); V.cy = Math.min(1 - h, Math.max(h, V.cy));
      const s = f * V.zoom, tx = (it.w * f) / 2 - V.cx * it.w * s, ty = (it.h * f) / 2 - V.cy * it.h * s;
      for (const p of panels) p.m.style.transform = `translate(${tx}px, ${ty}px) scale(${s})`;
    }
    function zoomAt(mx, my, zoom) {   // keep the image point under (mx, my) fixed
      const f = fit(), s = f * V.zoom, pw = it.w * f, ph = it.h * f;
      const u = (mx - (pw / 2 - V.cx * it.w * s)) / (it.w * s), v = (my - (ph / 2 - V.cy * it.h * s)) / (it.h * s);
      V.zoom = Math.min(maxZoom(), Math.max(1, zoom));
      const s2 = f * V.zoom;
      V.cx = (pw / 2 - (mx - u * it.w * s2)) / (it.w * s2);
      V.cy = (ph / 2 - (my - v * it.h * s2)) / (it.h * s2);
      apply();
    }
    const local = (vp, x, y) => { const r = vp.getBoundingClientRect(); return [x - r.left, y - r.top]; };
    function wire(vp) {
      vp.addEventListener("wheel", e => {
        e.preventDefault();
        const k = e.deltaMode === 1 ? 0.04 : e.deltaMode === 2 ? 0.8 : 0.0015, dy = Math.max(-120, Math.min(120, e.deltaY));
        zoomAt(...local(vp, e.clientX, e.clientY), V.zoom * Math.exp(-dy * k));
      }, { passive: false });
      vp.addEventListener("pointerdown", e => {
        if (e.button > 0) return;
        vp.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY });
        const [a, b] = [...ptrs.values()];
        g = ptrs.size === 2 ? { pinch: Math.hypot(a.x - b.x, a.y - b.y), zoom: V.zoom } : {};
        vp.classList.add("dragging");
      });
      vp.addEventListener("pointermove", e => {
        const prev = ptrs.get(e.pointerId); if (!prev || !g) return;
        ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
        if (g.pinch && ptrs.size === 2) {
          const [a, b] = [...ptrs.values()];
          zoomAt(...local(vp, (a.x + b.x) / 2, (a.y + b.y) / 2), g.zoom * Math.hypot(a.x - b.x, a.y - b.y) / g.pinch);
        } else {
          const s = fit() * V.zoom;
          V.cx -= (e.clientX - prev.x) / (it.w * s); V.cy -= (e.clientY - prev.y) / (it.h * s); apply();
        }
      });
      const toggle = (x, y) => { lastToggle = performance.now(); if (V.zoom < 1.5) zoomAt(...local(vp, x, y), 4); else { V.zoom = 1; apply(); } };
      const up = e => {
        const d = ptrs.get(e.pointerId);
        if (d && e.pointerType !== "mouse" && ptrs.size === 1 && Math.hypot(e.clientX - d.x0, e.clientY - d.y0) < 10) {
          const now = performance.now();
          if (tap && now - tap.t < 350 && Math.hypot(e.clientX - tap.x, e.clientY - tap.y) < 30) { toggle(e.clientX, e.clientY); tap = null; }
          else tap = { t: now, x: e.clientX, y: e.clientY };
        }
        ptrs.delete(e.pointerId); if (!ptrs.size) { vp.classList.remove("dragging"); g = null; } else if (g?.pinch) g = {};
      };
      vp.addEventListener("pointerup", up); vp.addEventListener("pointercancel", up);
      // touch double taps are handled in up(); some phones also synthesize a dblclick, which would undo them
      vp.addEventListener("dblclick", e => { if (!tap && performance.now() - lastToggle > 500) toggle(e.clientX, e.clientY); });
    }
    // one row of lanes; 2x2 on phones
    function layout() {
      const n = panels.length, W = innerWidth * 0.96, H = innerHeight * 0.86 - 70;
      const cols = innerWidth > 700 ? n : 2, rows = Math.ceil(n / cols);
      const w = Math.min((W - (cols - 1) * GAP) / cols, ((H - (rows - 1) * GAP) / rows) * it.w / it.h);
      stage.style.gridTemplateColumns = `repeat(${cols}, ${Math.floor(w)}px)`;
      apply();
    }
    LB?.open(s => {
      stage = s; s.classList.add("zl-stage"); s.closest("dialog").classList.add("lanes");
      s.style.setProperty("--ar", `${it.w}/${it.h}`);
      panels = cells.map(c => {
        const m = it.kind === "video" ? el("video", { muted: true, playsinline: true, preload: "none", "data-src": c.src, poster: c.poster })
                                      : el("img", { src: c.src, alt: `${c.label}: ${it.title}`, draggable: "false" });
        m.style.width = it.w + "px"; m.style.height = it.h + "px";
        const vp = el("figure", { class: "zl-vp" }, m,
          el("figcaption", {}, c.m === "reference" ? c.label : `${c.label} ${c.speed.toFixed(2)}×`));
        wire(vp); return { vp, m };
      });
      s.append(...panels.map(p => p.vp), el("p", { class: "zl-hint" }, TOUCH ? "Pinch to zoom · drag to pan · double-tap for 4× · all four move together"
                                              : "Scroll to zoom · drag to pan · double-click for 4× · all four move together"));
    }, it.title, t);
    layout();
    // re-layout on width changes only: a phone's address bar hiding on scroll changes just the height
    let lastW = innerWidth;
    const onResize = () => { if (innerWidth !== lastW) { lastW = innerWidth; layout(); } };
    addEventListener("resize", onResize);
    $("#lightbox").addEventListener("close", () => removeEventListener("resize", onResize), { once: true });
  }

  /* ---------------------------------------------------------------- hero click: SoftCache wiped against the reference */
  function heroWipe() {
    const H = D.hero, stage = $("#hero-stage"); if (!H || !stage || H.strip) return;
    clickable(stage, e => e.target.closest(".gen"), () =>
      openWipe({ ...H.ours, speed: H.speedup }, H.reference, H.prompt, $("video", stage).currentTime || 0, H.w, H.h));
  }

  /* ---------------------------------------------------------------- gallery */
  function gallery() {
    const wall = $("#wall"); if (!wall) return;
    const io = new IntersectionObserver(es => es.forEach(async e => {
      const v = e.target;
      if (e.isIntersecting) { await loadVideo(v); sync(v); v.play().catch(() => {}); } else v.pause();
    }), { threshold: 0.1 });
    // every baked wipe is the same length, so pin them all to one page clock: a tile that starts late joins in phase
    const sync = v => { const t = (performance.now() / 1000) % v.duration; if (Math.abs(v.currentTime - t) > 0.05) v.currentTime = t; };
    // posters wait until the wall is a screen away: at page open they competed with the hero on slow links
    new IntersectionObserver((es, o) => {
      if (!es.some(e => e.isIntersecting)) return;
      $$("video", wall).forEach(v => v.poster = v.dataset.poster); o.disconnect();
    }, { rootMargin: "100% 0px" }).observe(wall);
    $$("video", wall).forEach(v => {
      v.muted = true; io.observe(v);
      let last = 0;   // re-sync once per loop, when the clip wraps back to 0
      v.addEventListener("timeupdate", () => { if (v.currentTime < last) sync(v); last = v.currentTime; });
    });
    lightbox(wall);
    // heading icon: the tiles' seam (0.5 + 0.4 sin over one 8.125 s clip, on the same performance.now() clock as sync())
    const seam = $(".compare .seam");
    if (seam && !REDUCED) (function tick() {
      seam.style.transform = `translateX(${0.4 * 56 * Math.sin(2 * Math.PI * ((performance.now() / 1000) % 8.125) / 8.125)}px)`;   // 56 = frame width in icon units
      requestAnimationFrame(tick);
    })();
    if (!wall.classList.contains("mosaic") || REDUCED) return;

    // mosaic, in stages: one clip fills the frame, then a 2x2 block, then 3x3, then the whole wall.
    // Each stage scales a centred k x k block of tiles to the frame width.
    const W = wall.offsetWidth, H = wall.offsetHeight, tiles = $$("figure", wall);
    const cols = getComputedStyle(wall).gridTemplateColumns.split(" ").length, rows = Math.ceil(tiles.length / cols);
    function fit(k) {
      const c0 = Math.floor((cols - k) / 2), r0 = Math.floor((rows - k) / 2);
      const a = tiles[r0 * cols + c0], b = tiles[(r0 + k - 1) * cols + c0 + k - 1];
      const x0 = a.offsetLeft, y0 = a.offsetTop, w = b.offsetLeft + b.offsetWidth - x0, h = b.offsetTop + b.offsetHeight - y0;
      const s = W / w;
      return `translate(${W / 2 - s * (x0 + w / 2)}px, ${H / 2 - s * (y0 + h / 2)}px) scale(${s})`;
    }
    const stages = [1, 2, 3].filter(k => k < cols && k <= rows).map(fit).concat("none");
    wall.style.transformOrigin = "0 0";
    wall.style.transform = stages[0];
    let timers = [];
    eachEntry(wall.parentElement, () => {
      wall.classList.add("settle");
      timers = stages.slice(1).map((tf, i) => setTimeout(() => wall.style.transform = tf, 1400 * (i + 1)));   // 1.4 s per stage
    }, () => {   // off screen: snap back to one clip, without the transition
      timers.forEach(clearTimeout);
      wall.classList.remove("settle"); wall.style.transform = stages[0];
    });
  }

  // click a wall tile: the clip opens large, wiped against its reference, in lockstep
  function lightbox(wall) {
    const G = D.gallery;
    clickable(wall, e => e.target.closest("figure"), fig => {
      const c = G.clips[+fig.dataset.i];
      openWipe({ ...c, speed: G.speed }, c.ref, c.prompt, $("video", fig).currentTime);
    });
  }

  /* ---------------------------------------------------------------- schedule animation */
  function heat() {
    const H = D.heat; const host = $("#heat-svg"); if (!H || !host) return;
    const N = H.steps, LAB = 78, W = 900, CW = (W - LAB) / N;
    let raf = 0, tiles = [], t0 = 0;
    const SWEEP = { one: 4200, many: 650 };   // ms for the cursor to cross the grid; 25 prompts replay faster
    // default: one prompt, tall rows; the toggle swaps in every prompt as thin rows
    let many = false;
    const GROUPS = {
      one: [{ name: LABEL.seacache, rows: [H.seacache[0]], h: 30 }, { name: LABEL.ours, rows: [H.ours[0]], h: 30 }],
      many: [{ name: LABEL.seacache, rows: H.seacache, h: 5 }, { name: LABEL.ours, rows: H.ours, h: 5 }],
    };

    function build() {
      const groups = GROUPS[many ? "many" : "one"]; const svg = [];
      let y = 6; tiles = [];
      for (const g of groups) {
        svg.push(`<text x="${LAB - 10}" y="${y + (g.rows.length * (g.h + 2)) / 2 + 4}" text-anchor="end" class="rowlab">${g.name}</text>`);
        for (const row of g.rows) {
          for (let i = 0; i < row.length; i++) {
            const [pos, fresh] = row[i]; const end = i + 1 < row.length ? row[i + 1][0] : N;
            const x = LAB + pos * CW, w = Math.max(CW * 0.9, (end - pos) * CW - CW * 0.1);
            svg.push(`<rect class="${fresh ? "f" : "c"}" x="${x.toFixed(1)}" y="${y}" width="${w.toFixed(1)}" height="${g.h}" data-start="${pos}"/>`);
          }
          y += g.h + 2;
        }
        y += 10;
      }
      svg.push(`<line class="cursor" x1="${LAB}" x2="${LAB}" y1="0" y2="${y}"/>`);
      for (const s of [0, 10, 20, 30, 40, 50]) svg.push(`<text x="${LAB + s * CW}" y="${y + 14}" text-anchor="middle">${s}</text>`);
      svg.push(`<text x="${LAB + N * CW / 2}" y="${y + 30}" text-anchor="middle">scheduled step</text>`);
      host.setAttribute("viewBox", `0 0 ${W + 12} ${y + 36}`); host.innerHTML = svg.join("");
      tiles = $$("rect", host);
      if (armed) play(); else $(".cursor", host).style.opacity = 0;   // empty until it scrolls to mid-viewport
    }
    function play() {
      cancelAnimationFrame(raf); tiles.forEach(t => t.classList.remove("on"));
      const cur = $(".cursor", host);
      if (REDUCED) { tiles.forEach(t => t.classList.add("on")); cur?.remove(); return; }
      t0 = performance.now();
      const step = now => {
        const p = Math.min(1, (now - t0) / SWEEP[many ? "many" : "one"]), x = LAB + p * N * CW;
        cur.setAttribute("x1", x); cur.setAttribute("x2", x);
        for (const t of tiles) if (!t.classList.contains("on") && +t.dataset.start <= p * N) t.classList.add("on");
        if (p < 1) raf = requestAnimationFrame(step); else cur.style.opacity = 0;
      };
      cur.style.opacity = ""; raf = requestAnimationFrame(step);
    }
    let armed = false;
    $("#heat-replay").addEventListener("click", () => { armed = true; play(); });
    for (const b of $$("#heat-modes button")) b.addEventListener("click", () => {
      many = b.dataset.many === "1"; armed = true;
      $$("#heat-modes button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
      build();
    });
    build();
    eachEntry(host.closest(".heat"), () => { if (!armed) { armed = true; play(); } }, () => {
      armed = false; cancelAnimationFrame(raf); tiles.forEach(t => t.classList.remove("on"));
      const cur = $(".cursor", host); if (cur) cur.style.opacity = 0;
    });
  }

  /* ---------------------------------------------------------------- teaser walk: band, grid, then the
     sampler's steps one by one (each landing node takes its fresh/cached colour), callouts last */
  function teaser() {
    const svg = $("#teaser"); if (!svg) return;
    const band = $(".tz-band", svg), sched = $$(".tz-sched", svg).filter(c => !c.closest(".tz-call"));
    const steps = $$(".tz-step", svg), nodes = $$(".tz-node", svg).filter(c => !c.closest(".tz-call")), call = $(".tz-call", svg);
    const len = band.getTotalLength();
    let timers = [];
    const at = (ms, f) => timers.push(setTimeout(f, ms));
    function reset() {
      timers.forEach(clearTimeout); timers = [];
      svg.classList.add("run");
      band.style.strokeDasharray = band.style.strokeDashoffset = len;
      for (const l of steps) { const n = Math.hypot(l.x2.baseVal.value - l.x1.baseVal.value, l.y2.baseVal.value - l.y1.baseVal.value); l.style.strokeDasharray = l.style.strokeDashoffset = n; l.classList.remove("on"); }
      [...sched, ...nodes, call].forEach(e => e.classList.remove("on"));
      svg.getBoundingClientRect();   // flush so the transitions restart
    }
    function run() {
      reset();
      if (REDUCED) { svg.classList.remove("run"); band.style.strokeDashoffset = 0; steps.forEach(l => { l.style.strokeDashoffset = 0; l.classList.add("on"); }); [...sched, ...nodes, call].forEach(e => e.classList.add("on")); return; }
      at(0, () => band.style.strokeDashoffset = 0);                         // 900 ms draw
      sched.forEach((c, i) => at(700 + i * 55, () => c.classList.add("on")));
      let t = 700 + sched.length * 55 + 200;
      at(t, () => nodes[0].classList.add("on"));
      steps.forEach((l, i) => {
        t += 250; at(t, () => { l.style.strokeDashoffset = 0; });           // 380 ms draw
        at(t + 380, () => { l.classList.add("on"); nodes[i + 1].classList.add("on"); });
        t += 480;
      });
      at(t + 250, () => call.classList.add("on"));
    }
    $("#teaser-replay").addEventListener("click", run);
    reset();
    eachEntry(svg, run, reset);
  }

  /* ---------------------------------------------------------------- compare viewer. The wipe is the supp
     site's hero-wipe: an invisible full-cover range input drives --split, so a drag anywhere moves the
     seam and the media (draggable=false) never gets picked up. */
  // fill a .cmp-stage with a (right) wiped against ref (left); media are draggable=false and an invisible
  // full-cover range input drives --split, so a drag anywhere moves the seam
  function wipe(stage, it, a, ref, split, onSplit) {
    stage.innerHTML = "";
    stage.style.setProperty("--ar", `${it.w}/${it.h}`); stage.style.setProperty("--split", split + "%");
    const mk = x => it.kind === "video"
      ? el("video", { muted: true, playsinline: true, preload: "none", "data-src": x.src, poster: x.poster, draggable: "false" })
      : el("img", { src: x.src, alt: it.title, draggable: "false" });
    const range = el("input", { type: "range", class: "cmp-range", min: 0, max: 100, step: 0.5, value: split, "aria-label": `Wipe between the reference and ${a.label}` });
    range.addEventListener("input", () => { onSplit(+range.value); stage.style.setProperty("--split", range.value + "%"); });
    stage.append(mk(a), el("div", { class: "ref" }, mk(ref)), el("div", { class: "handle" }),
      el("span", { class: "lbl l" }, LABEL.reference), el("span", { class: "lbl r" }, `${a.label} ${a.speed.toFixed(2)}×`), range);
  }

  function bench(block) {
    const B = D.bench[block.dataset.model]; if (!B) return;
    const stage = $(".cmp-lanes", block), thumbs = $(".thumbs", block);
    let id = null, lock = null, inView = false;
    const LANES = ["reference", "seacache", "dpcache", "ours"];
    const METRICS = [["psnr", "PSNR", 2], ["ssim", "SSIM", 3], ["lpips", "LPIPS", 3], ["dreamsim", "DreamSim", 3]];

    function pickBand(band) {
      $$(".bands button", block).forEach(b => b.setAttribute("aria-pressed", String(b.dataset.band === band)));
      $$("tr[data-band]", block).forEach(tr => tr.classList.toggle("dim", tr.dataset.band !== band));
      const ids = Object.keys(B.items).filter(k => String(B.items[k].band) === band);
      thumbs.replaceChildren(...ids.map(k => {
        const it = B.items[k], r = it.cells.find(c => c.m === "reference");
        return el("button", { "aria-pressed": "false", title: it.title, "data-id": k },
          el("img", { loading: "lazy", src: r.thumb || r.poster || r.src, alt: it.title, draggable: "false" }));   // loading before src
      }));
      show(ids[0]);
    }
    function show(k) {
      id = k; $$("button", thumbs).forEach(x => x.setAttribute("aria-pressed", String(x.dataset.id === k)));
      render();
    }
    $(".bands", block).addEventListener("click", e => { const b = e.target.closest("button"); if (b) pickBand(b.dataset.band); });
    thumbs.addEventListener("click", e => { const b = e.target.closest("button"); if (b) show(b.dataset.id); });

    function render() {
      const it = B.items[id], frame = lock?.frame || 0;
      lock?.pause(); lock = null;
      // all four methods side by side, each with its own speed and metrics (vs the reference)
      stage.style.setProperty("--ar", `${it.w}/${it.h}`);
      stage.replaceChildren(...LANES.map(m => it.cells.find(c => c.m === m)).filter(Boolean).map(c => {
        const x = c.metrics || {};
        return el("figure", { tabindex: "0", title: "Open large" },
          it.kind === "video" ? el("video", { muted: true, playsinline: true, preload: "none", "data-src": c.lane || c.src, poster: c.poster })
                              : el("img", { loading: "lazy", src: c.src, alt: `${c.label}: ${it.title}` }),
          el("figcaption", { class: c.m === "ours" ? "ours" : null }, c.m === "reference" ? c.label : `${c.label} ${c.speed.toFixed(2)}×`),
          el("p", { class: "lane-metrics" }, METRICS.filter(([k]) => x[k] != null).map(([k, n, d]) => `${n} ${fmt(x[k], d)}`).join(" · ")));
      }));
      $(".cmp-title", block).textContent = `“${it.title}”`;
      if (it.kind === "video") { lock = new Lockstep($$("video", stage)); lock.frame = frame; if (inView) start(); }
    }
    // fetch the shown clips once (a screen ahead, or on screen); each lane's bar fills with its own clip's bytes,
    // since the lanes only start once all four have arrived
    function fetchLanes(mine) {
      if (mine.fetching) return mine.fetching;
      const bars = mine.videos.map(v => { const b = el("span", { class: "load-bar", "aria-hidden": "true" }); v.after(b); return b; });
      const tick = () => {
        mine.videos.forEach((v, i) => { const p = progress.get(v.dataset.src); if (p?.[1]) bars[i].style.setProperty("--p", p[0] / p[1]); });
        if (!mine.done) requestAnimationFrame(tick);
      };
      tick();
      return mine.fetching = mine.load().then(vs => { mine.done = true; bars.forEach(b => b.remove()); return vs; });
    }
    function start() {
      const mine = lock;
      fetchLanes(mine).then(() => {
        if (lock !== mine) return;
        if (!mine.started) { mine.started = true; mine.seek(mine.frame); }
        if (inView && !REDUCED) mine.play();
      });
    }
    // click a lane: all four open large with a shared zoom; the page lanes wait meanwhile
    clickable(stage, e => e.target.closest("figure"), () => {
      const it = B.items[id]; lock?.pause();
      openLanes(it, LANES.map(m => it.cells.find(c => c.m === m)).filter(Boolean), lock ? lock.frame / lock.fps : 0);
    });
    $("#lightbox")?.addEventListener("close", () => { if (lock && inView) start(); });
    new IntersectionObserver(([e]) => { inView = e.isIntersecting; if (!lock) return; if (inView) start(); else lock.pause(); }, { threshold: 0.3 }).observe(stage);
    new IntersectionObserver(([e]) => { if (e.isIntersecting && lock) fetchLanes(lock); }, { rootMargin: "100% 0px" }).observe(stage);
    pickBand($(".bands button[aria-pressed=true]", block).dataset.band);   // the middle band, set by the build
  }

  /* ---------------------------------------------------------------- bibtex */
  $("#copy-bib")?.addEventListener("click", async e => {
    try { await navigator.clipboard.writeText($("#bib").textContent); e.target.textContent = "Copied"; setTimeout(() => e.target.textContent = "Copy BibTeX", 1500); } catch (x) {}
  });

  /* hero strip: four lanes of one clip, loaded when near the screen, played together while visible */
  function heroStrip(H, stage) {
    if (!H?.strip || !stage) return;
    const vs = $$("video", stage);
    new IntersectionObserver(([e]) => {
      if (!e.isIntersecting) return vs.forEach(v => v.pause());
      vs.forEach(v => { if (!v.src) v.src = v.dataset.src; });
      vs.forEach(v => v.play().catch(() => {}));
    }, { threshold: 0.3 }).observe(stage);
    // click a lane: all four open large with a shared zoom
    clickable(stage, e => e.target.closest("figure"), () => {
      const t = vs[0].currentTime; vs.forEach(v => v.pause());
      openLanes({ kind: "video", w: H.w, h: H.h, title: H.prompt }, ["reference", "seacache", "dpcache", "ours"].map(m => H[m]), t);
    });
    $("#lightbox")?.addEventListener("close", () => { if (stage.getBoundingClientRect().top < innerHeight) vs.forEach(v => v.play().catch(() => {})); });
    setInterval(() => vs.slice(1).forEach(v => { if (Math.abs(v.currentTime - vs[0].currentTime) > 0.08) v.currentTime = vs[0].currentTime; }), 200);
  }

  denoise(); heroWipe(); heroStrip(D.hero, $("#hero-stage")); (D.rivals || []).forEach((r, i) => heroStrip(r, $(`#rival-stage-${i}`))); gallery(); teaser(); heat(); $$(".bench").forEach(bench);
})();
