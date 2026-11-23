// EBIM task2 run viewer: mode timeline, labelled joint plots per arm, health strip.
//
// Canvas drawing from typed arrays that are rebuilt only when the episode data changed. Every chart has two
// stacked canvases: `base` (data lines, axes; redrawn on new data, resize or a moved time window) and `over`
// (cursor, plan of the chunk at the cursor; redrawn when the cursor moves). Redraws are coalesced with
// requestAnimationFrame; an idle live poll (the 500 ms 'data' event with nothing new) costs a signature check.
//
// Exports:
//   mountPlots(root)  build the UI into root's #timeline, #arms and #health and subscribe to app.js.
//   chunkAt(step)     k of the chunk shown at a step (the step's own chunk_k, else the newest chunk received at
//                     or before it), or null.
//   plotStats         {frames, builds, lastMs, maxMs, buildMs}: drawing cost counters (for checks; not shown).

import { state, onChange, stepIndex, setCursor, setSelectedDim, episodeInfo, modeColor, MODES } from './app.js';

const SIDES_ORDER = ['right', 'left']; // right arm first
const ARM_BASE = { left: 0, right: 8 };
const GRIP = { left: 7, right: 15 };
const SIDE_NAME = { left: 'Left arm', right: 'Right arm' };
const DIMS = 16;
const DEFAULT_DT = 0.05; // 20 Hz
const BASE_BUDGET_MS = 8; // base-layer drawing per animation frame
const WINDOWS = [
  { key: 'all', label: 'whole episode', s: 0 },
  { key: '30', label: '30 s', s: 30 },
  { key: '10', label: '10 s', s: 10 },
];

// Chart colours. Series: neutral ink + the first three validated categorical slots (blue, orange, aqua pass the
// all-pairs CVD and normal-vision checks; aqua is under 3:1, so the legend and tooltips carry identity too).
const C = {
  surface: '#ffffff',
  grid: '#ecebe6',
  axis: '#c3c2b7',
  text: '#6b6a65',
  ink: '#0b0b0b',
  measured: '#3f3e3b',
  commanded: '#2a78d6',
  action: '#eb6834',
  plan: '#1baf7a',
  stale: '#9c9b95',
  clamp: '#d03b3b',
  freeze: '#b3261e',
  lane: '#b7d3f6',
  series: '#2a78d6',
};
const FONT = '11px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
const FONT_B = '600 11px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
const MINUS = '−';
// Text inside a mode fill: near-black on the light fills, white on the dark ones.
const MODE_INK = { run: '#ffffff', wait: '#0b0b0b', leash: '#0b0b0b', hold: '#0b0b0b', paused: '#ffffff' };

export const plotStats = { frames: 0, builds: 0, lastMs: 0, maxMs: 0, buildMs: 0 };

// ---------------------------------------------------------------- data (typed arrays)

let D = null; // built episode data, see build()
let dataSig = '';

function num(v) {
  return typeof v === 'number' && isFinite(v) ? v : NaN;
}

function nanArray(n) {
  const a = new Float32Array(n);
  a.fill(NaN);
  return a;
}

function config() {
  return state.runInfo && state.runInfo.config ? state.runInfo.config : null;
}

function activeArms() {
  const cfg = config();
  const arms = cfg && Array.isArray(cfg.active_arms) ? cfg.active_arms : ['left', 'right'];
  const out = SIDES_ORDER.filter(function (s) { return arms.indexOf(s) >= 0; });
  return out.length ? out : SIDES_ORDER.slice();
}

function pinnedValue(dim) {
  const cfg = config();
  if (!cfg || !cfg.pin || typeof cfg.pin !== 'object') return null;
  const v = cfg.pin[String(dim)];
  return typeof v === 'number' ? v : null;
}

function alignmentChunks() {
  const s = state.summary;
  return s && s.alignment && Array.isArray(s.alignment.chunks) ? s.alignment.chunks : [];
}

// Everything the drawings depend on apart from the cursor, the window and the canvas size.
function signature() {
  const steps = state.steps;
  const n = steps.length;
  const al = alignmentChunks()
    .map(function (c) { return c.k + ':' + c.receipt_step + ':' + c.stale_rows; })
    .join(',');
  const info = episodeInfo();
  const cfg = config();
  const safety = cfg && cfg.safety ? cfg.safety.infer_dry_freeze_s : '';
  return [
    state.run, state.ep, n, n ? steps[n - 1].step : '', state.chunks.length, al,
    info ? info.ended_by + '|' + info.reason : '', cfg ? JSON.stringify(cfg.pin) + '|' + safety : '',
  ].join('#');
}

function build() {
  const t0 = performance.now();
  plotStats.builds += 1;
  const steps = state.steps;
  const n = steps.length;
  const d = {
    n: n,
    step: new Int32Array(n),
    plan: new Float64Array(n),
    t: new Float64Array(n),
    mode: new Uint8Array(n),
    clamped: new Uint8Array(n),
    queue: new Float32Array(n),
    inflight: new Uint8Array(n),
    dry: new Float32Array(n),
    meas: [],
    cmd: [],
    act: [],
    dt: DEFAULT_DT,
    tMin: 0,
    tMax: 1,
    yr: [],
    knuckleRange: [],
    byK: {},
    received: [], // chunks with a receipt step, ascending by receipt
    maxQueue: 0,
    maxDry: 0,
  };
  for (let i = 0; i < DIMS; i += 1) {
    d.meas.push(nanArray(n));
    d.cmd.push(nanArray(n));
    d.act.push(nanArray(n));
  }
  for (let i = 0; i < n; i += 1) {
    const r = steps[i];
    d.step[i] = r.step;
    d.plan[i] = typeof r.plan_step === 'number' ? r.plan_step : r.step;
    const t = num(r.t);
    d.t[i] = isNaN(t) ? (i ? d.t[i - 1] + DEFAULT_DT : 0) : t;
    const m = MODES.indexOf(r.mode);
    d.mode[i] = m < 0 ? 255 : m;
    d.clamped[i] = r.clamped ? 1 : 0;
    d.queue[i] = num(r.queue);
    d.inflight[i] = r.in_flight ? 1 : 0;
    d.dry[i] = num(r.dry_s);
    if (d.queue[i] > d.maxQueue) d.maxQueue = d.queue[i];
    if (d.dry[i] > d.maxDry) d.maxDry = d.dry[i];
    ['left', 'right'].forEach(function (side) {
      const base = ARM_BASE[side];
      const ms = r.measured && Array.isArray(r.measured[side]) ? r.measured[side] : null;
      const wp = r.waypoints && Array.isArray(r.waypoints[side]) ? r.waypoints[side] : null;
      for (let j = 0; j < 7; j += 1) {
        if (ms) d.meas[base + j][i] = num(ms[j]);
        if (wp) d.cmd[base + j][i] = num(wp[j]);
      }
      if (r.knuckle && typeof r.knuckle[side] === 'number') d.meas[GRIP[side]][i] = num(r.knuckle[side]);
    });
    if (Array.isArray(r.action)) {
      for (let j = 0; j < DIMS; j += 1) d.act[j][i] = num(r.action[j]);
    }
  }
  if (n > 1) {
    const dt = (d.t[n - 1] - d.t[0]) / Math.max(1, d.step[n - 1] - d.step[0]);
    if (isFinite(dt) && dt > 0) d.dt = dt;
  }

  // Chunks: rows from /chunks, receipt step and stale rows from /summary alignment (fallback: first record whose
  // plan_step reached obs_step + latency_steps, no stale rows).
  const align = {};
  alignmentChunks().forEach(function (c) { align[c.k] = c; });
  state.chunks.forEach(function (c) {
    const a = align[c.k];
    let receipt = a ? a.receipt_step : undefined;
    if (receipt === undefined && n && typeof c.obs_step === 'number' && typeof c.latency_steps === 'number') {
      const i = lowerBound(d.plan, n, c.obs_step + c.latency_steps);
      receipt = i < n ? d.step[i] : null;
    }
    const rows = Array.isArray(c.rows) ? c.rows : [];
    const entry = {
      k: c.k,
      rows: rows,
      receipt: typeof receipt === 'number' ? receipt : null,
      // Rows skipped on load (async). It can exceed the chunk length (latency > 50 steps): `start` keeps the true
      // offset (row r runs at plan step obs_step + r), `stale` is clamped to the rows for greying.
      start: 0,
      skipped: a && typeof a.stale_rows === 'number' && a.stale_rows > 0 ? a.stale_rows : 0,
      stale: 0,
      latMs: num(c.latency_ms),
      latSteps: num(c.latency_steps),
    };
    entry.stale = Math.min(rows.length, entry.skipped);
    if (entry.receipt !== null) entry.start = entry.receipt - entry.skipped;
    d.byK[c.k] = entry;
    if (entry.receipt !== null) d.received.push(entry);
  });
  d.received.sort(function (a, b) { return a.receipt - b.receipt || a.k - b.k; });

  // x domain: the whole episode plus the furthest planned row (so a live plan never runs off the right edge).
  let lastPlanned = n ? d.step[n - 1] + 1 : 0;
  d.received.forEach(function (c) {
    if (c.rows.length) lastPlanned = Math.max(lastPlanned, c.start + c.rows.length - 1);
  });
  d.tMin = n ? d.t[0] : 0;
  D = d; // tOfStep reads D
  d.tMax = n ? Math.max(d.t[n - 1] + d.dt, tOfStep(lastPlanned)) : 1;
  if (!(d.tMax > d.tMin)) d.tMax = d.tMin + 1;

  // y ranges (stable across the cursor): steps and every chunk row of that dim.
  for (let dim = 0; dim < DIMS; dim += 1) {
    const gripper = dim === 7 || dim === 15;
    let lo = Infinity;
    let hi = -Infinity;
    const arrs = gripper ? [d.act[dim]] : [d.meas[dim], d.cmd[dim], d.act[dim]];
    arrs.forEach(function (a) {
      for (let i = 0; i < n; i += 1) {
        const v = a[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    });
    d.received.forEach(function (c) {
      for (let r = 0; r < c.rows.length; r += 1) {
        const v = c.rows[r] ? num(c.rows[r][dim]) : NaN;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    });
    if (gripper) {
      d.yr.push(pad(Math.min(0, lo), Math.max(1, hi), 0.02, 0.04));
      let klo = Infinity;
      let khi = -Infinity;
      const k = d.meas[dim];
      for (let i = 0; i < n; i += 1) {
        if (k[i] < klo) klo = k[i];
        if (k[i] > khi) khi = k[i];
      }
      d.knuckleRange[dim] = klo <= khi ? pad(klo, khi, 0.02, 0.08) : null;
    } else {
      d.yr.push(lo <= hi ? pad(lo, hi, 0.02, 0.08) : { lo: -1, hi: 1 });
    }
  }
  plotStats.buildMs = performance.now() - t0;
  return d;
}

function pad(lo, hi, minSpan, frac) {
  let span = hi - lo;
  if (span < minSpan) {
    const mid = (lo + hi) / 2;
    lo = mid - minSpan / 2;
    hi = mid + minSpan / 2;
    span = minSpan;
  }
  return { lo: lo - span * frac, hi: hi + span * frac };
}

// First index i < n with arr[i] >= x (n if none).
function lowerBound(arr, n, x) {
  let lo = 0;
  let hi = n;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (arr[m] < x) lo = m + 1;
    else hi = m;
  }
  return lo;
}

// Time of a step number; extrapolated at the episode's step period outside / between records.
function tOfStep(s) {
  const d = D;
  if (!d || !d.n) return 0;
  let i = lowerBound(d.step, d.n, s);
  if (i < d.n && d.step[i] === s) return d.t[i];
  i = Math.max(0, i - 1);
  return d.t[i] + (s - d.step[i]) * d.dt;
}

// Index of the record nearest to time t.
function indexAtT(t) {
  const d = D;
  const i = lowerBound(d.t, d.n, t);
  if (i <= 0) return 0;
  if (i >= d.n) return d.n - 1;
  return t - d.t[i - 1] <= d.t[i] - t ? i - 1 : i;
}

function ensureData() {
  const sig = signature();
  if (sig !== dataSig || !D) {
    dataSig = sig;
    build();
    return true;
  }
  return false;
}

export function chunkAt(step) {
  ensureData();
  const d = D;
  if (!d || !d.n) return null;
  const idx = stepIndex(step);
  const rec = idx >= 0 ? state.steps[idx] : null;
  if (rec && rec.step === step && rec.chunk_k != null) return rec.chunk_k;
  // Newest chunk received at or before the step.
  let best = null;
  for (let i = 0; i < d.received.length; i += 1) {
    if (d.received[i].receipt <= step) best = d.received[i].k;
    else break;
  }
  return best;
}

// ---------------------------------------------------------------- formatting

function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const f = raw / p;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
}

function ticks(lo, hi, count) {
  if (!(hi > lo)) return { vals: [], step: 1 };
  const st = niceStep(hi - lo, count);
  const vals = [];
  for (let v = Math.ceil(lo / st - 1e-9) * st; v <= hi + st * 1e-9; v += st) {
    vals.push(Math.abs(v) < st * 1e-6 ? 0 : v);
  }
  return { vals: vals, step: st };
}

function fmtTick(v, step) {
  const dec = Math.max(0, Math.min(4, Math.ceil(-Math.log10(step) - 1e-9)));
  return fmtSigned(v.toFixed(dec));
}

function fmtSigned(text) {
  return text.charAt(0) === '-' ? MINUS + text.slice(1) : text;
}

function fmtVal(v, dec) {
  return typeof v === 'number' && isFinite(v) ? fmtSigned(v.toFixed(dec)) : '–';
}

function fmtPin(v) {
  return Number.isInteger(v) ? v.toFixed(1) : String(v);
}

function label(dim) {
  const row = state.labels && state.labels[dim];
  return {
    display: row && row.display ? row.display : '[' + dim + '] dim ' + dim,
    long: row && row.long ? row.long : '',
  };
}

// ---------------------------------------------------------------- DOM helpers

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function setText(e, text) {
  if (e.textContent !== text) e.textContent = text;
}

function key(kind, color) {
  const k = el('span', 'key key-' + kind);
  k.setAttribute('aria-hidden', 'true');
  if (color) k.style.setProperty('--key', color);
  return k;
}

// ---------------------------------------------------------------- charts

const charts = []; // every chart on the page
let tlChart = null;
let roots = null;
let tip = null;
let armsSig = '';
let viewKey = 'all';
let legendPlan = null;
let dirty = { build: true, size: true, tlBase: true, base: true, over: true };
let raf = 0;
let pending = []; // charts whose base layer still has to be redrawn
let lastCursor = null;
let lastWin = { x0: NaN, x1: NaN };

function makeChart(parent, opts) {
  const wrap = el('div', 'chart');
  wrap.style.height = opts.h + 'px';
  const base = document.createElement('canvas');
  const over = document.createElement('canvas');
  over.className = 'over';
  const hair = el('div', 'hair');
  hair.hidden = true;
  wrap.appendChild(base);
  wrap.appendChild(over);
  wrap.appendChild(hair);
  parent.appendChild(wrap);
  const c = Object.assign(
    { padL: 48, padR: 10, padT: 6, padB: 4, xLabels: false, dim: -1, side: null },
    opts,
    { wrap: wrap, base: base, over: over, hair: hair, w: 0, bctx: base.getContext('2d'), octx: over.getContext('2d') }
  );
  if (c.xLabels) c.padB += 16;
  charts.push(c);
  attachPointer(c);
  return c;
}

function resize(c) {
  const w = Math.floor(c.wrap.clientWidth);
  if (w === c.w || w <= 0) return w > 0;
  c.w = w;
  const dpr = window.devicePixelRatio || 1;
  [c.base, c.over].forEach(function (cv) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(c.h * dpr);
    cv.style.width = w + 'px';
    cv.style.height = c.h + 'px';
  });
  c.bctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return true;
}

function plotBox(c) {
  return { x: c.padL, y: c.padT, w: Math.max(1, c.w - c.padL - c.padR), h: Math.max(1, c.h - c.padT - c.padB) };
}

function xOf(c, win, t) {
  const b = plotBox(c);
  return b.x + (t - win.x0) / (win.x1 - win.x0) * b.w;
}

function tAtX(c, win, x) {
  const b = plotBox(c);
  return win.x0 + (x - b.x) / b.w * (win.x1 - win.x0);
}

// Time window shown by arms + health (the timeline always shows the whole episode).
function currentWindow() {
  const d = D;
  const all = { x0: d.tMin, x1: d.tMax };
  const w = WINDOWS.find(function (x) { return x.key === viewKey; });
  if (!w || !w.s || d.tMax - d.tMin <= w.s) return all;
  const centre = tOfStep(state.cursor);
  let x0 = centre - w.s / 2;
  x0 = Math.max(d.tMin, Math.min(d.tMax - w.s, x0));
  return { x0: x0, x1: x0 + w.s };
}

function indexRange(win) {
  const d = D;
  const i0 = Math.max(0, lowerBound(d.t, d.n, win.x0) - 1);
  const i1 = Math.min(d.n - 1, lowerBound(d.t, d.n, win.x1));
  return [i0, i1];
}

// One polyline from typed arrays; NaN breaks it. With more points than pixels, each pixel column draws its
// first / min / max / last value (same picture, a fraction of the vertices).
function drawSeries(ctx, c, win, vals, yr, range, stepped) {
  const d = D;
  const b = plotBox(c);
  const kx = b.w / (win.x1 - win.x0);
  const ky = b.h / (yr.hi - yr.lo);
  const bottom = b.y + b.h;
  const t = d.t;
  const i0 = range[0];
  const i1 = range[1];
  ctx.beginPath();
  let pen = false;
  if (i1 - i0 <= b.w) {
    let py = 0;
    for (let i = i0; i <= i1; i += 1) {
      const v = vals[i];
      if (v !== v) { pen = false; continue; }
      const x = b.x + (t[i] - win.x0) * kx;
      const y = bottom - (v - yr.lo) * ky;
      if (!pen) { ctx.moveTo(x, y); pen = true; } else {
        if (stepped) ctx.lineTo(x, py);
        ctx.lineTo(x, y);
      }
      py = y;
    }
  } else {
    let col = -1;
    let has = false;
    let cx = 0;
    let first = 0;
    let lo = 0;
    let hi = 0;
    let last = 0;
    // Each pixel column becomes one vertical stroke min..max, entered from the side it continues from.
    let prev = 0;
    const flush = function () {
      if (!has) return;
      const down = first <= prev; // arriving from below: draw hi..lo upward order keeps the path short
      const a = down ? hi : lo;
      const z = down ? lo : hi;
      if (!pen) { ctx.moveTo(cx, a); pen = true; } else ctx.lineTo(cx, a);
      if (z !== a) ctx.lineTo(cx, z);
      prev = last;
      has = false;
    };
    for (let i = i0; i <= i1; i += 1) {
      const v = vals[i];
      if (v !== v) { flush(); pen = false; col = -1; continue; }
      const x = b.x + (t[i] - win.x0) * kx;
      const y = bottom - (v - yr.lo) * ky;
      const px = Math.floor(x / 2);
      if (px !== col) {
        flush();
        col = px;
        cx = x;
        first = lo = hi = last = y;
        has = true;
      } else {
        if (y < lo) lo = y;
        if (y > hi) hi = y;
        last = y;
      }
    }
    flush();
  }
  ctx.stroke();
}

function clipBox(ctx, c, slack) {
  const b = plotBox(c);
  ctx.beginPath();
  ctx.rect(b.x, b.y - slack, b.w, b.h + 2 * slack);
  ctx.clip();
}

function yOf(c, yr, v) {
  const b = plotBox(c);
  return b.y + b.h - (v - yr.lo) / (yr.hi - yr.lo) * b.h;
}

function drawYAxis(ctx, c, yr, side, count) {
  const b = plotBox(c);
  let tk = ticks(yr.lo, yr.hi, count);
  // keep labels at least 13px apart
  while (count > 1 && tk.vals.length > 1 && tk.step / (yr.hi - yr.lo) * b.h < 13) {
    count -= 1;
    tk = ticks(yr.lo, yr.hi, count);
  }
  if (tk.vals.length > 1 && tk.step / (yr.hi - yr.lo) * b.h < 13) tk = { vals: [tk.vals[0], tk.vals[tk.vals.length - 1]], step: tk.step };
  ctx.font = FONT;
  ctx.textBaseline = 'middle';
  ctx.fillStyle = C.text;
  ctx.textAlign = side === 'left' ? 'right' : 'left';
  tk.vals.forEach(function (v) {
    const y = Math.round(yOf(c, yr, v)) + 0.5;
    if (side === 'left') {
      ctx.strokeStyle = C.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(b.x, y);
      ctx.lineTo(b.x + b.w, y);
      ctx.stroke();
    }
    const ty = Math.max(b.y + 4, Math.min(b.y + b.h - 3, y));
    ctx.fillText(fmtTick(v, tk.step), side === 'left' ? b.x - 6 : b.x + b.w + 6, ty);
  });
}

function drawXAxis(ctx, c, win) {
  const b = plotBox(c);
  const tk = ticks(win.x0, win.x1, Math.max(2, Math.floor(b.w / 90)));
  ctx.strokeStyle = C.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  tk.vals.forEach(function (v) {
    const x = Math.round(xOf(c, win, v)) + 0.5;
    ctx.moveTo(x, b.y);
    ctx.lineTo(x, b.y + b.h);
  });
  ctx.stroke();
  // baseline
  ctx.strokeStyle = C.axis;
  ctx.beginPath();
  ctx.moveTo(b.x, Math.round(b.y + b.h) + 0.5);
  ctx.lineTo(b.x + b.w, Math.round(b.y + b.h) + 0.5);
  ctx.stroke();
  if (!c.xLabels) return;
  ctx.font = FONT;
  ctx.fillStyle = C.text;
  ctx.textBaseline = 'top';
  ctx.textAlign = 'center';
  tk.vals.forEach(function (v) {
    const x = xOf(c, win, v);
    ctx.fillText(fmtTick(v, tk.step) + ' s', Math.max(b.x + 10, Math.min(b.x + b.w - 10, x)), b.y + b.h + 4);
  });
}

function clear(ctx, c) {
  ctx.clearRect(0, 0, c.w, c.h);
}

function cursorLine(ctx, c, win, top) {
  const x = xOf(c, win, tOfStep(state.cursor));
  const b = plotBox(c);
  if (x < b.x - 1 || x > b.x + b.w + 1) return;
  ctx.strokeStyle = C.ink;
  ctx.globalAlpha = 0.85;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x, top != null ? top : b.y);
  ctx.lineTo(x, b.y + b.h);
  ctx.stroke();
  ctx.globalAlpha = 1;
}

// ---- joint / gripper plots

function drawJointBase(c, win, range) {
  const ctx = c.bctx;
  const d = D;
  const dim = c.dim;
  const gripper = c.kind === 'gripper';
  const yr = d.yr[dim];
  clear(ctx, c);
  drawYAxis(ctx, c, yr, 'left', 3);
  const kr = gripper ? d.knuckleRange[dim] : null;
  if (kr) drawYAxis(ctx, c, kr, 'right', 3);
  drawXAxis(ctx, c, win);
  ctx.save();
  clipBox(ctx, c, 2);
  ctx.lineJoin = 'bevel';
  ctx.lineCap = 'butt';
  if (gripper) {
    if (kr) {
      ctx.strokeStyle = C.measured;
      ctx.lineWidth = 2;
      drawSeries(ctx, c, win, d.meas[dim], kr, range, false);
    }
    ctx.strokeStyle = C.action;
    ctx.lineWidth = 1.5;
    drawSeries(ctx, c, win, d.act[dim], yr, range, false);
  } else {
    ctx.strokeStyle = C.measured;
    ctx.lineWidth = 2;
    drawSeries(ctx, c, win, d.meas[dim], yr, range, false);
    ctx.strokeStyle = C.commanded;
    ctx.lineWidth = 2;
    drawSeries(ctx, c, win, d.cmd[dim], yr, range, false);
    ctx.strokeStyle = C.action;
    ctx.lineWidth = 1; // 1px stays on the canvas fast path
    drawSeries(ctx, c, win, d.act[dim], yr, range, false);
    // Step clamp fired on this joint: the commanded waypoint differs from the policy action.
    const b = plotBox(c);
    const yb = b.y + b.h;
    ctx.fillStyle = C.clamp;
    ctx.beginPath();
    let lastX = -10;
    for (let i = range[0]; i <= range[1]; i += 1) {
      if (!d.clamped[i]) continue;
      const a = d.act[dim][i];
      const w = d.cmd[dim][i];
      if (!(Math.abs(a - w) > 1e-9)) continue;
      const x = b.x + (d.t[i] - win.x0) / (win.x1 - win.x0) * b.w;
      if (x - lastX < 3) continue;
      lastX = x;
      ctx.moveTo(x, yb - 7);
      ctx.lineTo(x - 4, yb);
      ctx.lineTo(x + 4, yb);
      ctx.closePath();
    }
    ctx.fill();
  }
  ctx.restore();
}

function planOf(k) {
  return k === null || !D ? null : D.byK[k] || null;
}

function drawJointOver(c, win, plan) {
  const ctx = c.octx;
  clear(ctx, c);
  const d = D;
  if (plan && plan.receipt !== null && plan.rows.length) {
    const dim = c.dim;
    const yr = d.yr[dim];
    const startStep = plan.start;
    ctx.save();
    clipBox(ctx, c, 2);
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    const seg = function (r0, r1) {
      ctx.beginPath();
      let pen = false;
      for (let r = r0; r <= r1; r += 1) {
        const v = plan.rows[r] ? num(plan.rows[r][dim]) : NaN;
        if (v !== v) { pen = false; continue; }
        const x = xOf(c, win, tOfStep(startStep + r));
        const y = yOf(c, yr, v);
        if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };
    const last = plan.rows.length - 1;
    if (plan.stale > 0) {
      ctx.strokeStyle = C.stale;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      seg(0, Math.min(plan.stale, last));
      ctx.setLineDash([]);
    }
    if (plan.stale <= last) {
      ctx.strokeStyle = C.plan;
      ctx.globalAlpha = 0.9;
      ctx.lineWidth = 2;
      seg(plan.stale, last);
      ctx.globalAlpha = 1;
      // receipt marker: the row that runs at the receipt step, ringed in the surface colour
      const v = plan.rows[plan.stale] ? num(plan.rows[plan.stale][dim]) : NaN;
      if (v === v) {
        const x = xOf(c, win, tOfStep(plan.receipt));
        const y = yOf(c, yr, v);
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = C.plan;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = C.surface;
        ctx.stroke();
      }
    }
    ctx.restore();
  }
  cursorLine(ctx, c, win);
}

// ---- timeline

function drawTimelineBase(c, win) {
  const ctx = c.bctx;
  const d = D;
  const b = plotBox(c);
  clear(ctx, c);
  ctx.font = FONT;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'right';
  ctx.fillStyle = C.text;
  ctx.fillText('mode', b.x - 6, b.y + b.h / 2);
  drawXAxis(ctx, c, win);
  const segs = [];
  let i = 0;
  while (i < d.n) {
    let j = i;
    while (j + 1 < d.n && d.mode[j + 1] === d.mode[i]) j += 1;
    const x0 = xOf(c, win, d.t[i]);
    const x1 = xOf(c, win, j + 1 < d.n ? d.t[j + 1] : d.t[j] + d.dt);
    segs.push({ i: i, j: j, x0: x0, x1: x1 });
    i = j + 1;
  }
  segs.forEach(function (s) {
    const m = d.mode[s.i];
    const name = m === 255 ? 'unknown' : MODES[m];
    // 2px surface gap between neighbouring segments when there is room for it
    const gap = s.x1 - s.x0 > 4 ? 1 : 0;
    ctx.fillStyle = modeColor(name);
    ctx.fillRect(s.x0 + gap, b.y, Math.max(1, s.x1 - s.x0 - 2 * gap), b.h);
    const width = s.x1 - s.x0 - 8;
    if (width < 20) return;
    const reason = state.steps[s.i] && state.steps[s.i].reason;
    const words = [name.toUpperCase() + (reason ? ' · ' + reason : ''), name.toUpperCase()];
    ctx.font = FONT_B;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = MODE_INK[name] || '#ffffff';
    for (let k = 0; k < words.length; k += 1) {
      const text = fitText(ctx, words[k], width, k === 0 && reason);
      if (text) {
        ctx.fillText(text, s.x0 + 4, b.y + b.h / 2 + 0.5);
        break;
      }
    }
  });
  // episode end marker
  const info = episodeInfo();
  if (info && info.ended_by && d.n) {
    const freeze = info.ended_by === 'freeze';
    const x = Math.round(xOf(c, win, d.t[d.n - 1] + d.dt)) + 0.5;
    ctx.strokeStyle = freeze ? C.freeze : C.ink;
    ctx.lineWidth = freeze ? 2 : 1.5;
    ctx.beginPath();
    ctx.moveTo(x, 2);
    ctx.lineTo(x, b.y + b.h);
    ctx.stroke();
    const text = info.ended_by + ': ' + (info.reason || 'no reason recorded');
    ctx.font = freeze ? FONT_B : FONT;
    ctx.fillStyle = freeze ? C.freeze : C.ink;
    ctx.textBaseline = 'middle';
    const roomLeft = x - b.x - 6;
    const roomRight = c.w - x - 6;
    if (roomLeft >= roomRight) {
      ctx.textAlign = 'right';
      ctx.fillText(fitText(ctx, text, roomLeft, true) || '', x - 5, 8);
    } else {
      ctx.textAlign = 'left';
      ctx.fillText(fitText(ctx, text, roomRight, true) || '', x + 5, 8);
    }
  }
}

// The text, shortened with an ellipsis when allowed, or '' when it cannot fit.
function fitText(ctx, text, width, canShorten) {
  if (ctx.measureText(text).width <= width) return text;
  if (!canShorten) return '';
  let lo = 0;
  let hi = text.length;
  while (lo < hi) {
    const m = (lo + hi + 1) >> 1;
    if (ctx.measureText(text.slice(0, m) + '…').width <= width) lo = m;
    else hi = m - 1;
  }
  return lo >= 4 ? text.slice(0, lo) + '…' : '';
}

function drawTimelineOver(c, win, armWin) {
  const ctx = c.octx;
  const b = plotBox(c);
  clear(ctx, c);
  if (armWin.x0 > win.x0 || armWin.x1 < win.x1) {
    const x0 = xOf(c, win, armWin.x0);
    const x1 = xOf(c, win, armWin.x1);
    ctx.fillStyle = 'rgba(11, 11, 11, 0.08)';
    ctx.fillRect(x0, b.y - 4, x1 - x0, b.h + 8);
    ctx.strokeStyle = 'rgba(11, 11, 11, 0.45)';
    ctx.lineWidth = 1;
    ctx.strokeRect(Math.round(x0) + 0.5, b.y - 4.5, Math.round(x1 - x0), b.h + 9);
  }
  const x = xOf(c, win, tOfStep(state.cursor));
  ctx.strokeStyle = C.ink;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, b.y - 4);
  ctx.lineTo(x, b.y + b.h + 2);
  ctx.stroke();
  ctx.fillStyle = C.ink;
  ctx.beginPath();
  ctx.moveTo(x - 5, b.y - 9);
  ctx.lineTo(x + 5, b.y - 9);
  ctx.lineTo(x, b.y - 3);
  ctx.closePath();
  ctx.fill();
}

// ---- health

function drawHealthBase(c, win, range) {
  const ctx = c.bctx;
  const d = D;
  const b = plotBox(c);
  clear(ctx, c);
  ctx.lineJoin = 'round';
  if (c.kind === 'queue') {
    const yr = { lo: 0, hi: Math.max(1, d.maxQueue) * 1.15 };
    drawYAxis(ctx, c, { lo: 0, hi: Math.max(1, d.maxQueue) }, 'left', Math.min(3, Math.max(1, d.maxQueue)));
    drawXAxis(ctx, c, win);
    // in-flight lane above the plot
    ctx.font = FONT;
    ctx.fillStyle = C.text;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText('in flight', b.x - 6, 7);
    ctx.fillStyle = C.lane;
    const kx = b.w / (win.x1 - win.x0);
    let i = range[0];
    while (i <= range[1]) {
      if (!d.inflight[i]) { i += 1; continue; }
      let j = i;
      while (j + 1 <= range[1] && d.inflight[j + 1]) j += 1;
      const x0 = Math.max(b.x, b.x + (d.t[i] - win.x0) * kx);
      const x1 = Math.min(b.x + b.w, b.x + ((j + 1 < d.n ? d.t[j + 1] : d.t[j] + d.dt) - win.x0) * kx);
      if (x1 > x0) ctx.fillRect(x0, 2, Math.max(1, x1 - x0), 10);
      i = j + 1;
    }
    ctx.save();
    clipBox(ctx, c, 2);
    ctx.strokeStyle = C.series;
    ctx.lineWidth = 2;
    drawSeries(ctx, c, win, d.queue, yr, range, true);
    ctx.restore();
  } else if (c.kind === 'dry') {
    const cfg = config();
    const thr = cfg && cfg.safety && typeof cfg.safety.infer_dry_freeze_s === 'number' ? cfg.safety.infer_dry_freeze_s : null;
    const top = Math.max(thr !== null ? thr * 1.15 : 0, d.maxDry * 1.1, 0.5);
    const yr = { lo: 0, hi: top };
    drawYAxis(ctx, c, yr, 'left', 2);
    drawXAxis(ctx, c, win);
    ctx.save();
    clipBox(ctx, c, 2);
    ctx.strokeStyle = C.series;
    ctx.lineWidth = 2;
    drawSeries(ctx, c, win, d.dry, yr, range, true);
    ctx.restore();
    if (thr !== null) {
      const y = Math.round(yOf(c, yr, thr)) + 0.5;
      ctx.strokeStyle = C.clamp;
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(b.x, y);
      ctx.lineTo(b.x + b.w, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = FONT;
      ctx.fillStyle = C.text;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'bottom';
      ctx.fillText('FREEZE at infer_dry_freeze_s = ' + thr + ' s', b.x + b.w - 4, y - 2);
    }
  } else {
    const ms = c.kind === 'latms';
    const yr = latencyRange(ms);
    drawYAxis(ctx, c, yr, 'left', 2);
    drawXAxis(ctx, c, win);
    ctx.save();
    clipBox(ctx, c, 6);
    d.received.forEach(function (ch) {
      const v = ms ? ch.latMs : ch.latSteps;
      if (v !== v) return;
      const x = xOf(c, win, tOfStep(ch.receipt));
      if (x < b.x - 6 || x > b.x + b.w + 6) return;
      ctx.beginPath();
      ctx.arc(x, yOf(c, yr, v), 4, 0, Math.PI * 2);
      ctx.fillStyle = C.series;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = C.surface;
      ctx.stroke();
    });
    ctx.restore();
  }
}

function latencyRange(ms) {
  let max = 0;
  D.received.forEach(function (ch) {
    const v = ms ? ch.latMs : ch.latSteps;
    if (v > max) max = v;
  });
  return { lo: 0, hi: Math.max(ms ? 50 : 2, max) * 1.15 };
}

function drawHealthOver(c, win, k) {
  const ctx = c.octx;
  clear(ctx, c);
  if ((c.kind === 'latms' || c.kind === 'latsteps') && k !== null) {
    const ch = D.byK[k];
    const v = ch ? (c.kind === 'latms' ? ch.latMs : ch.latSteps) : NaN;
    if (ch && ch.receipt !== null && v === v) {
      ctx.beginPath();
      ctx.arc(xOf(c, win, tOfStep(ch.receipt)), yOf(c, latencyRange(c.kind === 'latms'), v), 7, 0, Math.PI * 2);
      ctx.strokeStyle = C.ink;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
  cursorLine(ctx, c, win, c.kind === 'queue' ? 2 : null);
}

// ---------------------------------------------------------------- frame

function schedule() {
  if (!raf) raf = requestAnimationFrame(frame);
}

function frame() {
  raf = 0;
  const t0 = performance.now();
  if (dirty.build) {
    ensureData();
    dirty.build = false;
  }
  const has = !!(state.ep && D && D.n);
  showSections(has);
  if (!has) return;
  if (dirty.size) {
    charts.forEach(function (c) {
      if (resize(c)) dirty.base = dirty.tlBase = dirty.over = true;
    });
    dirty.size = false;
  }
  const full = { x0: D.tMin, x1: D.tMax };
  const win = currentWindow();
  if (win.x0 !== lastWin.x0 || win.x1 !== lastWin.x1) {
    lastWin = win;
    dirty.base = dirty.over = true;
  }
  // Base layers are queued and drawn within a per-frame budget, so a full redraw (new data, resize, moved
  // window) is spread over a few frames instead of blocking one; overlays are cheap and drawn every time.
  if (dirty.tlBase || dirty.base) {
    pending = charts.filter(function (c) {
      return c.kind === 'timeline' ? dirty.tlBase || pending.indexOf(c) >= 0 : dirty.base || pending.indexOf(c) >= 0;
    });
    dirty.tlBase = dirty.base = false;
  }
  const range = indexRange(win);
  while (pending.length) {
    const c = pending.shift();
    if (!c.w) continue;
    if (c.kind === 'timeline') drawTimelineBase(c, full);
    else if (c.kind === 'joint' || c.kind === 'gripper') drawJointBase(c, win, range);
    else drawHealthBase(c, win, range);
    if (performance.now() - t0 > BASE_BUDGET_MS) break;
  }
  if (dirty.over) {
    const k = chunkAt(state.cursor);
    const plan = planOf(k);
    charts.forEach(function (c) {
      if (!c.w) return;
      if (c.kind === 'timeline') drawTimelineOver(c, full, win);
      else if (c.kind === 'joint' || c.kind === 'gripper') drawJointOver(c, win, plan);
      else drawHealthOver(c, win, k);
    });
    updatePlanLegend(plan);
    dirty.over = false;
  }
  if (pending.length) schedule();
  const ms = performance.now() - t0;
  plotStats.frames += 1;
  plotStats.lastMs = ms;
  if (ms > plotStats.maxMs) plotStats.maxMs = ms;
}

function showSections(has) {
  roots.timeline.hidden = !state.ep;
  tlChart.wrap.hidden = !has;
  setText(roots.tlNote, state.ep && !has ? 'Waiting for the first step of this episode.' : '');
  roots.arms.hidden = !has;
  roots.health.hidden = !has;
}

function updatePlanLegend(plan) {
  if (!legendPlan) return;
  let text = 'plan of the chunk at the cursor';
  if (plan && plan.receipt !== null) {
    text = 'plan: chunk ' + plan.k + ', received at step ' + plan.receipt;
    if (plan.skipped >= plan.rows.length && plan.rows.length) text += ' (all ' + plan.rows.length + ' rows stale)';
    else if (plan.stale) text += ' (' + plan.stale + ' stale rows)';
  } else if (D && D.n) {
    text = 'plan: no chunk received yet at the cursor';
  }
  setText(legendPlan, text);
}

// ---------------------------------------------------------------- pointer: hover tooltips, click to set cursor

function attachPointer(c) {
  let dragging = false;
  const stepAt = function (e) {
    if (!D || !D.n) return null;
    const rect = c.wrap.getBoundingClientRect();
    const win = c.kind === 'timeline' ? { x0: D.tMin, x1: D.tMax } : lastWin;
    const t = tAtX(c, win, e.clientX - rect.left);
    if (t > D.t[D.n - 1] + D.dt) return { i: -1, t: t };
    return { i: indexAtT(t), t: t };
  };
  c.wrap.addEventListener('mousemove', function (e) {
    const hit = stepAt(e);
    if (!hit) return;
    if (dragging && hit.i >= 0) setCursor(D.step[hit.i]);
    showTip(c, hit, e);
  });
  c.wrap.addEventListener('mouseleave', function () {
    hideTip(c);
  });
  c.wrap.addEventListener('mousedown', function (e) {
    if (e.button !== 0) return;
    const hit = stepAt(e);
    if (!hit) return;
    e.preventDefault();
    setCursor(hit.i >= 0 ? D.step[hit.i] : D.step[D.n - 1]);
    // Only the timeline drags: its scale is fixed, a windowed plot would scroll under the pointer.
    if (c.kind === 'timeline') {
      dragging = true;
      const up = function () {
        dragging = false;
        window.removeEventListener('mouseup', up);
      };
      window.addEventListener('mouseup', up);
    }
  });
}

function hideTip(c) {
  if (tip) tip.hidden = true;
  if (c) c.hair.hidden = true;
}

function tipRow(table, color, kind, name, value) {
  const tr = el('tr');
  const k = el('td');
  if (color) k.appendChild(key(kind, color));
  tr.appendChild(k);
  tr.appendChild(el('td', 'tip-name', name));
  tr.appendChild(el('td', 'tip-val', value));
  table.appendChild(tr);
}

function showTip(c, hit, e) {
  const d = D;
  tip.textContent = '';
  if (hit.i < 0) {
    if (c.kind === 'timeline') {
      hideTip(c);
      return;
    }
  }
  const i = hit.i;
  const rec = i >= 0 ? state.steps[i] : null;
  const head = el('div', 'tip-head');
  const table = el('table');
  if (rec) {
    head.textContent = 'step ' + rec.step + ' · t ' + fmtVal(d.t[i], 2) + ' s · ' + String(rec.mode).toUpperCase();
  } else {
    head.textContent = 't ' + fmtVal(hit.t, 2) + ' s (after the last step)';
  }
  tip.appendChild(head);
  if (c.kind === 'timeline') {
    if (rec && rec.reason) tip.appendChild(el('div', 'tip-reason', rec.reason));
    if (rec && rec.chunk_k != null) tip.appendChild(el('div', 'tip-sub', 'chunk ' + rec.chunk_k + ', row ' + rec.row));
  } else if (c.kind === 'joint' || c.kind === 'gripper') {
    const dim = c.dim;
    const lb = label(dim);
    head.textContent = lb.display + ' · ' + head.textContent;
    const plan = planOf(chunkAt(state.cursor));
    const step = rec ? rec.step : Math.round(d.step[d.n - 1] + (hit.t - d.t[d.n - 1]) / d.dt);
    if (c.kind === 'gripper') {
      tipRow(table, C.measured, 'line', 'measured knuckle', i >= 0 && d.meas[dim][i] === d.meas[dim][i] ? fmtVal(d.meas[dim][i], 3) + ' rad' : 'not recorded');
      tipRow(table, C.action, 'line', 'policy action width', i >= 0 ? fmtVal(d.act[dim][i], 3) : '–');
    } else {
      tipRow(table, C.measured, 'line', 'measured', i >= 0 ? fmtVal(d.meas[dim][i], 4) + ' rad' : '–');
      tipRow(table, C.commanded, 'line', 'commanded', i >= 0 ? fmtVal(d.cmd[dim][i], 4) : '–');
      tipRow(table, C.action, 'line', 'policy action', i >= 0 ? fmtVal(d.act[dim][i], 4) : '–');
      if (i >= 0 && d.clamped[i] && Math.abs(d.act[dim][i] - d.cmd[dim][i]) > 1e-9) {
        tipRow(table, C.clamp, 'tri', 'step clamp cut this joint', fmtVal(d.cmd[dim][i] - d.act[dim][i], 4));
      }
    }
    if (plan && plan.receipt !== null) {
      const r = step - plan.start;
      if (r >= 0 && r < plan.rows.length) {
        const v = plan.rows[r] ? num(plan.rows[r][dim]) : NaN;
        tipRow(table, r < plan.stale ? C.stale : C.plan, r < plan.stale ? 'dash' : 'line',
          'plan chunk ' + plan.k + ' row ' + r + (r < plan.stale ? ' (stale)' : ''), fmtVal(v, 4));
      }
    }
    const pin = pinnedValue(dim);
    if (pin !== null) tip.appendChild(el('div', 'tip-sub', 'policy saw ' + fmtPin(pin) + ' (pinned)'));
  } else if (c.kind === 'queue') {
    tipRow(table, C.series, 'line', 'queue', i >= 0 ? fmtVal(d.queue[i], 0) : '–');
    tipRow(table, C.lane, 'box', 'request in flight', i >= 0 ? (d.inflight[i] ? 'yes' : 'no') : '–');
  } else if (c.kind === 'dry') {
    tipRow(table, C.series, 'line', 'dry_s', i >= 0 ? fmtVal(d.dry[i], 2) + ' s' : '–');
  } else {
    // nearest received chunk
    let best = null;
    let bestDx = Infinity;
    d.received.forEach(function (ch) {
      const dx = Math.abs(tOfStep(ch.receipt) - hit.t);
      if (dx < bestDx) { bestDx = dx; best = ch; }
    });
    if (best) {
      head.textContent = 'chunk ' + best.k + ' · received at step ' + best.receipt;
      tipRow(table, C.series, 'dot', 'latency', fmtVal(best.latMs, 0) + ' ms');
      tipRow(table, null, '', '', fmtVal(best.latSteps, 0) + ' steps');
      if (best.skipped) tipRow(table, null, '', 'stale rows', String(best.stale) + (best.skipped > best.stale ? ' (all)' : ''));
    }
  }
  if (table.firstChild) tip.appendChild(table);
  tip.hidden = false;
  // place near the pointer, inside the viewport
  const pad = 14;
  const tw = tip.offsetWidth;
  const th = tip.offsetHeight;
  let x = e.clientX + pad;
  let y = e.clientY + pad;
  if (x + tw > window.innerWidth - 4) x = e.clientX - pad - tw;
  if (y + th > window.innerHeight - 4) y = e.clientY - pad - th;
  tip.style.left = Math.max(4, x) + 'px';
  tip.style.top = Math.max(4, y) + 'px';
  // hairline at the hovered step
  if (c.kind !== 'timeline' && i >= 0) {
    c.hair.hidden = false;
    c.hair.style.left = Math.round(xOf(c, lastWin, d.t[i])) + 'px';
    c.hair.style.top = c.padT + 'px';
    c.hair.style.height = (c.h - c.padT - c.padB) + 'px';
  } else {
    c.hair.hidden = true;
  }
}

// ---------------------------------------------------------------- layout

function buildTimeline() {
  const root = roots.timeline;
  root.textContent = '';
  root.setAttribute('aria-label', 'Mode timeline');
  const head = el('div', 'sec-head');
  head.appendChild(el('h2', 'sec-title', 'Mode timeline'));
  const legend = el('div', 'legend');
  MODES.forEach(function (m) {
    const item = el('span', 'legend-item');
    item.appendChild(key('box', modeColor(m)));
    item.appendChild(document.createTextNode(m));
    legend.appendChild(item);
  });
  legend.appendChild(el('span', 'legend-note', 'hover for the reason · click or drag to move the cursor'));
  head.appendChild(legend);
  root.appendChild(head);
  roots.tlNote = el('div', 'note');
  root.appendChild(roots.tlNote);
  tlChart = makeChart(root, { kind: 'timeline', h: 66, padT: 22, padB: 4, xLabels: true });
}

function buildArms() {
  const sides = activeArms();
  const sig = sides.join(',');
  if (sig === armsSig) return;
  armsSig = sig;
  // drop old arm charts
  for (let i = charts.length - 1; i >= 0; i -= 1) {
    if (charts[i].kind === 'joint' || charts[i].kind === 'gripper') charts.splice(i, 1);
  }
  const root = roots.arms;
  root.textContent = '';
  root.setAttribute('aria-label', 'Joint plots');

  const head = el('div', 'sec-head');
  head.appendChild(el('h2', 'sec-title', 'Joints'));
  const legend = el('div', 'legend');
  const item = function (kind, color, text) {
    const s = el('span', 'legend-item');
    s.appendChild(key(kind, color));
    const t = el('span', null, text);
    s.appendChild(t);
    legend.appendChild(s);
    return t;
  };
  item('line', C.measured, 'measured');
  item('line', C.commanded, 'commanded (waypoints)');
  item('line', C.action, 'policy action');
  legendPlan = item('line', C.plan, 'plan of the chunk at the cursor');
  item('dash', C.stale, 'stale rows');
  item('tri', C.clamp, 'step clamp fired');
  item('cursor', C.ink, 'cursor');
  head.appendChild(legend);

  const view = el('div', 'view-switch');
  view.setAttribute('role', 'group');
  view.setAttribute('aria-label', 'Time window');
  view.appendChild(el('span', 'muted', 'Time window'));
  WINDOWS.forEach(function (w) {
    const b = el('button', null, w.label);
    b.type = 'button';
    b.setAttribute('data-view', w.key);
    b.setAttribute('aria-pressed', w.key === viewKey ? 'true' : 'false');
    b.addEventListener('click', function () {
      viewKey = w.key;
      view.querySelectorAll('button').forEach(function (x) {
        x.setAttribute('aria-pressed', x.getAttribute('data-view') === viewKey ? 'true' : 'false');
      });
      b.blur(); // keep the step keys working
      dirty.base = dirty.over = true;
      schedule();
    });
    view.appendChild(b);
  });
  head.appendChild(view);
  root.appendChild(head);

  const cols = el('div', 'arm-cols');
  cols.style.setProperty('--cols', String(sides.length));
  sides.forEach(function (side) {
    const col = el('div', 'arm-col');
    col.appendChild(el('h3', 'arm-title', SIDE_NAME[side]));
    const base = ARM_BASE[side];
    for (let j = 0; j < 8; j += 1) {
      const dim = j < 7 ? base + j : GRIP[side];
      const gripper = j === 7;
      const plot = el('div', 'plot');
      const ph = el('div', 'plot-head');
      const title = el('button', 'plot-title');
      title.type = 'button';
      title.setAttribute('data-dim', String(dim));
      title.addEventListener('click', function () {
        setSelectedDim(dim);
        title.blur();
      });
      ph.appendChild(title);
      const sub = el('span', 'plot-sub');
      ph.appendChild(sub);
      plot.appendChild(ph);
      col.appendChild(plot);
      const c = makeChart(plot, {
        kind: gripper ? 'gripper' : 'joint',
        dim: dim,
        side: side,
        h: gripper ? 84 : 64,
        padR: gripper ? 48 : 10,
        xLabels: gripper,
      });
      c.title = title;
      c.sub = sub;
    }
    cols.appendChild(col);
  });
  root.appendChild(cols);
  updateTitles();
  dirty.size = dirty.base = dirty.over = true;
}

function buildHealth() {
  const root = roots.health;
  root.textContent = '';
  root.setAttribute('aria-label', 'Health');
  const head = el('div', 'sec-head');
  head.appendChild(el('h2', 'sec-title', 'Health'));
  root.appendChild(head);
  const grid = el('div', 'health-grid');
  const add = function (kind, title, sub) {
    const box = el('div', 'plot');
    const ph = el('div', 'plot-head');
    ph.appendChild(el('span', 'plot-name', title));
    if (sub) ph.appendChild(el('span', 'plot-sub', sub));
    box.appendChild(ph);
    grid.appendChild(box);
    makeChart(box, { kind: kind, h: kind === 'queue' ? 92 : 80, padT: kind === 'queue' ? 18 : 8, xLabels: true });
  };
  add('queue', 'Queue (actions left) and request in flight', 'shaded lane = request in flight');
  add('dry', 'Dry time dry_s (s)', 'dashed = freeze threshold');
  add('latms', 'Chunk latency (ms)', 't_receive − t_request, at the receipt step');
  add('latsteps', 'Chunk latency (steps)', 'latency_steps, at the receipt step');
  root.appendChild(grid);
}

function updateTitles() {
  charts.forEach(function (c) {
    if (!c.title) return;
    const lb = label(c.dim);
    setText(c.title, lb.display);
    c.title.title = (lb.long ? lb.long + '. ' : '') + 'Click to show this dim in the chunk inspector.';
    let sub = 'rad';
    if (c.kind === 'gripper') {
      const d = D;
      const hasKnuckle = !!(d && d.knuckleRange[c.dim]);
      sub = 'action width 0–1 (left axis)' + (hasKnuckle ? ' · measured knuckle rad (right axis)' : ' · knuckle not recorded');
      const pin = pinnedValue(c.dim);
      if (pin !== null) sub += ' · policy saw ' + fmtPin(pin) + ' (pinned)';
    }
    setText(c.sub, sub);
  });
  updateSelected();
}

function updateSelected() {
  charts.forEach(function (c) {
    if (!c.title) return;
    const on = c.dim === state.selectedDim;
    c.title.setAttribute('aria-pressed', on ? 'true' : 'false');
    c.title.parentNode.parentNode.classList.toggle('selected', on);
  });
}

// ---------------------------------------------------------------- mount

export function mountPlots(root) {
  const scope = root || document;
  const find = function (id) {
    return scope.querySelector ? scope.querySelector('#' + id) : document.getElementById(id);
  };
  roots = { timeline: find('timeline'), arms: find('arms'), health: find('health'), tlNote: null };
  if (!roots.timeline || !roots.arms || !roots.health) {
    console.error('mountPlots: #timeline, #arms or #health missing');
    return;
  }
  tip = el('div', 'plot-tip');
  tip.hidden = true;
  tip.setAttribute('role', 'tooltip');
  document.body.appendChild(tip);

  buildTimeline();
  buildArms();
  buildHealth();
  showSections(false);

  if (typeof ResizeObserver === 'function') {
    const ro = new ResizeObserver(function () {
      dirty.size = true;
      schedule();
    });
    [roots.timeline, roots.arms, roots.health].forEach(function (r) { ro.observe(r); });
  } else {
    window.addEventListener('resize', function () {
      dirty.size = true;
      schedule();
    });
  }

  onChange(function (kind) {
    if (kind === 'labels') {
      updateTitles();
      return;
    }
    if (kind === 'dim') {
      updateSelected();
      return;
    }
    if (kind === 'selection' || kind === 'run') buildArms();
    const sig = signature();
    if (sig !== dataSig) {
      ensureData(); // rebuild the typed arrays now, so chunkAt() and the titles are current right away
      updateTitles();
      dirty.size = dirty.tlBase = dirty.base = dirty.over = true;
    }
    if (state.cursor !== lastCursor) {
      lastCursor = state.cursor;
      dirty.over = true;
    }
    if (dirty.build || dirty.size || dirty.tlBase || dirty.base || dirty.over) schedule();
  });
  schedule();
}
