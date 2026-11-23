// Chunk inspector and episode result.
//
// Inspector (#inspector): the chunk the plan line shows at the cursor (chunkAt() from plots.js, so both agree).
//   header      chunk k, request plan step, receipt step, latency, first, executed rows, stale rows
//   checks      start offset per active arm against safety.max_start_offset_rad, alignment mismatches, clamps
//   plot        one dimension (selector; shared with the joint plot titles through state.selectedDim):
//               the planned rows, the measured value at the steps that executed them, executed rows highlighted,
//               stale rows grey, a marker at execute_horizon
// Result (#result): result.json, recorder errors, stream_stats.json.
//
// Idle cost: every redraw is gated by a signature. A 500 ms 'data' poll that changed nothing relevant costs one
// signature compare; a cursor move inside the same chunk only redraws the small overlay canvas.

import { state, onChange, stepIndex, setCursor, setSelectedDim, episodeInfo } from '/static/app.js';
import { chunkAt } from '/static/plots.js';

const DIMS = 16;
const ARM_OF_DIM = [
  { side: 'left', base: 0 }, { side: 'right', base: 8 },
];
const MINUS = '−';
const EN = '–';

// Plot colours. The planned line is violet (categorical slot 7), not the aqua of the arm plots' plan line, which is
// also the WAIT mode colour. Ink = measured, grey = stale. Validated with the dataviz palette script (all pairs).
const C = {
  surface: '#ffffff',
  grid: '#ecebe6',
  axis: '#c3c2b7',
  text: '#6b6a65',
  ink: '#0b0b0b',
  measured: '#3f3e3b',
  plan: '#6250d6',
  planFaint: 'rgba(98, 80, 214, 0.45)',
  execBand: 'rgba(98, 80, 214, 0.12)',
  staleBand: 'rgba(120, 119, 112, 0.16)',
  stale: '#9c9b95',
  horizon: '#3f3e3b',
};
const FONT = '11px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
const FONT_B = '600 11px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
const PLOT_H = 250;
const PAD = { l: 52, r: 12, t: 12, b: 26 };

const els = {};
let mounted = false;
let sig = '';
let overSig = '';
let resultSig = '';
let cfgRef = null;
let cfgKey = '';
let raf = 0;
let hoverRow = -1;
let view = null; // what the plot currently draws, see computeView()

// ---------------------------------------------------------------- helpers

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function setText(e, text) {
  if (e.textContent !== text) e.textContent = text;
}

function num(v) {
  return typeof v === 'number' && isFinite(v) ? v : NaN;
}

function signed(text) {
  return text.charAt(0) === '-' ? MINUS + text.slice(1) : text;
}

function fmt(v, dec) {
  return typeof v === 'number' && isFinite(v) ? signed(v.toFixed(dec)) : EN;
}

// Generic value for the result and stream-stats tables.
function fmtAny(v) {
  if (v === null || v === undefined) return EN;
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  if (typeof v === 'number') {
    if (!isFinite(v)) return EN;
    return Number.isInteger(v) ? String(v) : signed(String(Number(v.toPrecision(4))));
  }
  if (typeof v === 'string') return v === '' ? EN : v;
  return JSON.stringify(v);
}

function config() {
  return state.runInfo && state.runInfo.config ? state.runInfo.config : null;
}

function activeArms() {
  const cfg = config();
  const arms = cfg && Array.isArray(cfg.active_arms) ? cfg.active_arms : ['left', 'right'];
  const out = ['right', 'left'].filter(function (s) { return arms.indexOf(s) >= 0; });
  return out.length ? out : ['right', 'left'];
}

function labelOf(dim) {
  const row = state.labels && state.labels[dim];
  return {
    display: row && row.display ? row.display : '[' + dim + '] dim ' + dim,
    long: row && row.long ? row.long : '',
  };
}

function chunkEntry(k) {
  const list = state.chunks;
  if (!list.length) return null;
  const guess = list[k];
  if (guess && guess.k === k) return guess;
  for (let i = list.length - 1; i >= 0; i -= 1) {
    if (list[i].k === k) return list[i];
  }
  return null;
}

function alignEntry(k) {
  const s = state.summary;
  const list = s && s.alignment && Array.isArray(s.alignment.chunks) ? s.alignment.chunks : null;
  if (!list) return null;
  const guess = list[k];
  if (guess && guess.k === k) return guess;
  for (let i = 0; i < list.length; i += 1) {
    if (list[i].k === k) return list[i];
  }
  return null;
}

// ---------------------------------------------------------------- view model of the inspected chunk

// The chunk to inspect: the one the plan line shows at the cursor. Exception: a chunk whose receipt tick has no
// step record yet (start check FREEZE, or a live episode whose steps lag by a tick) is not in the plan line's
// list. After a start-check FREEZE it is the last thing the episode did, so show it at any cursor; on a live
// episode show it at the newest step only (it is the chunk about to run).
function pickChunk() {
  const k = chunkAt(state.cursor);
  if (k !== null) return k;
  const n = state.steps.length;
  const info = episodeInfo();
  const froze = !!(info && info.ended_by === 'freeze');
  if (!n || (!froze && state.steps[n - 1].step !== state.cursor)) return null;
  const s = state.summary;
  const list = s && s.alignment && Array.isArray(s.alignment.chunks) ? s.alignment.chunks : [];
  for (let i = list.length - 1; i >= 0; i -= 1) {
    if (list[i].receipt_recorded === false && typeof list[i].receipt_step === 'number' &&
        list[i].receipt_step > state.cursor && chunkEntry(list[i].k)) {
      return list[i].k;
    }
  }
  return null;
}

// Rows that were executed and the measured value of the selected dim at those steps (from the step records).
function computeView(k, ch, al, dim) {
  const rows = ch && Array.isArray(ch.rows) ? ch.rows : [];
  const n = rows.length;
  const v = {
    k: k,
    dim: dim,
    n: n,
    plan: new Float64Array(n),
    meas: new Float64Array(n),
    exec: new Uint8Array(n),
    stepOf: new Int32Array(n),
    stale: 0,
    skipped: 0,
    lo: 0,
    hi: 1,
    rowsExec: null, // [first row, last row] or null
    gripper: dim === 7 || dim === 15,
  };
  v.meas.fill(NaN);
  v.stepOf.fill(-1);
  for (let r = 0; r < n; r += 1) v.plan[r] = rows[r] ? num(rows[r][dim]) : NaN;
  if (al && typeof al.stale_rows === 'number' && al.stale_rows > 0) {
    v.skipped = al.stale_rows;
    v.stale = Math.min(n, al.stale_rows);
  }
  const arm = dim < 8 ? ARM_OF_DIM[0] : ARM_OF_DIM[1];
  const joint = dim - arm.base;
  const steps = state.steps;
  let i0 = 0;
  let i1 = steps.length - 1;
  if (al && al.executed && al.executed.length === 2) {
    i0 = Math.max(0, stepIndex(al.executed[0]));
    i1 = Math.min(steps.length - 1, stepIndex(al.executed[1]));
  } else if (al && al.executed === null && al.n_executed === 0) {
    i1 = -1;
  }
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = i0; i <= i1; i += 1) {
    const rec = steps[i];
    if (rec.chunk_k !== k || typeof rec.row !== 'number' || rec.row < 0 || rec.row >= n) continue;
    const r = rec.row;
    v.exec[r] = 1;
    v.stepOf[r] = rec.step;
    if (!v.gripper) {
      const ms = rec.measured && Array.isArray(rec.measured[arm.side]) ? rec.measured[arm.side] : null;
      if (ms) v.meas[r] = num(ms[joint]);
    }
  }
  for (let r = 0; r < n; r += 1) {
    if (v.exec[r]) {
      if (!v.rowsExec) v.rowsExec = [r, r];
      else v.rowsExec[1] = r;
    }
    const a = v.plan[r];
    const m = v.meas[r];
    if (a < lo) lo = a;
    if (a > hi) hi = a;
    if (m < lo) lo = m;
    if (m > hi) hi = m;
  }
  if (!(lo <= hi)) {
    lo = -1;
    hi = 1;
  }
  if (v.gripper) {
    lo = Math.min(0, lo);
    hi = Math.max(1, hi);
  }
  const span = Math.max(hi - lo, 0.02);
  v.lo = lo - span * 0.08;
  v.hi = hi + span * 0.08;
  return v;
}

// ---------------------------------------------------------------- DOM skeleton

function buildInspector(root) {
  root.textContent = '';
  const head = el('div', 'sec-head');
  head.appendChild(el('h2', 'sec-title', 'Chunk inspector'));
  const sel = el('label', 'field insp-dim');
  sel.appendChild(document.createTextNode('Dimension '));
  els.dim = document.createElement('select');
  els.dim.setAttribute('aria-label', 'Policy dimension shown in the chunk inspector');
  sel.appendChild(els.dim);
  head.appendChild(sel);
  els.pickNote = el('span', 'note', 'click a joint title above to pick one');
  head.appendChild(els.pickNote);
  root.appendChild(head);

  els.empty = el('p', 'muted insp-empty');
  els.empty.hidden = true;
  root.appendChild(els.empty);

  els.body = el('div', 'insp-body');
  els.header = el('div', 'insp-header');
  els.body.appendChild(els.header);

  const grid = el('div', 'insp-grid');
  els.side = el('div', 'insp-side');
  grid.appendChild(els.side);

  const plotBlock = el('div', 'insp-plot');
  els.plotTitle = el('div', 'insp-plot-title');
  els.plotName = el('b', 'insp-plot-name');
  els.plotSub = el('span', 'plot-sub');
  els.plotTitle.appendChild(els.plotName);
  els.plotTitle.appendChild(els.plotSub);
  plotBlock.appendChild(els.plotTitle);

  els.legend = el('div', 'legend insp-legend');
  const items = [
    ['line', C.plan, 'planned row (executed rows bold)'],
    ['dash', C.stale, 'stale row (skipped on load)'],
    ['dot', C.measured, 'measured at the step that ran the row'],
    ['box', C.execBand, 'executed rows'],
    ['dash', C.horizon, 'execute_horizon'],
  ];
  items.forEach(function (it) {
    const item = el('span', 'legend-item');
    const k = el('span', 'key key-' + it[0]);
    k.setAttribute('aria-hidden', 'true');
    k.style.setProperty('--key', it[1]);
    item.appendChild(k);
    item.appendChild(document.createTextNode(it[2]));
    els.legend.appendChild(item);
  });
  plotBlock.appendChild(els.legend);

  els.chart = el('div', 'chart insp-chart');
  els.chart.style.height = PLOT_H + 'px';
  els.base = document.createElement('canvas');
  els.over = document.createElement('canvas');
  els.over.className = 'over';
  els.chart.appendChild(els.base);
  els.chart.appendChild(els.over);
  plotBlock.appendChild(els.chart);
  els.readout = el('div', 'insp-readout muted');
  els.readout.setAttribute('aria-live', 'off');
  plotBlock.appendChild(els.readout);
  grid.appendChild(plotBlock);
  els.body.appendChild(grid);
  root.appendChild(els.body);

  els.bctx = els.base.getContext('2d');
  els.octx = els.over.getContext('2d');
  els.w = 0;

  els.dim.addEventListener('change', function () {
    const d = parseInt(els.dim.value, 10);
    els.dim.blur(); // keep the step keys (arrows, space) working
    if (isFinite(d)) setSelectedDim(d);
  });

  // Hover: one readout line under the plot. Click: move the cursor to the step that ran the row.
  els.chart.addEventListener('mousemove', function (e) {
    const r = rowAtEvent(e);
    if (r !== hoverRow) {
      hoverRow = r;
      scheduleOver();
    }
  });
  els.chart.addEventListener('mouseleave', function () {
    if (hoverRow !== -1) {
      hoverRow = -1;
      scheduleOver();
    }
  });
  els.chart.addEventListener('mousedown', function (e) {
    if (e.button !== 0 || !view) return;
    const r = rowAtEvent(e);
    if (r >= 0 && view.stepOf[r] >= 0) {
      e.preventDefault();
      setCursor(view.stepOf[r]);
    }
  });

  if (typeof ResizeObserver === 'function') {
    new ResizeObserver(function () { refresh(); }).observe(els.chart);
  } else {
    window.addEventListener('resize', function () { refresh(); });
  }
}

function fillDimSelect() {
  const want = [];
  for (let i = 0; i < DIMS; i += 1) want.push(labelOf(i).display);
  const have = Array.prototype.map.call(els.dim.options, function (o) { return o.textContent; });
  if (have.length !== DIMS || have.join('|') !== want.join('|')) {
    els.dim.textContent = '';
    want.forEach(function (text, i) {
      const o = document.createElement('option');
      o.value = String(i);
      o.textContent = text;
      els.dim.appendChild(o);
    });
  }
  if (els.dim.value !== String(state.selectedDim)) els.dim.value = String(state.selectedDim);
}

// ---------------------------------------------------------------- text panels

function pill(cls, text, title) {
  const p = el('span', 'pill ' + cls, text);
  if (title) p.title = title;
  return p;
}

function kvSpan(label, value, cls, title) {
  const s = el('span', 'insp-kv');
  s.appendChild(document.createTextNode(label + ' '));
  const b = el('b', cls || '', value);
  if (title) b.title = title;
  s.appendChild(b);
  return s;
}

function sep() {
  return el('span', 'insp-sep', '·');
}

function renderHeader(k, ch, al, info) {
  const h = els.header;
  h.textContent = '';
  const parts = [];
  parts.push(kvSpan('chunk', String(k)));
  parts.push(kvSpan('requested at plan step', ch && typeof ch.obs_step === 'number' ? String(ch.obs_step) : EN));
  const approx = al && al.receipt_approx ? '≈ ' : '';
  parts.push(kvSpan('received step', al && typeof al.receipt_step === 'number' ? approx + al.receipt_step : EN,
    '', al && al.receipt_approx ? 'async: the step the receipt landed on is approximate' : ''));
  const ms = ch ? num(ch.latency_ms) : NaN;
  const ls = ch ? num(ch.latency_steps) : NaN;
  parts.push(kvSpan('latency', fmt(ms, 0) + ' ms / ' + (isFinite(ls) ? String(ls) : EN) + ' steps'));
  parts.push(kvSpan('first:', al && al.first !== null && al.first !== undefined ? (al.first ? 'yes' : 'no') : EN));
  let executed = 'none';
  if (view && view.rowsExec) executed = view.rowsExec[0] + EN + view.rowsExec[1];
  parts.push(kvSpan('executed rows', executed + (al ? ' (' + al.n_executed + ')' : '')));
  const nRows = view ? view.n : 0;
  let stale = String(view ? view.stale : 0);
  let staleTitle = '';
  if (view && view.skipped > nRows && nRows > 0) staleTitle = view.skipped + ' rows skipped on load; the chunk only has ' + nRows;
  if (view && nRows > 0 && view.stale >= nRows) stale += ' (all rows stale)';
  parts.push(kvSpan('stale rows', stale, view && view.stale > 0 ? 'amber-text' : '', staleTitle));
  parts.forEach(function (p, i) {
    if (i) h.appendChild(sep());
    h.appendChild(p);
  });
}

function startOffsetRow(side, value, limit, checked) {
  const row = el('div', 'insp-row');
  row.appendChild(el('span', 'insp-row-name', side + ' arm'));
  // The runner checks the start offset only for the first chunk after start / HOLD (runner.py first_chunk).
  const over = checked && typeof value === 'number' && typeof limit === 'number' && value > limit;
  const val = el('span', 'insp-row-val' + (over ? ' over' : ''), value === null || value === undefined ? EN : value.toFixed(3) + ' rad');
  row.appendChild(val);
  if (typeof limit === 'number') {
    row.appendChild(el('span', 'insp-row-lim muted', 'max ' + limit + ' rad'));
  }
  if (over) row.appendChild(pill('pill-red', 'OVER LIMIT'));
  return row;
}

// Async has no value check on the server (mismatches is null): count the run ticks whose executed action differs
// from the row of the chunk they were assigned to, i.e. the ticks where the executed value was blended.
let blendKey = '';
let blendCount = null;
function blended() {
  const steps = state.steps;
  const key = state.run + '#' + state.ep + '#' + steps.length + '#' + (steps.length ? steps[steps.length - 1].step : '') +
    '#' + state.chunks.length;
  if (key === blendKey) return blendCount;
  blendKey = key;
  blendCount = null;
  if (!steps.length || !state.chunks.length) return null;
  const byK = {};
  state.chunks.forEach(function (c) { byK[c.k] = c; });
  let count = 0;
  for (let i = 0; i < steps.length; i += 1) {
    const rec = steps[i];
    if (rec.mode !== 'run' || rec.chunk_k == null || typeof rec.row !== 'number' || !Array.isArray(rec.action)) continue;
    const c = byK[rec.chunk_k];
    const row = c && Array.isArray(c.rows) ? c.rows[rec.row] : null;
    if (!row) continue;
    for (let j = 0; j < DIMS; j += 1) {
      if (Math.abs(num(rec.action[j]) - num(row[j])) > 1e-5) {
        count += 1;
        break;
      }
    }
  }
  blendCount = count;
  return count;
}

function renderSide(k, ch, al, info) {
  const s = els.side;
  s.textContent = '';
  const cfg = config();
  const safety = cfg && cfg.safety ? cfg.safety : null;
  const limit = safety && typeof safety.max_start_offset_rad === 'number' ? safety.max_start_offset_rad : null;

  const so = el('div', 'insp-card');
  so.appendChild(el('h3', 'insp-h', 'Start offset (row 0 vs measured at receipt)'));
  activeArms().forEach(function (side) {
    const v = al && al.start_offset ? al.start_offset[side] : null;
    so.appendChild(startOffsetRow(side, typeof v === 'number' ? v : null, limit, !!(al && al.first === true)));
  });
  if (al && al.first !== true) {
    so.appendChild(el('div', 'note', al.first === false
      ? 'not checked (not first chunk): the start check only applies to the first chunk after start / HOLD'
      : 'not checked (first chunk not known yet)'));
  }
  if (limit === null) so.appendChild(el('div', 'note', 'safety.max_start_offset_rad not known for this run'));
  s.appendChild(so);

  const notes = el('div', 'insp-card');
  notes.appendChild(el('h3', 'insp-h', 'Alignment'));
  const mode = cfg && cfg.mode === 'async' ? 'async' : 'sync';
  const summary = state.summary;
  const mism = mode === 'async' ? blended() : summary && summary.alignment ? summary.alignment.mismatches : null;
  const line = el('div', 'insp-row');
  if (typeof mism === 'number') {
    const text = mode === 'async'
      ? mism + ' ticks differ from their chunk row (expected when chunks blend)'
      : 'alignment: ' + mism + ' mismatches';
    const t = el('span', mode !== 'async' && mism > 0 ? 'pill pill-amber' : 'insp-row-val', text);
    t.title = mode === 'async'
      ? 'run ticks whose executed action differs from the row of their chunk (async blends chunks)'
      : 'run ticks whose executed action differs from the row the queue should have given';
    line.appendChild(t);
  } else {
    line.appendChild(el('span', 'muted', 'alignment: not available yet'));
  }
  notes.appendChild(line);

  const nClamped = al && typeof al.n_clamped === 'number' ? al.n_clamped : 0;
  const cl = el('div', 'insp-row');
  cl.appendChild(el('span', nClamped > 0 ? 'pill pill-amber' : 'insp-row-val', nClamped + ' clamped tick' + (nClamped === 1 ? '' : 's')));
  notes.appendChild(cl);

  if (al && al.receipt_recorded === false) {
    const froze = info && info.ended_by === 'freeze';
    const msg = froze ? 'receipt tick not recorded (start check FREEZE)' : 'receipt tick not yet recorded';
    const r = el('div', 'insp-row');
    r.appendChild(el('span', froze ? 'pill pill-red' : 'pill pill-amber', msg));
    notes.appendChild(r);
    if (froze && info.reason) notes.appendChild(el('div', 'note insp-reason', info.reason));
  }
  if (!al) {
    notes.appendChild(el('div', 'note', 'no alignment info for this chunk yet'));
  }
  s.appendChild(notes);
}

// ---------------------------------------------------------------- plot

function sizeCanvas() {
  const w = Math.floor(els.chart.clientWidth);
  if (w <= 0) return false;
  if (w === els.w) return true;
  els.w = w;
  const dpr = window.devicePixelRatio || 1;
  [els.base, els.over].forEach(function (cv) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(PLOT_H * dpr);
    cv.style.width = w + 'px';
    cv.style.height = PLOT_H + 'px';
  });
  els.bctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  els.octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return true;
}

function box() {
  return { x: PAD.l, y: PAD.t, w: Math.max(1, els.w - PAD.l - PAD.r), h: PLOT_H - PAD.t - PAD.b };
}

function xOfRow(r) {
  const b = box();
  const n = Math.max(1, view.n);
  return b.x + (r + 0.5) / n * b.w;
}

function yOfVal(v) {
  const b = box();
  return b.y + b.h - (v - view.lo) / (view.hi - view.lo) * b.h;
}

function rowAtEvent(e) {
  if (!view || !view.n || !els.w) return -1;
  const rect = els.chart.getBoundingClientRect();
  const b = box();
  const x = e.clientX - rect.left;
  if (x < b.x || x > b.x + b.w) return -1;
  return Math.max(0, Math.min(view.n - 1, Math.floor((x - b.x) / b.w * view.n)));
}

function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const f = raw / p;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
}

function drawBase() {
  const ctx = els.bctx;
  const w = els.w;
  ctx.clearRect(0, 0, w, PLOT_H);
  ctx.fillStyle = C.surface;
  ctx.fillRect(0, 0, w, PLOT_H);
  if (!view || !view.n) return;
  const b = box();
  const n = view.n;
  const colW = b.w / n;

  // Row bands: stale grey, executed violet tint.
  for (let r = 0; r < n; r += 1) {
    if (r < view.stale) ctx.fillStyle = C.staleBand;
    else if (view.exec[r]) ctx.fillStyle = C.execBand;
    else continue;
    ctx.fillRect(b.x + r * colW, b.y, Math.ceil(colW), b.h);
  }

  // Grid + y axis.
  ctx.font = FONT;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'right';
  ctx.lineWidth = 1;
  const st = niceStep(view.hi - view.lo, 5);
  const dec = Math.max(0, Math.min(4, Math.ceil(-Math.log10(st) - 1e-9)));
  for (let v = Math.ceil(view.lo / st - 1e-9) * st; v <= view.hi + st * 1e-9; v += st) {
    const y = Math.round(yOfVal(v)) + 0.5;
    ctx.strokeStyle = C.grid;
    ctx.beginPath();
    ctx.moveTo(b.x, y);
    ctx.lineTo(b.x + b.w, y);
    ctx.stroke();
    ctx.fillStyle = C.text;
    ctx.fillText(signed((Math.abs(v) < st * 1e-6 ? 0 : v).toFixed(dec)), b.x - 6, y);
  }
  // x axis: rows.
  ctx.strokeStyle = C.axis;
  ctx.beginPath();
  ctx.moveTo(b.x, b.y + b.h + 0.5);
  ctx.lineTo(b.x + b.w, b.y + b.h + 0.5);
  ctx.stroke();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillStyle = C.text;
  const xs = n > 40 ? 5 : n > 20 ? 2 : 1;
  for (let r = 0; r < n; r += xs) {
    ctx.fillText(String(r), xOfRow(r), b.y + b.h + 4);
  }
  ctx.textAlign = 'right';
  ctx.fillText('row', b.x + b.w, b.y + b.h + 14);

  // execute_horizon marker (between row H-1 and row H).
  const cfg = config();
  const horizon = cfg && typeof cfg.execute_horizon === 'number' ? cfg.execute_horizon : null;
  if (horizon !== null && horizon > 0 && horizon < n) {
    const x = Math.round(b.x + horizon * colW) + 0.5;
    ctx.save();
    ctx.strokeStyle = C.horizon;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(x, b.y);
    ctx.lineTo(x, b.y + b.h);
    ctx.stroke();
    ctx.restore();
    ctx.font = FONT_B;
    ctx.fillStyle = C.horizon;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('execute_horizon = ' + horizon, x + 4, b.y + 2);
  }

  ctx.save();
  ctx.beginPath();
  ctx.rect(b.x, b.y - 4, b.w, b.h + 8);
  ctx.clip();

  // Planned rows: faint line over all rows, then bold over executed, grey dashed over stale.
  const px = function (r) { return xOfRow(r); };
  const path = function (from, to, pred) {
    ctx.beginPath();
    let pen = false;
    for (let r = from; r < to; r += 1) {
      const v = view.plan[r];
      if (v !== v || (pred && !pred(r))) { pen = false; continue; }
      if (!pen) { ctx.moveTo(px(r), yOfVal(v)); pen = true; } else ctx.lineTo(px(r), yOfVal(v));
    }
    ctx.stroke();
  };
  ctx.lineJoin = 'round';
  ctx.strokeStyle = C.planFaint;
  ctx.lineWidth = 1.5;
  path(view.stale, n, null);
  ctx.strokeStyle = C.plan;
  ctx.lineWidth = 3;
  path(view.stale, n, function (r) { return view.exec[r] === 1; });
  if (view.stale > 0) {
    ctx.save();
    ctx.strokeStyle = C.stale;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    path(0, Math.min(n, view.stale + 1), null);
    ctx.restore();
  }
  // Row dots on the planned line.
  for (let r = 0; r < n; r += 1) {
    const v = view.plan[r];
    if (v !== v) continue;
    ctx.fillStyle = r < view.stale ? C.stale : view.exec[r] ? C.plan : C.planFaint;
    ctx.beginPath();
    ctx.arc(px(r), yOfVal(v), view.exec[r] ? 2.5 : 1.8, 0, 6.2832);
    ctx.fill();
  }

  // Measured at the steps that ran the rows.
  ctx.strokeStyle = C.measured;
  ctx.lineWidth = 1.5;
  path2(view.meas, px);
  ctx.fillStyle = C.measured;
  for (let r = 0; r < n; r += 1) {
    const m = view.meas[r];
    if (m !== m) continue;
    ctx.beginPath();
    ctx.arc(px(r), yOfVal(m), 2, 0, 6.2832);
    ctx.fill();
  }
  ctx.restore();
}

function path2(vals, px) {
  const ctx = els.bctx;
  ctx.beginPath();
  let pen = false;
  for (let r = 0; r < vals.length; r += 1) {
    const v = vals[r];
    if (v !== v) { pen = false; continue; }
    if (!pen) { ctx.moveTo(px(r), yOfVal(v)); pen = true; } else ctx.lineTo(px(r), yOfVal(v));
  }
  ctx.stroke();
}

// Cursor row (the step the cursor sits on ran this row of this chunk) and the hover row.
function cursorRow() {
  if (!view) return -1;
  const idx = stepIndex(state.cursor);
  const rec = idx >= 0 ? state.steps[idx] : null;
  if (rec && rec.step === state.cursor && rec.chunk_k === view.k && typeof rec.row === 'number' && rec.row < view.n) return rec.row;
  return -1;
}

function drawOver() {
  const ctx = els.octx;
  ctx.clearRect(0, 0, els.w, PLOT_H);
  if (!view || !view.n) return;
  const b = box();
  const cr = cursorRow();
  if (cr >= 0) {
    const x = Math.round(xOfRow(cr)) + 0.5;
    ctx.strokeStyle = C.ink;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, b.y);
    ctx.lineTo(x, b.y + b.h);
    ctx.stroke();
    const v = view.plan[cr];
    if (v === v) {
      ctx.fillStyle = C.surface;
      ctx.strokeStyle = C.ink;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(xOfRow(cr), yOfVal(v), 5, 0, 6.2832);
      ctx.fill();
      ctx.stroke();
    }
  }
  if (hoverRow >= 0) {
    const x = Math.round(xOfRow(hoverRow)) + 0.5;
    ctx.strokeStyle = 'rgba(11, 11, 11, 0.35)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, b.y);
    ctx.lineTo(x, b.y + b.h);
    ctx.stroke();
  }
  setText(els.readout, readoutText(hoverRow >= 0 ? hoverRow : cr, hoverRow >= 0));
}

function readoutText(r, hovered) {
  if (!view || r < 0 || r >= view.n) {
    return 'hover a row to read it; click to move the cursor to the step that ran it';
  }
  const parts = ['row ' + r];
  const ch = chunkEntry(view.k);
  if (ch && typeof ch.obs_step === 'number') parts.push('plan step ' + (ch.obs_step + r));
  parts.push('planned ' + fmt(view.plan[r], 4));
  if (!view.gripper) parts.push('measured ' + fmt(view.meas[r], 4));
  parts.push(r < view.stale ? 'stale (skipped on load)' : view.exec[r] ? 'executed at step ' + view.stepOf[r] : 'not executed');
  return (hovered ? '' : 'cursor: ') + parts.join(' · ');
}

// ---------------------------------------------------------------- scheduling

function scheduleOver() {
  if (raf) return;
  raf = window.requestAnimationFrame(function () {
    raf = 0;
    if (mounted && view) drawOver();
  });
}

function renderPlotHead() {
  const dim = state.selectedDim;
  const lab = labelOf(dim);
  els.plotName.textContent = lab.display;
  els.plotName.title = lab.long;
  const cfg = config();
  let sub = view && view.gripper
    ? ' · width 0–1 · measured knuckle (rad) is another scale, see the arm plot'
    : ' · rad';
  const pin = cfg && cfg.pin && typeof cfg.pin === 'object' ? cfg.pin[String(dim)] : undefined;
  if (typeof pin === 'number') sub += ' · policy saw ' + (Number.isInteger(pin) ? pin.toFixed(1) : pin) + ' (pinned)';
  setText(els.plotSub, sub);
}

// Everything the inspector's text and base layer depend on apart from the cursor.
function signature(k, ch, al, info) {
  const steps = state.steps;
  const last = steps.length ? steps[steps.length - 1].step : -1;
  // The step records only matter while the chunk is still executing (or there is no alignment span yet).
  const growing = !al || !al.executed || al.executed[1] >= last;
  const s = state.summary;
  return [
    state.run, state.ep, k, state.selectedDim, els.w, cfgKey,
    ch ? ch.rows.length + ':' + ch.latency_ms + ':' + ch.latency_steps : '-',
    al ? JSON.stringify(al) : '-',
    growing ? steps.length : '',
    s && s.alignment ? s.alignment.mismatches : '',
    config() && config().mode === 'async' ? blended() : '',
    info ? info.ended_by + '|' + info.reason : '',
    state.labels ? state.labels.length : 0,
  ].join('#');
}

function refresh() {
  if (!mounted) return;
  const root = els.root;
  const has = !!state.ep;
  if (root.hidden === has) root.hidden = !has;
  if (!has) {
    sig = '';
    overSig = '';
    return;
  }
  sizeCanvas(); // not laid out yet (width 0): the ResizeObserver calls refresh again
  if (state.runInfo !== cfgRef) {
    cfgRef = state.runInfo;
    cfgKey = JSON.stringify(config());
  }
  const k = pickChunk();
  const ch = k === null ? null : chunkEntry(k);
  const al = k === null ? null : alignEntry(k);
  const info = episodeInfo();
  const s = k === null ? 'none#' + state.run + '#' + state.ep + '#' + (state.steps.length ? 1 : 0) : signature(k, ch, al, info);
  if (s !== sig) {
    sig = s;
    fillDimSelect();
    if (k === null || !ch) {
      els.body.hidden = true;
      els.empty.hidden = false;
      setText(els.empty, state.steps.length
        ? 'No chunk has been received at or before this step.'
        : 'Waiting for the first step of this episode.');
      view = null;
    } else {
      els.body.hidden = false;
      els.empty.hidden = true;
      view = computeView(k, ch, al, state.selectedDim);
      renderHeader(k, ch, al, info);
      renderSide(k, ch, al, info);
      renderPlotHead();
      if (els.w) drawBase();
    }
    overSig = '';
  }
  if (view) {
    const o = cursorRow() + '#' + hoverRow + '#' + sig;
    if (o !== overSig) {
      overSig = o;
      scheduleOver();
    }
  }
}

// ---------------------------------------------------------------- result panel

function statValue(key, v) {
  const out = { text: fmtAny(v), cls: '' };
  if (key === 'ended_by' && v === 'freeze') out.cls = 'bad';
  else if ((key === 'holds' || key === 'step_clamps' || key === 'infer_errors' || key === 'overruns') && typeof v === 'number' && v > 0) {
    out.cls = 'warn';
  } else if (/_ms_(mean|max)$/.test(key) && typeof v === 'number') out.text = v.toFixed(1) + ' ms';
  return out;
}

const STAT_ORDER = [
  'ended_by', 'reason', 'mode', 'steps', 'holds', 'hold_steps', 'wait_steps', 'leash_steps', 'pause_steps',
  'step_clamps', 'infer_ms_mean', 'infer_ms_max', 'requests', 'chunks', 'infer_errors', 'overruns',
  'waypoints', 'gripper_publishes', 'dry_run',
];

function flatten(obj, prefix, out) {
  Object.keys(obj).forEach(function (key) {
    const v = obj[key];
    const name = prefix ? prefix + '.' + key : key;
    if (v && typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length) flatten(v, name, out);
    else out.push([name, v]);
  });
  return out;
}

function renderResult() {
  const info = episodeInfo();
  const summary = state.summary;
  const result = summary ? summary.result : null;
  const stats = summary ? summary.stream_stats : null;
  const s = JSON.stringify([state.run, state.ep, result, stats, info ? [info.ended_by, info.reason, info.outcome, info.live] : null]);
  if (s === resultSig) return;
  resultSig = s;
  const root = els.resultRoot;
  root.hidden = !state.ep;
  root.textContent = '';
  if (!state.ep) return;

  const head = el('div', 'sec-head');
  head.appendChild(el('h2', 'sec-title', 'Episode result'));
  root.appendChild(head);
  const grid = el('div', 'res-grid');

  // result.json
  const card = el('div', 'insp-card res-card');
  card.appendChild(el('h3', 'insp-h', 'result.json'));
  if (!result) {
    card.appendChild(el('p', 'muted', 'written after the operator answers the outcome prompt'));
    if (info && info.ended_by) {
      const t = el('div', 'res-ended');
      t.appendChild(pill(info.ended_by === 'freeze' ? 'pill-red' : 'pill-neutral', 'ended by ' + info.ended_by));
      if (info.reason) t.appendChild(el('span', 'res-reason', info.reason));
      card.appendChild(t);
    } else if (info && info.live) {
      card.appendChild(el('div', 'note', 'episode still running'));
    }
  } else {
    const top = el('div', 'res-top');
    top.appendChild(el('span', 'res-outcome', 'outcome ' + fmtAny(result.outcome)));
    card.appendChild(top);
    card.appendChild(el('div', 'res-note', result.note ? 'note: ' + result.note : 'note: ' + EN));
    const st = result.stats && typeof result.stats === 'object' ? result.stats : {};
    const keys = STAT_ORDER.filter(function (key) { return key in st; });
    Object.keys(st).forEach(function (key) { if (keys.indexOf(key) < 0) keys.push(key); });
    const dl = el('div', 'res-stats');
    keys.forEach(function (key) {
      const v = statValue(key, st[key]);
      const item = el('div', 'res-stat');
      item.appendChild(el('span', 'res-stat-name', key));
      item.appendChild(el('span', 'res-stat-val ' + v.cls, v.text));
      dl.appendChild(item);
    });
    card.appendChild(dl);
    if (typeof result.n_chunks === 'number') card.appendChild(el('div', 'note', result.n_chunks + ' chunks recorded'));
  }
  const errs = result && Array.isArray(result.recorder_errors) ? result.recorder_errors : [];
  const eb = el('div', 'res-errors' + (errs.length ? ' bad' : ''));
  if (!result) eb.hidden = true;
  else if (errs.length) {
    eb.appendChild(el('b', '', 'recorder errors (' + errs.length + ')'));
    const ul = el('ul', '');
    errs.forEach(function (e) { ul.appendChild(el('li', '', typeof e === 'string' ? e : JSON.stringify(e))); });
    eb.appendChild(ul);
  } else {
    eb.textContent = 'recorder errors: none';
  }
  card.appendChild(eb);
  grid.appendChild(card);

  // stream_stats.json
  const sc = el('div', 'insp-card res-card');
  sc.appendChild(el('h3', 'insp-h', 'stream_stats.json'));
  if (!stats) {
    sc.appendChild(el('p', 'muted', 'no stream_stats.json in this episode'));
  } else {
    const rows = flatten(stats, '', []);
    if (!rows.length) sc.appendChild(el('p', 'muted', 'empty'));
    else {
      const table = el('table', 'res-table');
      rows.slice(0, 120).forEach(function (r) {
        const tr = document.createElement('tr');
        tr.appendChild(el('th', '', r[0]));
        const fault = /(^|\.)fault$/.test(r[0]) && r[1] !== null && r[1] !== undefined && r[1] !== '';
        tr.appendChild(el('td', fault ? 'bad' : '', fmtAny(r[1])));
        table.appendChild(tr);
      });
      sc.appendChild(table);
      if (rows.length > 120) sc.appendChild(el('div', 'note', (rows.length - 120) + ' more rows not shown'));
    }
  }
  grid.appendChild(sc);
  root.appendChild(grid);
}

// ---------------------------------------------------------------- mount

export function mountInspector(root) {
  const scope = root || document;
  const find = function (id) {
    return scope.querySelector ? scope.querySelector('#' + id) : document.getElementById(id);
  };
  els.root = find('inspector');
  els.resultRoot = find('result');
  if (!els.root || !els.resultRoot) {
    console.error('mountInspector: #inspector or #result missing');
    return;
  }
  buildInspector(els.root);
  els.root.hidden = true;
  els.resultRoot.hidden = true;
  mounted = true;
  fillDimSelect();

  onChange(function (kind) {
    if (kind === 'labels') {
      fillDimSelect();
      sig = '';
    }
    if (kind === 'dim') {
      fillDimSelect();
    }
    if (kind === 'cursor' || kind === 'dim' || kind === 'labels' || kind === 'selection' || kind === 'run' ||
        kind === 'data' || kind === 'follow') {
      refresh();
      if (kind !== 'cursor' && kind !== 'follow' && kind !== 'dim') renderResult();
    }
  });
  refresh();
  renderResult();
}
