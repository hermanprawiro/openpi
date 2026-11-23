// EBIM task2 run viewer: page shell (navigation, status header, images, cursor, live follow).
//
// Read-only. Talks to robot_client/viewer.py over GET only. ES2017 + modules, no external resources.
//
// Exports for plots.js / inspector.js:
//   state          shared state object (fields documented below)
//   onChange(fn)   subscribe; fn(kind) is called after the state changed, kind is one of
//                    'selection'  run or episode changed (data was reset),
//                    'run'        state.runInfo refreshed,
//                    'data'       steps / chunks / summary arrived,
//                    'cursor'     state.cursor moved,
//                    'follow'     state.follow flipped,
//                    'dim'        state.selectedDim changed,
//                    'labels'     state.labels loaded.
//                  Returns an unsubscribe function. Listener errors are logged, never rethrown.
//   stepIndex(step)  index into state.steps of the record whose .step is the greatest one <= step
//                    (0 when step precedes the first record, -1 when there are no records).
//   setCursor(step)  move the cursor (a user action: leaves follow unless it lands on the newest step).
//   setFollow(on)    set follow; turning it on snaps to the newest step.
//   setSelectedDim(d)
//   episodeInfo()    the episode entry {name, dry_run, ended_by, reason, live, ...} or null.
//   modeColor(mode)  CSS colour of a step mode (reads the --mode-* custom properties).
//   MODES, CAMERAS
//
// state = {
//   labels,      // /api/labels rows
//   runs,        // /api/runs "runs" list (newest first)
//   run,         // selected run name or null
//   runInfo,     // /api/runs/<run> or null
//   ep,          // selected episode name or null
//   steps,       // /steps records (+ chunk_k, row, clamped), ascending by .step
//   chunks,      // /chunks entries (k, obs_step, t_request, t_receive, latency_ms, latency_steps, rows)
//   summary,     // /summary or null
//   cursor,      // current step number (a .step value)
//   follow,      // true = follow the live run: stick to the newest step, open its new episodes, and switch to a
//                //   newer live run when one starts (the selected run may have ended: the button then reads
//                //   "run ended"). Off: nothing moves; a "run X is live — click to follow" banner offers it.
//   selectedDim, // action/state dim shown in the chunk inspector (default 11)
//   playing      // replay running
// }

export const MODES = ['run', 'wait', 'leash', 'hold', 'paused'];
export const CAMERAS = ['head', 'wrist_left', 'wrist_right'];

export const state = {
  labels: [],
  runs: [],
  run: null,
  runInfo: null,
  ep: null,
  steps: [],
  chunks: [],
  summary: null,
  cursor: 0,
  follow: true, // a fresh page (no #step in the URL) follows whatever run is live
  selectedDim: 11,
  playing: false,
};

const RUN_POLL_MS = 2000;
const EPISODE_POLL_MS = 500;
const LIST_POLL_MS = 5000; // always on: the page notices a new or resumed run while another one is shown
const FRESH_S = 60; // a live run whose newest write is older than this loses to a fresher live run
const STALE_STEP_S = 3; // following a live episode whose newest step is older than this: "last step N s ago"
const PLAY_STEPS_PER_S = 20;
const BIG_STEP = 20;
const MINUS = '−';

// ---------------------------------------------------------------- subscribers

const listeners = [];

export function onChange(fn) {
  listeners.push(fn);
  return function unsubscribe() {
    const i = listeners.indexOf(fn);
    if (i >= 0) listeners.splice(i, 1);
  };
}

function emit(kind) {
  try {
    render();
  } catch (e) {
    console.error('render failed', e);
  }
  listeners.slice().forEach(function (fn) {
    try {
      fn(kind);
    } catch (e) {
      console.error('onChange listener failed', e);
    }
  });
}

// ---------------------------------------------------------------- helpers

export function stepIndex(step) {
  const steps = state.steps;
  if (!steps.length) return -1;
  let lo = 0;
  let hi = steps.length - 1;
  if (step < steps[0].step) return 0;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (steps[mid].step <= step) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

export function modeColor(mode) {
  const known = MODES.indexOf(mode) >= 0 ? mode : 'paused';
  const v = getComputedStyle(document.documentElement).getPropertyValue('--mode-' + known).trim();
  return v || '#7a7a76';
}

// Greatest value <= x in an ascending array of numbers, or null.
function floorIn(list, x) {
  if (!list || !list.length) return null;
  if (x < list[0]) return null;
  let lo = 0;
  let hi = list.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (list[mid] <= x) lo = mid;
    else hi = mid - 1;
  }
  return list[lo];
}

function lastStep() {
  return state.steps.length ? state.steps[state.steps.length - 1].step : null;
}

function seg(s) {
  return encodeURIComponent(s);
}

function runUrl(run) {
  return '/api/runs/' + seg(run);
}

function epUrl() {
  return runUrl(state.run) + '/' + seg(state.ep);
}

export function episodeInfo() {
  if (!state.ep) return null;
  const cands = [];
  if (state.summary && state.summary.episode && state.summary.episode.name === state.ep) {
    cands.push(state.summary.episode);
  }
  if (state.runInfo && state.runInfo.episodes) {
    const e = state.runInfo.episodes.find(function (x) { return x.name === state.ep; });
    if (e) cands.push(e);
  }
  if (!cands.length) return null;
  // The one that knows how the episode ended wins; otherwise the first.
  const ended = cands.find(function (c) { return c.ended_by; });
  const entry = ended || cands[0];
  // Live only while no source says it ended.
  const live = !ended && cands.some(function (c) { return c.live; });
  return Object.assign({}, entry, { live: live });
}

function epLive() {
  const e = episodeInfo();
  return !!(e && e.live);
}

function runLive() {
  return !!(state.runInfo && state.runInfo.live);
}

// ---------------------------------------------------------------- network + notes

const notes = {};

function setNote(key, text) {
  if (text) notes[key] = text;
  else delete notes[key];
  renderNote();
}

function clearNotes(keys) {
  keys.forEach(function (k) { delete notes[k]; });
  renderNote();
}

// Resolves to the parsed JSON, or null (and a grey note) on any failure. Never rejects.
async function api(key, path) {
  let resp;
  try {
    resp = await fetch(path, { cache: 'no-store' });
  } catch (e) {
    setNote(key, 'server unreachable');
    return null;
  }
  if (resp.status === 404) {
    setNote(key, 'no data yet');
    return null;
  }
  if (!resp.ok) {
    setNote(key, 'server error ' + resp.status);
    return null;
  }
  try {
    const data = await resp.json();
    setNote(key, null);
    return data;
  } catch (e) {
    setNote(key, 'unreadable response');
    return null;
  }
}

// ---------------------------------------------------------------- loading

let runGen = 0; // bumped when the selected run changes
let epGen = 0; // bumped when the selected episode changes
let runTimer = null;
let epTimer = null;
let listTimer = null;
let playTimer = null;
let stepsAge = null; // {age: seconds since steps.jsonl was written (server clock), at: Date.now() of that reply}

async function ensureLabels() {
  if (state.labels.length) return;
  const rows = await api('labels', '/api/labels');
  if (Array.isArray(rows) && rows.length) {
    state.labels = rows;
    emit('labels');
  }
}

async function loadRuns() {
  const data = await api('runs', '/api/runs');
  if (!data || !Array.isArray(data.runs)) {
    renderPickers();
    return;
  }
  state.runs = data.runs;
  emit('run');
  await followRuns();
}

function startedAt(r) {
  if (typeof r.started === 'number') return r.started;
  return typeof r.mtime === 'number' ? r.mtime : 0;
}

// The run the page follows: the most recently started live run, preferring runs written to in the last FRESH_S
// seconds (a killed client leaves its run "live" for hours; the client is also silent at the start gate).
export function followTarget() {
  const live = state.runs.filter(function (r) { return r.live; });
  if (!live.length) return null;
  const fresh = live.filter(isFresh);
  const pool = fresh.length ? fresh : live;
  return pool.reduce(function (best, r) { return startedAt(r) > startedAt(best) ? r : best; });
}

function listEntry(name) {
  return state.runs.find(function (r) { return r.name === name; }) || null;
}

function isFresh(r) {
  return typeof r.age_s !== 'number' || r.age_s <= FRESH_S;
}

// After a run list refresh: follow a newer live run, or catch up with the selected run's liveness.
async function followRuns() {
  const target = followTarget();
  const cur = state.run ? listEntry(state.run) : null;
  // A stale "live" run (a killed client) that started before the selected one is not worth leaving it for.
  const worth = target && (!cur || isFresh(target) || startedAt(target) > startedAt(cur));
  if (state.follow && worth && target.name !== state.run) {
    await selectRun(target.name, { follow: true });
    return;
  }
  // The selected run went live again (`run --run-name` reused its folder) or ended while its own poll was off.
  if (cur && state.runInfo && !!cur.live !== runLive()) await refreshRun();
}

function resetEpisodeData() {
  state.ep = null;
  state.steps = [];
  state.chunks = [];
  state.summary = null;
  state.cursor = 0;
  stepsAge = null;
}

// opts: {ep, step, follow}. follow true/false sets it; omitted, the page follows the run iff it is live (and no
// step was asked for).
async function selectRun(name, opts) {
  opts = opts || {};
  const gen = ++runGen;
  ++epGen;
  stopPlay();
  state.run = name || null;
  state.runInfo = null;
  resetEpisodeData();
  state.follow = opts.follow != null ? !!opts.follow : false;
  clearNotes(['run', 'steps', 'chunks', 'summary']);
  emit('selection');
  if (!state.run) {
    schedule();
    writeHash();
    return;
  }
  const info = await api('run', runUrl(name));
  if (gen !== runGen) return;
  if (info) {
    state.runInfo = info;
    if (opts.follow == null) state.follow = opts.step == null && !!info.live;
    emit('run');
    const eps = info.episodes || [];
    if (eps.length) {
      const wanted = eps.find(function (e) { return e.name === opts.ep; });
      await loadEpisode(wanted ? wanted.name : eps[eps.length - 1].name, opts.step);
    } else {
      loadRuns(); // refresh the list: the check-obs fallback needs it
    }
  }
  if (gen !== runGen) return;
  schedule();
  writeHash();
}

async function loadEpisode(name, startStep) {
  const gen = ++epGen;
  stopPlay();
  state.ep = name;
  state.steps = [];
  state.chunks = [];
  state.summary = null;
  state.cursor = 0;
  stepsAge = null;
  clearNotes(['steps', 'chunks', 'summary']);
  emit('selection');
  const base = epUrl();
  const got = await Promise.all([
    api('steps', base + '/steps?from=0'),
    api('chunks', base + '/chunks?from=0'),
    api('summary', base + '/summary'),
  ]);
  if (gen !== epGen) return;
  applySteps(got[0]);
  applyChunks(got[1]);
  if (got[2]) state.summary = got[2];
  noteStepsAge(got[0]);
  const last = lastStep();
  if (last !== null) {
    if (startStep != null && !isNaN(startStep)) state.cursor = clampStep(startStep);
    else if (state.follow) state.cursor = last;
    else state.cursor = state.steps[0].step;
  }
  emit('data');
  schedule();
  writeHash();
}

function applySteps(resp) {
  if (!resp || !Array.isArray(resp.steps)) return 0;
  let added = 0;
  let last = lastStep();
  resp.steps.forEach(function (rec) {
    if (last === null || rec.step > last) {
      state.steps.push(rec);
      last = rec.step;
      added += 1;
    }
  });
  return added;
}

function noteStepsAge(resp) {
  if (resp && typeof resp.age_s === 'number') stepsAge = { age: resp.age_s, at: Date.now() };
}

// Seconds since the newest step was written, while following a live episode that has steps; otherwise null.
function liveStepAge() {
  if (!state.follow || !stepsAge || !state.steps.length || !runLive() || !epLive()) return null;
  return stepsAge.age + (Date.now() - stepsAge.at) / 1000;
}

function applyChunks(resp) {
  if (!resp || !Array.isArray(resp.chunks)) return 0;
  let added = 0;
  let last = state.chunks.length ? state.chunks[state.chunks.length - 1].k : -1;
  resp.chunks.forEach(function (c) {
    if (c.k > last) {
      state.chunks.push(c);
      last = c.k;
      added += 1;
    }
  });
  return added;
}

async function refreshRun() {
  const gen = runGen;
  const name = state.run;
  if (!name) return;
  const info = await api('run', runUrl(name));
  if (gen !== runGen || !info) return;
  const hadInfo = !!state.runInfo;
  const wasLive = runLive();
  state.runInfo = info;
  if (!hadInfo) state.follow = !!info.live;
  const eps = info.episodes || [];
  const newest = eps.length ? eps[eps.length - 1].name : null;
  const jump = newest && newest !== state.ep && (!state.ep || (state.follow && info.live));
  if (jump) {
    emit('run');
    await loadEpisode(newest);
    return;
  }
  emit('run');
  if (wasLive && !info.live && state.ep) await pollEpisode(false); // the run just ended: catch up once
}

async function pollEpisode(again) {
  const gen = epGen;
  if (!state.ep) return;
  const wasLive = epLive();
  const last = lastStep();
  const lastK = state.chunks.length ? state.chunks[state.chunks.length - 1].k : -1;
  const base = epUrl();
  // The catch-up read (again === false) is the last one for this episode: `fresh` stops the server from answering
  // with the alignment of the poll just before it, which would drop the final records.
  const fresh = again === false ? '&fresh=1' : '';
  const got = await Promise.all([
    api('steps', base + '/steps?from=' + (last === null ? 0 : last + 1) + fresh),
    api('chunks', base + '/chunks?from=' + (lastK + 1)),
    api('summary', base + '/summary' + (fresh ? '?fresh=1' : '')),
  ]);
  if (gen !== epGen) return;
  applySteps(got[0]);
  applyChunks(got[1]);
  if (got[2]) state.summary = got[2];
  noteStepsAge(got[0]);
  const newest = lastStep();
  if (state.follow && newest !== null) state.cursor = newest;
  emit('data');
  // The episode ended during this poll: one more read so the final records are not missed.
  if (again !== false && wasLive && !epLive()) await pollEpisode(false);
}

// ---------------------------------------------------------------- timers

function wantRunPoll() {
  return !!state.run && (!state.runInfo || (state.follow && runLive()));
}

function wantEpisodePoll() {
  return !!state.run && !!state.ep && state.follow && runLive() && epLive();
}

function schedule() {
  const run = wantRunPoll();
  if (run && !runTimer) runTimer = setTimeout(runTick, RUN_POLL_MS);
  if (!run && runTimer) { clearTimeout(runTimer); runTimer = null; }

  const ep = wantEpisodePoll();
  if (ep && !epTimer) epTimer = setTimeout(episodeTick, EPISODE_POLL_MS);
  if (!ep && epTimer) { clearTimeout(epTimer); epTimer = null; }

  if (!listTimer) listTimer = setTimeout(listTick, LIST_POLL_MS);
}

async function runTick() {
  runTimer = null;
  try {
    await ensureLabels();
    await refreshRun();
  } catch (e) {
    console.error('run poll failed', e);
  }
  schedule();
}

async function episodeTick() {
  epTimer = null;
  try {
    await pollEpisode();
  } catch (e) {
    console.error('episode poll failed', e);
  }
  schedule();
}

async function listTick() {
  listTimer = null;
  try {
    await ensureLabels();
    await loadRuns();
  } catch (e) {
    console.error('run list poll failed', e);
  }
  schedule();
}

// ---------------------------------------------------------------- cursor, follow, replay

function clampStep(step) {
  const first = state.steps[0].step;
  const last = lastStep();
  return Math.max(first, Math.min(last, Math.round(step)));
}

export function setCursor(step) {
  if (!state.steps.length || isNaN(step)) return;
  const target = clampStep(step);
  const left = state.follow && target !== lastStep();
  if (target === state.cursor && !left) return;
  state.cursor = target;
  if (left) {
    state.follow = false;
    emit('follow');
    schedule();
  }
  emit('cursor');
  if (!state.playing) writeHash(); // not 20 times a second during replay; stopPlay() writes it
}

export function setFollow(on) {
  on = !!on;
  if (on === state.follow) return;
  state.follow = on;
  if (on) {
    stopPlay();
    const last = lastStep();
    if (last !== null) state.cursor = last;
    // Catch up right away (a newer live run, or this run's newest episode), then the timers keep it current.
    const gen = runGen;
    followRuns().then(function () {
      if (gen !== runGen || !state.run) return null; // switched to another run
      return refreshRun().then(function () { return runLive() ? pollEpisode() : null; });
    }).then(schedule).catch(function (e) {
      console.error('follow catch-up failed', e);
    });
  }
  emit('follow');
  if (on) emit('cursor');
  schedule();
  writeHash();
}

export function setSelectedDim(dim) {
  if (state.selectedDim === dim) return;
  state.selectedDim = dim;
  emit('dim');
}

function stopPlay() {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
  }
  if (state.playing) {
    state.playing = false;
    renderCursorBar();
    writeHash();
  }
}

function togglePlay() {
  if (state.playing) {
    stopPlay();
    return;
  }
  if (state.steps.length < 2) return;
  if (state.cursor >= lastStep()) {
    if (runLive() && state.follow) return;
    setCursor(state.steps[0].step);
  }
  state.playing = true;
  playTimer = setInterval(function () {
    if (state.cursor >= lastStep()) {
      stopPlay();
      return;
    }
    setCursor(state.cursor + 1);
  }, 1000 / PLAY_STEPS_PER_S);
  renderCursorBar();
}

// ---------------------------------------------------------------- URL hash

function writeHash() {
  const parts = [];
  if (state.run) parts.push('run=' + seg(state.run));
  if (state.run && state.ep) parts.push('ep=' + seg(state.ep));
  if (state.ep && !state.follow && state.steps.length) parts.push('step=' + state.cursor);
  const hash = parts.length ? '#' + parts.join('&') : '';
  if (hash === location.hash || (!hash && !location.hash)) return;
  try {
    history.replaceState(null, '', hash || location.pathname + location.search);
  } catch (e) {
    // Hash sync is a convenience only.
  }
}

function parseHash() {
  const out = { run: null, ep: null, step: null };
  const text = location.hash.replace(/^#/, '');
  if (!text) return out;
  text.split('&').forEach(function (pair) {
    const i = pair.indexOf('=');
    if (i < 0) return;
    let key = pair.slice(0, i);
    let val = pair.slice(i + 1);
    try {
      val = decodeURIComponent(val);
    } catch (e) {
      return;
    }
    if (key === 'run' && val) out.run = val;
    else if (key === 'ep' && val) out.ep = val;
    else if (key === 'step' && /^\d{1,9}$/.test(val)) out.step = parseInt(val, 10);
  });
  return out;
}

function applyHash() {
  const h = parseHash();
  if (!h.run) {
    if (state.run) selectRun(null, { follow: true }).then(loadRuns);
    return;
  }
  if (h.run !== state.run) {
    // A link without #step was written while following: keep following (a newer live run wins).
    selectRun(h.run, { ep: h.ep, step: h.step, follow: h.step == null }).then(loadRuns);
    return;
  }
  if (h.ep && h.ep !== state.ep) {
    state.follow = false;
    loadEpisode(h.ep, h.step);
    return;
  }
  if (h.step !== null && h.step !== state.cursor) setCursor(h.step);
}

// ---------------------------------------------------------------- rendering

const $ = function (id) { return document.getElementById(id); };
let els = null;
let pickerSig = { run: '', ep: '' };
let eventsSig = '';

function setText(el, text) {
  if (el.textContent !== text) el.textContent = text;
}

function render() {
  if (!els) return;
  renderBanners(); // before the pickers: the follow note defers to the live banner
  renderPickers();
  renderStatus();
  renderEvents();
  renderImages();
  renderCursorBar();
  renderNote();
}

function renderNote() {
  if (!els) return;
  const texts = Object.keys(notes).map(function (k) { return notes[k]; });
  let text = '';
  if (texts.indexOf('server unreachable') >= 0) text = 'server unreachable, retrying';
  else if (texts.length) text = texts[0];
  setText(els.net, text);
}

function kindLabel(kind) {
  if (kind === 'episodes') return 'episode run';
  if (kind === 'check-obs') return 'check-only run';
  return kind || '';
}

function renderPickers() {
  const runs = state.runs.slice();
  if (state.run && !runs.some(function (r) { return r.name === state.run; })) {
    runs.push({ name: state.run, kind: state.runInfo ? 'episodes' : '', live: runLive() });
  }
  const runSig = JSON.stringify([state.run, runs.map(function (r) { return [r.name, r.kind, r.live]; })]);
  if (runSig !== pickerSig.run) {
    pickerSig.run = runSig;
    const sel = els.runSelect;
    sel.textContent = '';
    const blank = document.createElement('option');
    blank.value = '';
    blank.textContent = runs.length ? 'select a run' : 'no runs found';
    sel.appendChild(blank);
    runs.forEach(function (r) {
      const o = document.createElement('option');
      o.value = r.name;
      o.textContent = r.name + (r.kind === 'check-obs' ? ' (check-only)' : '') + (r.live ? ' ● live' : '');
      sel.appendChild(o);
    });
    sel.value = state.run || '';
  }
  const cur = runs.find(function (r) { return r.name === state.run; });
  setText(els.runKind, cur ? kindLabel(cur.kind) : '');

  const eps = (state.runInfo && state.runInfo.episodes) || [];
  const epSig = JSON.stringify([state.ep, eps.map(function (e) { return [e.name, e.live, e.ended_by]; })]);
  if (epSig !== pickerSig.ep) {
    pickerSig.ep = epSig;
    const sel = els.epSelect;
    sel.textContent = '';
    if (!eps.length) {
      const o = document.createElement('option');
      o.value = '';
      o.textContent = state.run ? 'no episode yet' : '';
      sel.appendChild(o);
    }
    eps.forEach(function (e) {
      const o = document.createElement('option');
      o.value = e.name;
      const tail = e.live ? ' ● live' : (e.ended_by ? ' · ' + e.ended_by : '');
      o.textContent = e.name + (e.dry_run ? ' (dry)' : '') + tail;
      sel.appendChild(o);
    });
    sel.value = state.ep || '';
    sel.disabled = !eps.length;
  }

  // Follow on a run that is not live (it ended, or it is still loading) is never shown lit: "run ended".
  const ended = !!state.run && state.follow && !!state.runInfo && !runLive();
  els.follow.setAttribute('aria-pressed', state.follow ? 'true' : 'false');
  els.follow.setAttribute('data-state', ended ? 'ended' : (state.follow && runLive() ? 'live' : 'off'));
  els.follow.disabled = !state.run;
  setText(els.followLabel, ended ? 'run ended' : 'LIVE follow');
  els.follow.title = ended
    ? 'This run has ended. Follow stays on: the next live run opens by itself. Click to turn follow off.'
    : 'Stick to the newest step of a live run';
  let note = '';
  if (ended) note = 'the next live run opens by itself';
  else if (state.run && runLive() && !state.follow && !els.liveBanner.textContent) note = 'live data paused while follow is off';
  setText(els.followNote, note);
  renderLiveAge();
}

function renderLiveAge() {
  const age = liveStepAge();
  const show = age !== null && age > STALE_STEP_S;
  els.liveAge.hidden = !show;
  if (!show) {
    setText(els.liveAge, '');
    return;
  }
  const n = Math.round(age);
  setText(els.liveAge, 'last step ' + (n < 120 ? n + ' s' : Math.round(n / 60) + ' min') + ' ago');
}

function config() {
  return state.runInfo && state.runInfo.config ? state.runInfo.config : null;
}

function renderStatus() {
  const idx = stepIndex(state.cursor);
  const rec = idx >= 0 ? state.steps[idx] : null;
  const cfg = config();
  const safety = cfg && cfg.safety ? cfg.safety : {};

  const mode = rec ? rec.mode : '';
  els.modeChip.setAttribute('data-mode', mode);
  setText(els.modeText, rec ? mode : '–');
  setText(els.modeReason, rec && mode !== 'run' && rec.reason ? rec.reason : '');
  setText(els.queue, rec ? String(rec.queue) : '–');
  setText(els.inflight, rec ? (rec.in_flight ? '● yes' : '○ no') : '–');
  let dry = '–';
  if (rec && typeof rec.dry_s === 'number') {
    dry = rec.dry_s.toFixed(2) + ' s';
    if (typeof safety.infer_dry_freeze_s === 'number') dry += ' / ' + safety.infer_dry_freeze_s + ' s';
  }
  setText(els.dry, dry);

  const server = state.runInfo && state.runInfo.server ? state.runInfo.server : {};
  const ckpt = server.checkpoint != null ? server.checkpoint : server.config;
  setText(els.ckpt, ckpt != null && ckpt !== '' ? String(ckpt) : '–');

  let sync = '–';
  if (cfg) sync = cfg.mode === 'async' ? 'async' : 'sync H=' + cfg.execute_horizon;
  setText(els.sync, sync);

  const info = episodeInfo();
  const badge = els.badge;
  if (info && info.ended_by === 'freeze') {
    badge.hidden = false;
    badge.className = 'badge badge-freeze';
    setText(badge, 'FREEZE: ' + (info.reason || 'no reason recorded'));
  } else if (info && info.ended_by === 'interrupted') {
    badge.hidden = false;
    badge.className = 'badge badge-interrupted';
    setText(badge, 'INTERRUPTED');
    badge.title = info.reason || '';
  } else {
    badge.hidden = true;
    setText(badge, '');
  }
}

function checkObsSource() {
  if (!state.run) return null;
  if (state.runInfo && state.runInfo.check_obs && state.runInfo.check_obs.length) return state.run;
  const other = state.runs.find(function (r) { return r.kind === 'check-obs' && r.name !== state.run; });
  return other ? other.name : null;
}

function renderBanners() {
  const info = episodeInfo();
  els.dryBanner.hidden = !(info && info.dry_run);

  // Follow off and some run is live (another one, or this one again): offer it instead of switching.
  const target = state.follow ? null : followTarget();
  const offer = target && (target.name !== state.run || !runLive()) ? target.name : '';
  els.liveBanner.hidden = !offer;
  els.liveBanner.setAttribute('data-run', offer);
  setText(els.liveBanner, offer ? 'run ' + offer + ' is live \u2014 click to follow' : '');

  let text = '';
  if (!state.run) {
    if (!state.runs.length) text = 'No run selected, and no runs found yet. This list refreshes every few seconds.';
    else if (state.follow) text = 'No run selected. Pick one above; a live run is selected automatically.';
    else text = 'No run selected. Pick one above.';
  } else if (!state.ep) {
    if (!state.runInfo) {
      text = 'Waiting for run data.';
    } else {
      const src = checkObsSource();
      text = 'Episode view starts after Enter at the start gate. ' +
        (src ? 'These are the check-obs snapshots from ' + src + '.' : 'No check-obs snapshots yet.');
    }
  }
  els.gateBanner.hidden = !text;
  setText(els.gateBanner, text);
}

function eventText(e) {
  const direct = ['message', 'reason', 'note', 'previous_error'];
  for (let i = 0; i < direct.length; i += 1) {
    const v = e[direct[i]];
    if (v != null && v !== '') return String(v);
  }
  if (e.kind === 'prepare') return 'ok=' + e.ok;
  const skip = { wall: 1, mono: 1, kind: 1, episode: 1, steps: 1 };
  return Object.keys(e)
    .filter(function (k) { return !skip[k] && (typeof e[k] !== 'object' || e[k] === null); })
    .map(function (k) { return k + '=' + e[k]; })
    .join(' ');
}

function eventTone(e) {
  if (e.kind === 'freeze_before_start' || e.kind === 'refused' || e.kind === 'release_errors') return 'ev-bad';
  if (e.kind === 'prepare' && e.ok === false) return 'ev-bad';
  if (e.kind === 'interrupted' || e.kind === 'worker_restarted' || e.kind === 'takeover') return 'ev-warn';
  return '';
}

function renderEvents() {
  const events = state.runInfo && Array.isArray(state.runInfo.run_events) ? state.runInfo.run_events : [];
  const sig = (state.run || '') + '|' + events.length + '|' + (events.length ? JSON.stringify(events[events.length - 1]) : '');
  if (sig === eventsSig) return;
  eventsSig = sig;
  const box = els.events;
  box.textContent = '';
  box.hidden = !events.length;
  events.forEach(function (e) {
    const row = document.createElement('div');
    row.className = 'ev ' + eventTone(e);
    const t = document.createElement('span');
    t.className = 'ev-time';
    t.textContent = typeof e.wall === 'number'
      ? new Date(e.wall * 1000).toLocaleTimeString([], { hour12: false })
      : '';
    const k = document.createElement('span');
    k.className = 'ev-kind';
    k.textContent = e.kind + (e.episode != null ? ' (ep ' + e.episode + ')' : '');
    const m = document.createElement('span');
    m.className = 'ev-msg';
    m.textContent = eventText(e);
    row.appendChild(t);
    row.appendChild(k);
    row.appendChild(m);
    box.appendChild(row);
  });
  box.scrollTop = box.scrollHeight;
}

function showImage(fig, url, caption, emptyText) {
  const img = fig.img;
  const ph = fig.placeholder;
  if (!url) {
    img.hidden = true;
    img.removeAttribute('src');
    img.removeAttribute('data-src');
    ph.hidden = false;
    setText(ph, emptyText);
  } else if (img.getAttribute('data-src') !== url) {
    img.setAttribute('data-src', url);
    img.hidden = false;
    ph.hidden = true;
    img.src = url;
  }
  setText(fig.cap, caption);
}

function renderImages() {
  CAMERAS.forEach(function (cam) {
    const fig = els.cams[cam];
    if (state.ep) {
      const list = state.summary && state.summary.images ? state.summary.images[cam] : null;
      const s = floorIn(list, state.cursor);
      if (s === null) {
        showImage(fig, null, '', state.summary ? 'no frame at or before cursor' : 'no frame yet');
      } else {
        const d = state.cursor - s;
        showImage(
          fig,
          '/img/' + seg(state.run) + '/' + seg(state.ep) + '/' + seg(cam) + '/' + s,
          'frame @ step ' + s + ' (cursor ' + state.cursor + ', ' + (d ? MINUS + d : '0') + ' steps)',
          ''
        );
      }
    } else {
      const src = checkObsSource();
      showImage(
        fig,
        src ? '/img/' + seg(src) + '/check_obs/' + seg(cam) : null,
        src ? 'check-obs snapshot' : '',
        'no snapshot yet'
      );
    }
  });
}

function renderCursorBar() {
  const n = state.steps.length;
  const scrub = els.scrub;
  scrub.disabled = n === 0;
  const max = Math.max(0, n - 1);
  if (scrub.max !== String(max)) scrub.max = String(max);
  const idx = Math.max(0, stepIndex(state.cursor));
  if (scrub.value !== String(idx)) scrub.value = String(idx);
  els.play.disabled = n < 2;
  setText(els.play, state.playing ? 'Pause' : 'Play');
  if (!n) {
    setText(els.readout, state.ep ? 'no steps yet' : 'no steps');
    return;
  }
  const rec = state.steps[idx];
  setText(
    els.readout,
    'step ' + state.cursor + ' / ' + lastStep() +
      (rec && typeof rec.t === 'number' ? '  ·  t ' + rec.t.toFixed(2) + ' s' : '')
  );
}

// ---------------------------------------------------------------- wiring

function onKey(e) {
  if (e.altKey || e.ctrlKey || e.metaKey) return;
  const t = e.target;
  const tag = t && t.tagName;
  if (tag === 'SELECT' || tag === 'TEXTAREA' || (tag === 'INPUT' && t.type !== 'range')) return;
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
    e.preventDefault();
    const d = (e.shiftKey ? BIG_STEP : 1) * (e.key === 'ArrowLeft' ? -1 : 1);
    setCursor(state.cursor + d);
  } else if (e.key === ' ' || e.code === 'Space') {
    if (tag === 'BUTTON') return;
    e.preventDefault();
    togglePlay();
  }
}

function init() {
  els = {
    runSelect: $('run-select'),
    runKind: $('run-kind'),
    epSelect: $('ep-select'),
    follow: $('follow'),
    followLabel: $('follow-label'),
    followNote: $('follow-note'),
    liveAge: $('live-age'),
    liveBanner: $('live-banner'),
    net: $('net-note'),
    modeChip: $('mode-chip'),
    modeText: $('mode-text'),
    modeReason: $('mode-reason'),
    queue: $('st-queue'),
    inflight: $('st-inflight'),
    dry: $('st-dry'),
    ckpt: $('st-ckpt'),
    sync: $('st-sync'),
    badge: $('badge'),
    dryBanner: $('dry-banner'),
    gateBanner: $('gate-banner'),
    events: $('run-events'),
    scrub: $('scrub'),
    play: $('play'),
    readout: $('cursor-readout'),
    cams: {},
  };
  CAMERAS.forEach(function (cam) {
    const figEl = document.querySelector('.cam[data-camera="' + cam + '"]');
    const fig = {
      img: figEl.querySelector('img'),
      placeholder: figEl.querySelector('.placeholder'),
      cap: figEl.querySelector('.cap'),
    };
    fig.img.addEventListener('error', function () {
      // Mid-write or missing frame: show the placeholder, and let the next render try the URL again.
      fig.img.hidden = true;
      fig.img.removeAttribute('data-src');
      fig.placeholder.hidden = false;
      setText(fig.placeholder, 'image not available');
    });
    els.cams[cam] = fig;
  });

  els.runSelect.addEventListener('change', function () {
    const name = els.runSelect.value || null;
    // A picked run is followed iff it is live; "select a run" stops following (a live run is offered instead).
    selectRun(name, name ? {} : { follow: false });
    els.runSelect.blur(); // a focused select swallows the step keys and changes its option on arrow keys
  });
  els.liveBanner.addEventListener('click', function () {
    const name = els.liveBanner.getAttribute('data-run');
    els.liveBanner.blur();
    if (!name) return;
    if (name === state.run) setFollow(true);
    else selectRun(name, { follow: true });
  });
  els.epSelect.addEventListener('change', function () {
    const name = els.epSelect.value;
    els.epSelect.blur(); // see the run picker
    if (!name || name === state.ep) return;
    const eps = (state.runInfo && state.runInfo.episodes) || [];
    if (!eps.length || eps[eps.length - 1].name !== name) state.follow = false;
    loadEpisode(name);
  });
  els.follow.addEventListener('click', function () {
    setFollow(!state.follow);
    els.follow.blur();
  });
  els.play.addEventListener('click', function () {
    togglePlay();
    els.play.blur();
  });
  els.scrub.addEventListener('input', function () {
    const rec = state.steps[parseInt(els.scrub.value, 10)];
    if (rec) setCursor(rec.step);
  });
  document.addEventListener('keydown', onKey);
  window.addEventListener('hashchange', applyHash);

  render();
  start();
}

async function start() {
  try {
    await ensureLabels();
    const h = parseHash();
    if (h.run) {
      // A link without #step was written while following: keep following (a newer live run wins).
      await selectRun(h.run, { ep: h.ep, step: h.step, follow: h.step == null });
    }
    await loadRuns();
  } catch (e) {
    console.error('start failed', e);
  }
  schedule();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();
