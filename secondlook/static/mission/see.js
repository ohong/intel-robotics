// SEE: latest camera frame, server-rendered anomaly overlay, score gauge, and a 60 s
// score history. Everything comes from /api/status and the two JPEG routes; when the
// camera or detector is not producing, the panel says so instead of holding old pixels.
const el = id => document.getElementById(id);
const HISTORY_MS = 60000;
const SVG = 'http://www.w3.org/2000/svg';
const history = [];  // {t, score, threshold, disposition, id}
let lastFrameId = null, lastOverlayId = null, loading = false;

const fmt = (value, digits = 2) => Number.isFinite(value) ? value.toFixed(digits) : '—';

async function loadImage(img, url) {
  const response = await fetch(url, {cache: 'no-store'});
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const blob = await response.blob();
  const previous = img.src;
  img.src = URL.createObjectURL(blob);
  if (previous.startsWith('blob:')) URL.revokeObjectURL(previous);
  return response.headers.get('X-Observation-Id');
}

function scaleMax(threshold) {
  const scores = history.map(sample => sample.score).filter(Number.isFinite);
  const peak = Math.max(0, ...scores, Number.isFinite(threshold) ? threshold : 0);
  return peak > 0 ? peak * 1.25 : 1;
}

function renderGauge(detector) {
  const score = detector.score, threshold = detector.threshold;
  const max = scaleMax(threshold);
  const arc = el('gauge-value');
  const length = arc.getTotalLength();
  const fraction = Number.isFinite(score) ? Math.min(1, Math.max(0, score / max)) : 0;
  arc.style.strokeDasharray = `${length}`;
  arc.style.strokeDashoffset = `${length * (1 - fraction)}`;
  const tick = el('gauge-threshold');
  tick.hidden = !Number.isFinite(threshold);
  if (Number.isFinite(threshold)) {
    const angle = Math.PI * (1 - Math.min(1, threshold / max));
    const [cx, cy, inner, outer] = [100, 100, 68, 94];
    tick.setAttribute('x1', cx + inner * Math.cos(angle));
    tick.setAttribute('y1', cy - inner * Math.sin(angle));
    tick.setAttribute('x2', cx + outer * Math.cos(angle));
    tick.setAttribute('y2', cy - outer * Math.sin(angle));
  }
  el('gauge-score').textContent = fmt(score);
  el('gauge-threshold-label').textContent = Number.isFinite(threshold) ? `threshold ${fmt(threshold)}` : 'threshold not validated';
  const disposition = detector.disposition || detector.status;
  el('see-disposition').textContent = disposition;
  el('see-disposition').className = `disposition ${String(disposition).toLowerCase()}`;
}

function renderSparkline(now) {
  while (history.length && now - history[0].t > HISTORY_MS) history.shift();
  const svg = el('sparkline');
  const width = 300, height = 64;
  const scores = history.filter(sample => Number.isFinite(sample.score));
  const threshold = history.at(-1)?.threshold;
  const max = scaleMax(threshold);
  const x = t => width - (now - t) / HISTORY_MS * width;
  const y = value => height - value / max * (height - 4) - 2;
  const line = el('spark-line'), limit = el('spark-threshold'), dots = el('spark-dots');
  line.setAttribute('points', scores.map(sample => `${x(sample.t).toFixed(1)},${y(sample.score).toFixed(1)}`).join(' '));
  limit.hidden = !Number.isFinite(threshold);
  if (Number.isFinite(threshold)) { limit.setAttribute('y1', y(threshold)); limit.setAttribute('y2', y(threshold)); }
  dots.replaceChildren(...scores.filter(sample => sample.disposition === 'ANOMALOUS').map(sample => {
    const dot = document.createElementNS(SVG, 'circle');
    Object.entries({cx: x(sample.t), cy: y(sample.score), r: 2.6}).forEach(([k, v]) => dot.setAttribute(k, v));
    return dot;
  }));
  svg.classList.toggle('no-data', scores.length === 0);
  el('spark-caption').textContent = scores.length ? `${scores.length} scored observations · last 60 s` : 'No scored observations in the last 60 s';
}

function setEmpty(message) {
  el('see-view').classList.add('is-empty');
  el('see-empty').textContent = message;
  lastFrameId = lastOverlayId = null;
}

export function renderSee(status) {
  const {camera, detector} = status;
  const now = Date.now();
  const live = camera.status === 'LIVE';
  const badge = el('see-badge');
  badge.textContent = live ? status.mode : 'NO FEED';
  badge.className = `badge ${live ? status.mode.toLowerCase() : ''}`;

  const fresh = detector.status === 'FRESH' && Number.isFinite(detector.score);
  if (fresh && detector.observation_id !== history.at(-1)?.id) {
    history.push({t: now, score: detector.score, threshold: detector.threshold, disposition: detector.disposition, id: detector.observation_id});
  }
  renderSparkline(now);
  el('see-metrics').hidden = !fresh;
  if (fresh) {
    renderGauge(detector);
    el('m-model').textContent = `${fmt(detector.model_latency_ms ?? detector.latency_ms, 1)} ms`;
    el('m-processing').textContent = `${fmt(detector.processing_ms, 1)} ms`;
    el('m-e2e').textContent = `${fmt(detector.observation_to_result_ms, 0)} ms`;
    el('m-age').textContent = `${fmt(detector.age_ms, 0)} ms`;
  } else {
    renderGauge({score: NaN, threshold: detector.threshold, disposition: detector.status});
  }
  el('see-reason').textContent = fresh ? (detector.decision_reason || detector.map_display || '')
    : detector.reason || (detector.status === 'STALE' ? 'Detector result is older than the freshness bound' : '');

  if (!live) return setEmpty(`No camera feed · ${camera.error || 'camera unavailable'}`);
  el('see-view').classList.remove('is-empty');
  if (loading) return;
  loading = true;
  const jobs = [];
  if (camera.observation_id !== lastFrameId) {
    jobs.push(loadImage(el('see-frame'), '/api/frame/latest.jpg').then(id => { lastFrameId = id; }));
  }
  if (fresh && detector.observation_id !== lastOverlayId) {
    jobs.push(loadImage(el('see-overlay'), `/api/detector.jpg?observation_id=${encodeURIComponent(detector.observation_id)}`)
      .then(id => { lastOverlayId = id; }));
  }
  el('see-overlay').classList.toggle('stale', !fresh);
  Promise.allSettled(jobs).finally(() => { loading = false; });
}

export function clearSee(message) {
  setEmpty(message);
  el('see-badge').textContent = 'NO FEED';
  el('see-badge').className = 'badge';
}

el('see-toggle').addEventListener('click', () => {
  const on = el('see-view').classList.toggle('show-overlay');
  el('see-toggle').textContent = on ? 'Heatmap on' : 'Heatmap off';
  el('see-toggle').setAttribute('aria-pressed', on);
});
