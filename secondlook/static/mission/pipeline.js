// THINK: the decision path each observation takes, drawn from /api/status. Node text is
// the runtime's own fields and reason strings; the thought stream records state changes
// as factual sentences built from those fields, never as model reasoning.
const el = id => document.getElementById(id);
const STREAM_LIMIT = 60;
let lastObservation = null;
let previous = {};

const fmt = (value, digits = 1) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const precision = value => value ? String(value).replace(/^<Type: '(.+)'>$/, '$1') : 'precision unreported';
const shortId = id => id ? String(id).slice(0, 8) : '—';

function node(id, tone, value, detail) {
  const item = el(id);
  item.dataset.tone = tone;
  item.querySelector('.node-value').textContent = value;
  item.querySelector('.node-detail').textContent = detail;
  item.title = detail;
}

export function streamThought(text, {source = 'runtime', tone = ''} = {}) {
  const list = el('thoughts');
  const entry = document.createElement('li');
  entry.className = `thought ${tone}`;
  const time = document.createElement('time');
  time.textContent = new Date().toLocaleTimeString([], {hour12: false});
  const tag = document.createElement('b');
  tag.textContent = source;
  const body = document.createElement('span');
  body.textContent = text;
  entry.append(time, tag, body);
  list.prepend(entry);
  while (list.children.length > STREAM_LIMIT) list.lastElementChild.remove();
  return body;  // callers streaming a transcript extend this text in place
}

function pulse() {
  const graph = el('pipeline');
  graph.classList.remove('pulse');
  void graph.offsetWidth;  // restart the CSS animation for this observation
  graph.classList.add('pulse');
}

export function renderPipeline(status) {
  const {camera, detector, policy, controller} = status;
  const live = camera.status === 'LIVE';
  node('node-camera', live ? 'ok' : 'bad', camera.status,
       live ? `${camera.transport} · frame age ${fmt(camera.frame_age_ms, 0)} ms` : camera.error || 'No fresh camera frame');

  const fresh = detector.status === 'FRESH';
  const devices = (detector.execution_devices || [detector.device]).filter(Boolean).join('+');
  node('node-detector', fresh ? 'ok' : detector.status === 'BLOCKED' ? 'bad' : 'warn',
       fresh ? `${fmt(detector.model_latency_ms ?? detector.latency_ms)} ms` : detector.status,
       fresh ? `${detector.backend} · ${devices} · ${precision(detector.inference_precision)}` : detector.reason || 'No detector result');

  const disposition = fresh ? detector.disposition : '—';
  node('node-decision', !fresh ? 'idle' : disposition === 'ANOMALOUS' ? 'bad' : disposition === 'NORMAL' ? 'ok' : 'warn',
       disposition,
       fresh ? `score ${fmt(detector.score, 2)} vs threshold ${Number.isFinite(detector.threshold) ? fmt(detector.threshold, 2) : 'not validated'}` : 'Waiting for a fresh detector result');
  node('node-policy', policy.status === 'BLOCKED' ? 'warn' : 'ok', policy.status, policy.reason || '');
  node('node-gate', 'warn', 'HOLD', controller.reason || '');
  node('node-arm', 'warn', controller.status, `controller backend ${controller.backend || 'UNAVAILABLE'} · no motion commands are sent from this page`);

  if (fresh && detector.observation_id !== lastObservation) {
    lastObservation = detector.observation_id;
    pulse();
    if (detector.disposition !== previous.disposition) {
      streamThought(`Observation ${shortId(detector.observation_id)} scored ${fmt(detector.score, 2)}` +
        `${Number.isFinite(detector.threshold) ? ` against threshold ${fmt(detector.threshold, 2)}` : ' with no validated threshold'}` +
        ` → ${detector.disposition} (${detector.backend} on ${devices}, ${fmt(detector.model_latency_ms ?? detector.latency_ms)} ms).`,
        {source: 'detector', tone: detector.disposition === 'ANOMALOUS' ? 'bad' : ''});
    }
  }
  const changes = [
    ['camera', camera.status, `Camera ${camera.status}${camera.error ? `: ${camera.error}` : ''}.`],
    ['detector', detector.status, `Detector ${detector.status}${detector.reason ? `: ${detector.reason}` : ''}.`],
    ['policy', policy.status, `Policy ${policy.status}: ${policy.reason}.`],
    ['controller', controller.status, `Controller ${controller.status}. Motion stays off.`],
    ['task', status.task_instruction, `Task instruction ${status.task_instruction ? `set to “${status.task_instruction}”` : 'not set'}.`],
  ];
  for (const [key, value, text] of changes) {
    if (previous[key] !== value) streamThought(text, {source: key});
  }
  previous = {camera: camera.status, detector: detector.status, policy: policy.status, controller: controller.status,
              task: status.task_instruction, disposition: fresh ? detector.disposition : previous.disposition};
}
