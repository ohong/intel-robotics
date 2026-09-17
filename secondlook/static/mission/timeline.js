// EVIDENCE TIMELINE: the newest records from the append-only evidence log, one tick per
// observation. Clicking a tick shows that record's stored fields. Frames are not
// retained in the log, so the detail view shows fields only.
const el = id => document.getElementById(id);
const POLL_MS = 3000;
const LIMIT = 240;
const KIND_LABEL = {real: 'LIVE', replay: 'REPLAY', synthetic: 'SYNTHETIC', mock: 'MOCK'};
let records = [];
let selectedId = null;

const fmt = (value, digits = 2) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const clock = iso => new Date(iso).toLocaleTimeString([], {hour12: false});
const precision = value => value ? String(value).replace(/^<Type: '(.+)'>$/, '$1') : 'unreported';

function tone(record) {
  if (record.event === 'observation_invalid') return 'invalid';
  return String(record.detector?.disposition || 'unknown').toLowerCase();
}

function detail(record) {
  const d = record.detector || {};
  const rows = [
    ['recorded', `${clock(record.recorded_at_utc)} · ${record.evidence_kind}`],
    ['observation', String(record.observation_id).slice(0, 13)],
    ['event', record.event],
    ['disposition', record.event === 'observation_invalid' ? `INVALID · ${d.reason || (d.quality_flags || []).join(', ')}` : d.disposition],
    ['score', `${fmt(d.score)} vs ${Number.isFinite(d.threshold) ? fmt(d.threshold) : 'no validated threshold'}`],
    ['runtime', `${d.backend || '—'} · ${(d.execution_devices || [d.device]).filter(Boolean).join('+') || '—'} · ${precision(d.inference_precision)}`],
    ['model call', `${fmt(d.model_latency_ms ?? d.latency_ms, 1)} ms`],
    ['reason', d.decision_reason || '—'],
    ['policy', `${record.policy?.status || '—'}`],
    ['controller', record.controller_result],
    ['outcome', record.outcome],
  ];
  const list = document.createElement('dl');
  for (const [key, value] of rows) {
    const term = document.createElement('dt'), text = document.createElement('dd');
    term.textContent = key;
    text.textContent = value ?? '—';
    list.append(term, text);
  }
  el('timeline-detail').replaceChildren(list);
}

function render() {
  const strip = el('timeline-strip');
  const badge = el('timeline-badge');
  if (!records.length) {
    strip.replaceChildren();
    strip.dataset.empty = 'No evidence records in this session log yet';
    el('timeline-detail').textContent = 'Select an observation to see its stored record.';
    el('timeline-range').textContent = '';
    badge.textContent = 'EMPTY';
    badge.className = 'badge';
    return;
  }
  delete strip.dataset.empty;
  const kinds = [...new Set(records.map(record => record.evidence_kind))];
  badge.textContent = kinds.map(kind => KIND_LABEL[kind] || kind.toUpperCase()).join(' + ');
  badge.className = `badge ${kinds.length === 1 ? (KIND_LABEL[kinds[0]] || '').toLowerCase() : ''}`;

  const times = records.map(record => Date.parse(record.recorded_at_utc));
  const start = times[0], span = Math.max(60000, times.at(-1) - start);
  const scores = records.map(record => record.detector?.score).filter(Number.isFinite);
  const thresholds = records.map(record => record.detector?.threshold).filter(Number.isFinite);
  const max = Math.max(1, ...scores, ...thresholds) * 1.1;

  strip.style.setProperty('--tick', `${Math.max(2, Math.min(6, strip.clientWidth / records.length - 1))}px`);
  const ticks = records.map((record, index) => {
    const tick = document.createElement('button');
    tick.type = 'button';
    tick.className = `tick ${tone(record)}`;
    const score = record.detector?.score;
    tick.style.left = `${(times[index] - start) / span * 100}%`;
    tick.style.height = Number.isFinite(score) ? `${Math.max(4, score / max * 100)}%` : '4px';
    tick.setAttribute('aria-label', `${clock(record.recorded_at_utc)} ${tone(record)} score ${fmt(score)}`);
    tick.classList.toggle('selected', record.observation_id === selectedId);
    tick.addEventListener('click', () => { selectedId = record.observation_id; render(); });
    return tick;
  });
  if (thresholds.length) {
    const line = document.createElement('div');
    line.className = 'threshold';
    line.style.bottom = `${thresholds.at(-1) / max * 100}%`;
    ticks.push(line);
  }
  strip.replaceChildren(...ticks);
  const selected = records.find(record => record.observation_id === selectedId) || records.at(-1);
  detail(selected);
  const counts = records.reduce((total, record) => ({...total, [tone(record)]: (total[tone(record)] || 0) + 1}), {});
  el('timeline-range').textContent = `${clock(records[0].recorded_at_utc)} – ${clock(records.at(-1).recorded_at_utc)} · ` +
    Object.entries(counts).map(([name, count]) => `${count} ${name}`).join(' · ');
}

async function poll() {
  try {
    const response = await fetch(`/api/evidence/recent?n=${LIMIT}`, {cache: 'no-store', signal: AbortSignal.timeout(2500)});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    records = (await response.json()).records;
    render();
  } catch (error) {
    records = [];
    render();
    el('timeline-strip').dataset.empty = `Evidence log unavailable: ${error.message}`;
  }
}

export function startTimeline() {
  poll();
  setInterval(poll, POLL_MS);
}
