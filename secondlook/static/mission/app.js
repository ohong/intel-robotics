// Mission control: top bar from /api/status, twin driven by LIVE or REPLAY joints. Read-only.
import { createTwin } from './twin.js';
import { startReplay } from './replay.js';
import { renderSee, clearSee } from './see.js';
import { renderPipeline, streamThought } from './pipeline.js';
const el = id => document.getElementById(id);
let startedAt = null;

function chip(id, text, tone) {
  const node = el(id);
  node.querySelector('span').textContent = text;
  node.className = `chip ${tone}`;
}

function renderStatus(s) {
  startedAt = Date.parse(s.started_at_utc);
  el('revision').textContent = `${s.mode} · rev ${String(s.revision).slice(0, 7)}`;
  chip('chip-camera', s.camera.status, s.camera.status === 'LIVE' ? 'ok' : 'bad');
  const detector = s.detector;
  const detectorText = detector.status === 'BLOCKED' ? 'BLOCKED'
    : `${(detector.execution_devices || [detector.device]).join('+')} ${detector.status}`;
  chip('chip-detector', detectorText, detector.status === 'FRESH' ? 'ok' : detector.status === 'BLOCKED' ? 'bad' : 'warn');
  chip('chip-policy', s.policy.status, s.policy.status === 'BLOCKED' ? 'warn' : 'ok');
  chip('chip-controller', s.controller.status, 'warn');
  chip('chip-evidence', `${s.evidence.records_this_session} ${s.evidence.status === 'ERROR' ? 'ERROR' : 'obs'}`,
       s.evidence.status === 'ERROR' ? 'bad' : 'ok');
  el('chip-policy').title = s.policy.reason;
  el('chip-controller').title = s.controller.reason;
}

function tickClock() {
  if (!Number.isFinite(startedAt)) return;
  const seconds = Math.max(0, Math.floor((Date.now() - startedAt) / 1000));
  const pad = n => String(n).padStart(2, '0');
  el('clock').textContent = `T+${pad(Math.floor(seconds / 3600))}:${pad(Math.floor(seconds / 60) % 60)}:${pad(seconds % 60)}`;
}

async function poll() {
  try {
    const response = await fetch('/api/status', {cache: 'no-store', signal: AbortSignal.timeout(1800)});
    if (!response.ok) throw new Error(`status HTTP ${response.status}`);
    const status = await response.json();
    renderStatus(status);
    renderSee(status);
    renderPipeline(status);
    el('connection').textContent = `Local runtime connected · ${new Date().toLocaleTimeString()}`;
  } catch (error) {
    el('connection').textContent = `Local runtime unreachable: ${error.message}`;
    clearSee(`Local runtime unreachable: ${error.message}`);
    for (const id of ['chip-camera', 'chip-detector', 'chip-policy', 'chip-controller', 'chip-evidence']) chip(id, '—', '');
  }
}

poll();
setInterval(poll, 1000);
setInterval(tickClock, 250);

(async () => {
  let twin;
  try {
    twin = await createTwin(document.getElementById('twin'));
  } catch (error) {
    document.getElementById('twin').innerHTML = '<p class="empty"></p>';
    document.querySelector('#twin .empty').textContent = `Digital twin unavailable: ${error.message}`;
    return;
  }
  try {
    const replay = await startReplay(twin);
    if (!replay) document.getElementById('act-caption').textContent = 'No live arm feed and no replay dataset configured';
    else streamThought(`Twin replaying ${replay.episodes.length} recorded episodes from ${replay.dataset}.`, {source: 'replay'});
  } catch (error) {
    document.getElementById('act-caption').textContent = `Replay unavailable: ${error.message}`;
  }
})();
