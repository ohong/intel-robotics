// Mission control: top bar from /api/status, twin driven by LIVE or REPLAY joints. Read-only.
import { createTwin } from './twin.js';
import { startReplay } from './replay.js';
import { startRobot } from './robot.js';
import { createJointBars } from './joints.js';
import { renderSee, clearSee } from './see.js';
import { renderPipeline, streamThought } from './pipeline.js';
import { startTimeline } from './timeline.js';
import { startAccel } from './accel.js';
import { startLive } from './live.js';
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
startTimeline();
startAccel();

const JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'];
let pose = null;  // last pose drawn on the twin, reported to GPT-Live with the ACT badge as its source
startLive({getPose: () => {
  const source = el('act-badge').textContent;
  if (!pose || source === 'NO FEED') return {status: 'UNAVAILABLE', source, detail: el('act-caption').textContent};
  return {status: source, detail: el('act-caption').textContent, units: 'degrees; gripper 0-100', ...pose};
}}).catch(error => streamThought(`GPT-Live unavailable: ${error.message}`, {source: 'voice', tone: 'warn'}));

(async () => {
  let twin;
  try {
    twin = await createTwin(document.getElementById('twin'));
  } catch (error) {
    document.getElementById('twin').innerHTML = '<p class="empty"></p>';
    document.querySelector('#twin .empty').textContent = `Digital twin unavailable: ${error.message}`;
    return;
  }
  const drawJoints = createJointBars(twin, document.getElementById('joints'), JOINTS);
  const updateJoints = (measured, target) => {
    const named = values => values ? Object.fromEntries(JOINTS.map((name, i) => [name, Math.round(values[i] * 10) / 10])) : null;
    pose = {measured: named(measured), target: named(target)};
    drawJoints(measured, target);
  };
  let replay = null;
  try {
    replay = await startReplay(twin, updateJoints);
    if (replay) streamThought(`Twin replaying ${replay.summary.episodes.length} recorded episodes from ${replay.summary.dataset}.`, {source: 'replay'});
  } catch (error) {
    document.getElementById('act-caption').textContent = `Replay unavailable: ${error.message}`;
  }
  let wasLive = false;
  const onLive = (live, message) => {
    if (live !== wasLive) {
      streamThought(live ? 'Studio joint telemetry is live; the twin now follows the real arm.'
        : `Studio joint telemetry stopped: ${message || 'no fresh packet'}.`, {source: 'robot', tone: live ? '' : 'warn'});
      wasLive = live;
    }
    replay?.suspend(live);
    if (replay) return;
    // No replay to fall back to: dim the last pose and say it is not current.
    document.getElementById('twin').classList.toggle('stale', !live);
    if (!live) {
      el('act-badge').textContent = 'NO FEED';
      el('act-badge').className = 'badge';
      el('act-caption').textContent = message || 'No fresh Studio joint telemetry';
    }
  };
  let robot = false;
  try {
    robot = await startRobot(twin, updateJoints, {onLive});
  } catch (error) {
    streamThought(`Robot telemetry unavailable: ${error.message}`, {source: 'robot', tone: 'warn'});
  }
  for (const id of ['joints', 'legend']) el(id).hidden = !(replay || robot);
  if (!replay && !robot) el('act-caption').textContent = 'No live arm feed and no replay dataset configured';
})();
