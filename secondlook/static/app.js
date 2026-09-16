const el = id => document.getElementById(id);
const numeric = value => Number.isFinite(value) ? value.toFixed(1) : '—';
const urls = {};
function validationText(summary) {
  if (!summary || summary.status !== 'EXPERIMENTAL PILOT') {
    return summary?.reason || 'Model validation summary unavailable.';
  }
  const count = value => Number.isInteger(value) && value >= 0 ? value : 'not recorded';
  const limits = summary.split_limitations?.length ? summary.split_limitations.join(' ') : 'Split independence not documented.';
  const test = summary.unseen_block_test === 'PENDING' ? 'Unseen-block test pending.' : 'Evaluation recorded; unseen-block coverage not verified here.';
  return `EXPERIMENTAL PILOT · validation only. ${count(summary.validation_images)} images; ${count(summary.defect_images)} defect images. Missed defects: ${count(summary.missed_defects)}; rejected good parts: ${count(summary.rejected_good_parts)}; uncertain defects: ${count(summary.uncertain_defects)}; uncertain good parts: ${count(summary.uncertain_good_parts)}. ${test} ${limits} Repeated views are not independent trials.`;
}
function clearImage(id, message) {
  el(id).hidden = true;
  el(`${id}-empty`).hidden = false;
  el(`${id}-empty`).textContent = message;
  el(id).removeAttribute('src');
  delete el(id).dataset.observationId;
  if (urls[id]) URL.revokeObjectURL(urls[id]);
  delete urls[id];
}
async function refreshImage(id, path) {
  const response = await fetch(path, {cache: 'no-store', signal: AbortSignal.timeout(1800)});
  if (!response.ok) throw new Error('No fresh observation');
  const blob = await response.blob();
  if (urls[id]) URL.revokeObjectURL(urls[id]);
  urls[id] = URL.createObjectURL(blob);
  el(id).src = urls[id];
  el(id).hidden = false;
  el(`${id}-empty`).hidden = true;
  el(id).dataset.observationId = response.headers.get('X-Observation-Id') || '';
}
async function update() {
  try {
    const response = await fetch('/api/status', {cache: 'no-store', signal: AbortSignal.timeout(1800)});
    if (!response.ok) throw new Error(`Status HTTP ${response.status}`);
    const s = await response.json();
    el('mode').textContent = `${s.mode} · SOFTWARE DISARMED`;
    el('camera-state').textContent = s.camera.status;
    el('camera-meta').textContent = s.camera.status === 'LIVE'
      ? `${s.camera.device} · frame age ${numeric(s.camera.frame_age_ms)} ms · ${s.camera.timestamp_source}`
      : s.camera.error;
    el('detector-state').textContent = s.detector.status;
    const detectorReason = s.detector.decision_reason || s.detector.reason || 'No result';
    el('detector-meta').textContent = s.detector.score == null ? detectorReason
      : `${s.detector.disposition} · raw score ${s.detector.score.toFixed(4)} · threshold ${s.detector.threshold ?? 'not validated'} · ${s.detector.backend} / ${(s.detector.execution_devices || [s.detector.device]).join(', ')} / ${s.detector.inference_precision || 'precision unreported'} · ${s.detector.evidence_kind || 'unknown model provenance'} · ${s.detector.map_display}`;
    if (s.detector.observation_validity) {
      el('detector-meta').textContent += ` · observation ${s.detector.observation_validity}: presence ${s.detector.object_presence?.status || 'UNVERIFIED'}, occlusion ${s.detector.occlusion?.status || 'UNVERIFIED'}, defect visibility ${s.detector.defect_visibility?.status || 'UNVERIFIED'}`;
    }
    el('model-validation').textContent = validationText(s.model_validation);
    el('instruction').textContent = s.task_instruction || 'BLOCKED — Exact Challenge-Day task has not been supplied.';
    el('policy').textContent = `${s.policy.status} — ${s.policy.reason}`;
    el('evidence').textContent = `${s.evidence.status} · ${s.evidence.records_this_session} observation records this session. ${s.evidence.physical_trials} physical trials. Physical outcome: ${s.evidence.physical_outcome}.`;
    if(s.evidence.error) el('evidence').textContent += ` Write error: ${s.evidence.error}`;
    el('timing').textContent = `Model stage: ${numeric(s.detector.model_latency_ms)} ms · preprocessing + model: ${numeric(s.detector.latency_ms)} ms · observation to result: ${numeric(s.detector.observation_to_result_ms)} ms · result age: ${numeric(s.detector.age_ms)} ms`;
    el('revision').textContent = `Revision ${s.revision}`;
    renderVoice(s);
    await Promise.all([
      s.camera.status === 'LIVE' ? refreshImage('camera', '/api/frame/latest.jpg').catch(() => clearImage('camera', 'No fresh camera frame'))
        : clearImage('camera', s.camera.error || 'No fresh camera frame'),
      s.detector.status === 'FRESH' ? refreshImage('detector', `/api/detector.jpg?observation_id=${encodeURIComponent(s.detector.observation_id)}`).catch(() => clearImage('detector', 'Detector result changed or is stale; refreshing'))
        : clearImage('detector', detectorReason)
    ]);
    el('connection').textContent = `Local service connected · updated ${new Date().toLocaleTimeString()} · detector overlay uses its own observation`;
  } catch(error) {
    clearImage('camera', 'Connection unavailable — live image cleared');
    clearImage('detector', 'Connection unavailable — detector image cleared');
    el('mode').textContent = 'OFFLINE · SOFTWARE DISARMED';
    el('camera-state').textContent = 'UNAVAILABLE';
    el('camera-meta').textContent = 'Connection unavailable — current observation age is unknown.';
    el('detector-state').textContent = 'UNAVAILABLE';
    el('detector-meta').textContent = 'Connection unavailable — no current detector result.';
    el('model-validation').textContent = 'Connection unavailable — loaded model validation summary cannot be verified.';
    el('timing').textContent = 'Current processing times and result age are unavailable.';
    el('evidence').textContent = 'Connection unavailable — current evidence status and counts are unknown.';
    el('connection').textContent = `Local runtime unavailable: ${error.message}`;
  }
  setTimeout(update, 500);
}
update();

// Voice: speech sets an allowlisted task instruction; narration is read aloud one line at a time.
const voice = {recorder: null, chunks: [], lastSeq: null, queue: [], playing: false, audio: new Audio()};
function renderVoice(s) {
  const ready = s.voice?.status === 'READY';
  el('voice-state').textContent = s.voice?.status || 'UNAVAILABLE';
  el('talk').disabled = !ready;
  if (!ready) el('voice-result').textContent = s.voice?.reason || 'Voice unavailable.';
  const lines = s.narration || [];
  const newest = lines.length ? lines[lines.length - 1].seq : 0;
  if (voice.lastSeq === null) voice.lastSeq = newest;  // Do not replay the backlog on page load.
  el('narration').replaceChildren(...lines.slice(-12).reverse().map(line => {
    const item = document.createElement('li');
    item.textContent = `${new Date(line.at_utc).toLocaleTimeString()} · ${line.text}`;
    return item;
  }));
  for (const line of lines) {
    if (line.seq > voice.lastSeq && el('speak-toggle').checked && ready) voice.queue.push(line.text);
  }
  voice.lastSeq = Math.max(voice.lastSeq, newest);
  playNext();
}
async function playNext() {
  if (voice.playing || !voice.queue.length) return;
  voice.playing = true;
  const text = voice.queue.shift();
  try {
    const response = await fetch('/api/voice/speak', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text})});
    if (!response.ok) throw new Error((await response.json()).error);
    const url = URL.createObjectURL(await response.blob());
    voice.audio.src = url;
    await voice.audio.play();
    await new Promise(resolve => { voice.audio.onended = voice.audio.onerror = resolve; });
    URL.revokeObjectURL(url);
  } catch (error) {
    el('voice-result').textContent = `Narration audio failed: ${error.message}`;
  }
  voice.playing = false;
  playNext();
}
async function startTalking() {
  if (el('talk').disabled || voice.recorder) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    const mimeType = ['audio/webm', 'audio/mp4'].find(type => MediaRecorder.isTypeSupported(type));
    voice.recorder = new MediaRecorder(stream, mimeType ? {mimeType} : undefined);
    voice.chunks = [];
    voice.recorder.ondataavailable = event => voice.chunks.push(event.data);
    voice.recorder.onstop = () => { stream.getTracks().forEach(track => track.stop()); sendCommand(); };
    voice.recorder.start();
    el('talk').classList.add('recording');
    el('talk').textContent = 'Listening… release to send';
  } catch (error) {
    voice.recorder = null;
    el('voice-result').textContent = `Microphone unavailable: ${error.message}`;
  }
}
function stopTalking() {
  if (!voice.recorder) return;
  voice.recorder.stop();
  el('talk').classList.remove('recording');
  el('talk').textContent = 'Hold to talk';
}
async function sendCommand() {
  const type = voice.recorder.mimeType.split(';')[0] || 'audio/webm';
  voice.recorder = null;
  const blob = new Blob(voice.chunks, {type});
  el('voice-result').textContent = 'Transcribing…';
  try {
    const response = await fetch('/api/voice/command', {method: 'POST', headers: {'Content-Type': type}, body: blob});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    el('voice-result').textContent = result.matched
      ? `Heard “${result.transcript}” → task set (match ${result.match_score}).`
      : `Heard “${result.transcript}” → not a known task; instruction unchanged.`;
  } catch (error) {
    el('voice-result').textContent = `Voice command failed: ${error.message}`;
  }
}
el('talk').addEventListener('pointerdown', startTalking);
['pointerup', 'pointerleave', 'pointercancel'].forEach(name => el('talk').addEventListener(name, stopTalking));
document.addEventListener('keydown', event => { if (event.code === 'Space' && !event.repeat && event.target === document.body) { event.preventDefault(); startTalking(); } });
document.addEventListener('keyup', event => { if (event.code === 'Space') stopTalking(); });
