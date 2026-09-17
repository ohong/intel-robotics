// ASK THE ROBOT: GPT-Live full-duplex voice over WebRTC. The server exchanges the SDP offer
// so the OpenAI key never reaches the page. Tool calls are answered from this app's own
// read-only routes plus the allowlisted task instruction route; no tool can move the arm.
import { streamThought } from './pipeline.js';
const el = id => document.getElementById(id);
const SILENCE_HANGUP_MS = 90_000;
const SPEAKING_LEVEL = 0.04;
const MAX_TOOL_OUTPUT = 12_000;

const DETECTOR_FIELDS = ['status', 'reason', 'observation_id', 'score', 'threshold', 'disposition', 'decision',
  'backend', 'device', 'execution_devices', 'model_latency_ms', 'latency_ms', 'age_ms', 'evidence_kind'];

async function getJson(url, init) {
  const response = await fetch(url, {cache: 'no-store', ...init});
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

const pick = (object, keys) => Object.fromEntries(keys.filter(key => object?.[key] !== undefined).map(key => [key, object[key]]));

function toolHandlers(getPose) {
  return {
    async get_status() {
      const s = await getJson('/api/status');
      return {mode: s.mode, revision: s.revision, camera: pick(s.camera, ['status', 'transport', 'frame_age_ms', 'error']),
              detector: pick(s.detector, DETECTOR_FIELDS), model_validation: s.model_validation,
              task_instruction: s.task_instruction, policy: s.policy, controller: s.controller,
              evidence: pick(s.evidence, ['status', 'records_this_session', 'physical_trials', 'physical_outcome'])};
    },
    async get_recent_evidence({n = 5} = {}) {
      const count = Math.min(Math.max(Math.round(Number(n)) || 5, 1), 20);
      return getJson(`/api/evidence/recent?n=${count}`);
    },
    async get_arm_pose() {
      return getPose();
    },
    async set_task_instruction({phrase} = {}) {
      const result = await getJson('/api/task/instruction', {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({phrase: String(phrase ?? '')})});
      return {...result, controller: 'DISARMED', note: 'Selecting an instruction does not start or move the robot'};
    },
  };
}

function level(analyser, buffer) {
  analyser.getFloatTimeDomainData(buffer);
  let sum = 0;
  for (const sample of buffer) sum += sample * sample;
  return Math.sqrt(sum / buffer.length);
}

export async function startLive({getPose}) {
  const orb = el('orb'), label = el('orb-label');
  let status;
  try {
    status = await getJson('/api/live');
  } catch (error) {
    status = {status: 'UNAVAILABLE', error: error.message};
  }
  if (status.status !== 'READY') {
    orb.disabled = true;
    orb.dataset.state = 'unavailable';
    label.textContent = 'Voice unavailable';
    orb.title = status.error || 'GPT-Live unavailable';
    return;
  }
  orb.title = `${status.model} · ${status.scope}`;
  const tools = toolHandlers(getPose);
  let call = null;

  const setState = (state, text) => {
    orb.dataset.state = state;
    orb.setAttribute('aria-pressed', String(state !== 'idle'));
    label.textContent = text;
  };
  setState('idle', 'Ask the robot');

  async function answerTool(channel, item) {
    let output;
    const chip = streamThought(`${item.name}(${item.arguments || ''}) …`, {source: 'tool'});
    try {
      const handler = tools[item.name];
      if (!handler) throw new Error(`Unknown tool ${item.name}`);
      output = await handler(item.arguments ? JSON.parse(item.arguments) : {});
      const shown = JSON.stringify(output);
      chip.textContent = `${item.name} → ${shown.length > 160 ? `${shown.slice(0, 160)}…` : shown}`;
    } catch (error) {
      output = {error: error.message};
      chip.textContent = `${item.name} failed: ${error.message}`;
    }
    let text = JSON.stringify(output);
    if (text.length > MAX_TOOL_OUTPUT) text = JSON.stringify({truncated: true, partial: text.slice(0, MAX_TOOL_OUTPUT)});
    if (channel.readyState !== 'open') return;
    channel.send(JSON.stringify({type: 'response.item.create', event_id: crypto.randomUUID(),
      item: {type: 'function_call_output', call_id: item.call_id, output: text}}));
  }

  async function connect() {
    setState('connecting', 'Connecting…');
    const mic = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: true, noiseSuppression: true}});
    const pc = new RTCPeerConnection();
    const audio = new Audio();
    audio.autoplay = true;
    const context = new AudioContext();
    const analyser = context.createAnalyser();
    analyser.fftSize = 1024;
    const micAnalyser = context.createAnalyser();
    micAnalyser.fftSize = 1024;
    context.createMediaStreamSource(mic).connect(micAnalyser);
    const session = {pc, mic, audio, context, frame: 0, transcript: {kind: null, span: null}, pending: []};
    call = session;

    for (const track of mic.getTracks()) pc.addTrack(track, mic);
    pc.ontrack = event => {
      audio.srcObject = event.streams[0];
      context.createMediaStreamSource(event.streams[0]).connect(analyser);
    };
    const channel = pc.createDataChannel('oai-events');
    session.channel = channel;
    const started = new Promise((resolve, reject) => {
      session.onStarted = resolve;
      setTimeout(() => reject(new Error('GPT-Live session did not start within 15 s')), 15_000);
    });
    started.catch(() => {});  // awaited below; avoid an unhandled rejection if setup fails first

    channel.onmessage = async message => {
      const event = JSON.parse(message.data);
      if (event.type === 'session.started') return session.onStarted();
      if (event.type === 'session.closed') return hangUp('Session closed');
      if (event.type === 'error') {
        return streamThought(`GPT-Live error: ${event.error?.message || JSON.stringify(event).slice(0, 200)}`, {source: 'voice', tone: 'bad'});
      }
      if (event.type === 'session.input_transcript.delta' || event.type === 'session.output_transcript.delta') {
        const kind = event.type.includes('input') ? 'you' : 'robot';
        if (session.transcript.kind !== kind) {
          session.transcript = {kind, span: streamThought('', {source: kind === 'you' ? 'heard' : 'said'})};
        }
        session.transcript.span.textContent += event.delta ?? '';
        return;
      }
      const inner = event.type === 'response.event' ? event.event : null;
      if (inner?.type === 'response.output_item.done' && inner.item?.type === 'function_call') {
        session.transcript.kind = null;
        // Answer every call in this turn before asking the model to continue: only the handler
        // that sees no newer call after its wait sends response.create.
        session.pending.push(answerTool(channel, inner.item));
        const count = session.pending.length;
        await Promise.allSettled(session.pending);
        if (session.pending.length === count && count && channel.readyState === 'open') {
          session.pending = [];
          channel.send(JSON.stringify({type: 'response.create', event_id: crypto.randomUUID()}));
        }
      }
    };

    await pc.setLocalDescription(await pc.createOffer());
    await new Promise(resolve => {
      if (pc.iceGatheringState === 'complete') return resolve();
      const done = () => pc.iceGatheringState === 'complete' && resolve();
      pc.addEventListener('icegatheringstatechange', done);
      setTimeout(resolve, 2000);  // host candidates are enough on loopback-to-internet paths
    });
    const response = await fetch('/api/live/session', {
      method: 'POST', headers: {'Content-Type': 'application/sdp'}, body: pc.localDescription.sdp});
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || `HTTP ${response.status}`);
    await pc.setRemoteDescription({type: 'answer', sdp: await response.text()});
    await started;
    if (call !== session) return;

    setState('listening', 'Listening · tap to hang up');
    streamThought(`GPT-Live connected (${status.model}); it answers from read-only tools and cannot move the arm.`, {source: 'voice'});
    const remoteBuffer = new Float32Array(analyser.fftSize), micBuffer = new Float32Array(micAnalyser.fftSize);
    let lastSound = performance.now();
    const draw = () => {
      if (call !== session) return;
      const robot = level(analyser, remoteBuffer), you = level(micAnalyser, micBuffer);
      const now = performance.now();
      if (Math.max(robot, you) > SPEAKING_LEVEL) lastSound = now;
      if (now - lastSound > SILENCE_HANGUP_MS) return hangUp('Hung up after 90 s of silence');
      orb.style.setProperty('--level', Math.min(1, Math.max(robot, you) * 6).toFixed(3));
      const speaking = robot > SPEAKING_LEVEL ? 'speaking' : 'listening';
      if (orb.dataset.state !== speaking) setState(speaking, speaking === 'speaking' ? 'Speaking · tap to hang up' : 'Listening · tap to hang up');
      session.frame = requestAnimationFrame(draw);
    };
    draw();
  }

  function hangUp(reason) {
    const session = call;
    if (!session) return;
    call = null;
    cancelAnimationFrame(session.frame);
    try {
      if (session.channel?.readyState === 'open') session.channel.send(JSON.stringify({type: 'session.close'}));
    } catch { /* closing anyway */ }
    session.channel?.close();
    session.pc.close();
    session.mic.getTracks().forEach(track => track.stop());
    session.audio.srcObject = null;
    session.context.close();
    orb.style.setProperty('--level', '0');
    setState('idle', 'Ask the robot');
    if (reason) streamThought(`GPT-Live ended: ${reason}.`, {source: 'voice'});
  }

  orb.addEventListener('click', async () => {
    if (call) return hangUp('Hung up');
    try {
      await connect();
    } catch (error) {
      hangUp();  // releases the call if setup got as far as creating one
      streamThought(`GPT-Live could not connect: ${error.message}`, {source: 'voice', tone: 'bad'});
      setState('idle', 'Connection failed · tap to retry');
    }
  });
  addEventListener('pagehide', () => hangUp());
  return true;
}
