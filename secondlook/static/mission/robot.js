// LIVE driver: joint telemetry observed read-only from Studio's teleoperation session.
// While packets are fresh the twin follows them; when they stop, the page says so and hands
// the twin back to REPLAY (or dims it) instead of holding a pose that may no longer be true.
const el = id => document.getElementById(id);

export async function startRobot(twin, updateJoints, {onLive} = {}) {
  const probe = await fetch('/api/robot/latest', {cache: 'no-store'});
  if (probe.status === 404) return false;  // server started without --studio-robot-session
  let live = false;

  function setLive(next, message) {
    if (next !== live) {
      live = next;
    }
    onLive?.(live, message);
  }

  function render(packet) {
    if (packet.status !== 'LIVE') return setLive(false, packet.error);
    setLive(true);
    twin.setPose(packet.joint_names, packet.state);
    twin.setGhost(packet.joint_names, packet.action);
    updateJoints(packet.state, packet.action);
    el('act-badge').textContent = 'LIVE';
    el('act-badge').className = 'badge live';
    el('legend-target').textContent = packet.action ? 'leader action target' : 'no action target in tick';
    el('act-caption').textContent = `Studio ${packet.session} · joint telemetry · ${Math.round(packet.age_ms)} ms old`;
  }

  const stream = new EventSource('/api/robot/stream');
  stream.onmessage = event => render(JSON.parse(event.data));
  // EventSource reconnects by itself; until it does, no packet is fresh.
  stream.onerror = () => setLive(false, 'Robot telemetry stream disconnected');
  return true;
}
