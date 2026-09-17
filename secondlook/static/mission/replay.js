// REPLAY driver: plays a recorded LeRobot episode through the twin, with the recorded
// camera videos slaved to the same clock. Everything it shows is labelled REPLAY.
const el = id => document.getElementById(id);
const VIDEO_DRIFT_S = .25;

export async function startReplay(twin, {onFrame} = {}) {
  const response = await fetch('/api/replay/episodes', {cache: 'no-store'});
  if (response.status === 404) return null;  // server started without --replay-dataset
  if (!response.ok) throw new Error(`replay HTTP ${response.status}`);
  const summary = await response.json();

  const state = {episode: null, time: 0, playing: false, speed: 1, lastTick: null};
  const videos = [];
  const joints = summary.joint_names.map(name => {
    const row = document.createElement('div');
    row.className = 'joint';
    row.innerHTML = `<span>${name}</span><div class="track"><i></i><b></b></div><output>—</output>`;
    el('joints').append(row);
    return {name, limits: twin.limitsDeg(name) || [-180, 180], fill: row.querySelector('i'),
            marker: row.querySelector('b'), value: row.querySelector('output')};
  });

  for (const episode of summary.episodes) {
    el('episode').add(new Option(`Episode ${episode.index} · ${episode.task}`, episode.index));
  }
  summary.cameras.forEach((camera, index) => {
    const tile = document.createElement('div');
    tile.className = 'cam';
    const video = Object.assign(document.createElement('video'), {muted: true, playsInline: true, preload: 'auto'});
    const label = document.createElement('span');
    label.textContent = `REPLAY · ${camera.replace('observation.images.', '').replace(/_/g, '').trim()}`;
    tile.append(video, label);
    el('cams').append(tile);
    videos[index] = video;
  });

  const percent = (joint, degrees) => {
    const [low, high] = joint.limits;
    return Math.min(100, Math.max(0, (degrees - low) / (high - low) * 100));
  };

  function render() {
    const episode = state.episode;
    const frame = Math.min(episode.length - 1, Math.floor(state.time * summary.fps));
    const measured = episode.state[frame], target = episode.action[frame];
    twin.setPose(summary.joint_names, measured);
    twin.setGhost(summary.joint_names, target);
    joints.forEach((joint, index) => {
      const zero = percent(joint, 0), now = percent(joint, measured[index]);
      joint.fill.style.left = `${Math.min(zero, now)}%`;
      joint.fill.style.width = `${Math.abs(now - zero)}%`;
      joint.marker.style.left = `${percent(joint, target[index])}%`;
      joint.value.textContent = `${measured[index].toFixed(1)}°`;
    });
    el('scrub').value = frame;
    el('replay-time').textContent = `${state.time.toFixed(1)} / ${episode.duration_s.toFixed(1)} s · f${frame}`;
    onFrame?.({source: 'REPLAY', episode: episode.index, task: episode.task, frame, fps: summary.fps,
               joint_names: summary.joint_names, state: measured, action: target});
  }

  function syncVideos(force) {
    state.episode.videos.forEach((clip, index) => {
      const video = videos[index];
      if (!video) return;
      const wanted = clip.from_timestamp + state.time;
      if (force || Math.abs(video.currentTime - wanted) > VIDEO_DRIFT_S) video.currentTime = wanted;
      video.playbackRate = state.speed;
      if (state.playing && video.paused) video.play().catch(() => {});
      if (!state.playing && !video.paused) video.pause();
    });
  }

  async function load(index) {
    const reply = await fetch(`/api/replay/episodes/${index}`, {cache: 'no-store'});
    if (!reply.ok) throw new Error(`episode ${index} HTTP ${reply.status}`);
    state.episode = await reply.json();
    state.time = 0;
    el('scrub').max = state.episode.length - 1;
    el('act-caption').textContent = `${summary.dataset} · episode ${index} · “${state.episode.task}”`;
    state.episode.videos.forEach((clip, i) => {
      if (videos[i] && !videos[i].src.endsWith(clip.url)) videos[i].src = clip.url;
    });
    syncVideos(true);
    render();
  }

  function setPlaying(playing) {
    state.playing = playing;
    state.lastTick = null;
    el('play').textContent = playing ? '❚❚' : '▶';
    el('play').setAttribute('aria-label', playing ? 'Pause' : 'Play');
    syncVideos(true);
  }

  function tick(now) {
    if (state.playing && state.episode) {
      if (state.lastTick !== null) state.time += (now - state.lastTick) / 1000 * state.speed;
      state.lastTick = now;
      if (state.time >= state.episode.duration_s) {
        // Loop through the recorded episodes so the demo keeps moving unattended.
        const next = (state.episode.index + 1) % summary.episodes.length;
        el('episode').value = next;
        load(next).then(() => setPlaying(true));
        state.playing = false;
      } else {
        render();
        syncVideos(false);
      }
    }
    requestAnimationFrame(tick);
  }

  el('play').addEventListener('click', () => setPlaying(!state.playing));
  el('episode').addEventListener('change', event => load(Number(event.target.value)).then(() => setPlaying(state.playing)));
  el('speed').addEventListener('change', event => { state.speed = Number(event.target.value); syncVideos(true); });
  el('scrub').addEventListener('input', event => {
    state.time = Number(event.target.value) / summary.fps;
    render();
    syncVideos(true);
  });

  for (const id of ['cams', 'joints', 'legend', 'transport']) el(id).hidden = false;
  el('act-badge').textContent = 'REPLAY';
  el('act-badge').className = 'badge replay';
  await load(summary.episodes[0].index);
  requestAnimationFrame(tick);
  setPlaying(true);
  return summary;
}
