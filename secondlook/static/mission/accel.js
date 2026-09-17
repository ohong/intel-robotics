// INTEL ACCELERATION: recorded benchmark receipts from benchmarks.json, drawn as bars
// with their source and date. These are never live numbers; the live model latency is
// in the SEE and THINK panels.
const el = id => document.getElementById(id);

function group(data) {
  const section = document.createElement('section');
  section.className = 'bench';
  const head = document.createElement('header');
  const title = document.createElement('b');
  title.textContent = data.title;
  const source = document.createElement('small');
  source.textContent = `${data.source} · “${data.section}” · ${data.date}`;
  head.append(title, source);
  section.append(head);
  const slowest = Math.max(...data.rows.map(row => Number(row.median_ms)));
  for (const row of data.rows) {
    const line = document.createElement('div');
    line.className = `bench-row${row.deployed ? ' deployed' : ''}${row.parity.startsWith('failed') ? ' rejected' : ''}`;
    const name = document.createElement('span');
    name.textContent = row.runtime;
    const bar = document.createElement('i');
    bar.style.setProperty('--w', `${Number(row.median_ms) / slowest * 100}%`);
    const value = document.createElement('output');
    value.textContent = `${row.median_ms} ms`;
    line.title = `median ${row.median_ms} ms${row.p95_ms ? ` · p95 ${row.p95_ms} ms` : ''} · parity ${row.parity}${row.deployed ? ' · deployed' : ''}`;
    line.append(name, bar, value);
    section.append(line);
  }
  section.title = data.scope;
  return section;
}

export async function startAccel() {
  try {
    const response = await fetch('/static/mission/benchmarks.json', {cache: 'no-store'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const pilot = data.groups[0].rows;
    const baseline = Number(pilot[0].median_ms), deployed = pilot.find(row => row.deployed);
    el('accel-headline').textContent = `${(baseline / Number(deployed.median_ms)).toFixed(1)}× faster`;
    el('accel-sub').textContent = `${deployed.runtime} vs ${pilot[0].runtime} · median model call`;
    el('accel-groups').replaceChildren(...data.groups.map(group));
    el('accel-badge').title = data.note;
  } catch (error) {
    el('accel-groups').textContent = `Benchmark receipts unavailable: ${error.message}`;
  }
}
