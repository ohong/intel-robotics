// Joint HUD shared by REPLAY and LIVE: one bar per joint, filled from zero to the measured
// position, with a marker at the action target when one exists.
export function createJointBars(twin, container, names) {
  const rows = names.map(name => {
    const row = document.createElement('div');
    row.className = 'joint';
    row.innerHTML = `<span>${name}</span><div class="track"><i></i><b></b></div><output>—</output>`;
    container.append(row);
    return {limits: twin.limitsDeg(name) || [-180, 180], fill: row.querySelector('i'),
            marker: row.querySelector('b'), value: row.querySelector('output')};
  });
  const percent = ([low, high], degrees) => Math.min(100, Math.max(0, (degrees - low) / (high - low) * 100));
  return function update(measured, target) {
    rows.forEach((row, index) => {
      const zero = percent(row.limits, 0), now = percent(row.limits, measured[index]);
      row.fill.style.left = `${Math.min(zero, now)}%`;
      row.fill.style.width = `${Math.abs(now - zero)}%`;
      row.marker.hidden = !target;
      if (target) row.marker.style.left = `${percent(row.limits, target[index])}%`;
      row.value.textContent = `${measured[index].toFixed(1)}°`;
    });
  };
}
