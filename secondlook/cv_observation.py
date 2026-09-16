"""Associate conservative frame checks with anomaly output; never infer object truth."""
from __future__ import annotations

import math
import time
from typing import Any

from .anomaly import ObservationInvalidError


def quality_stats(rgb: Any) -> dict:
    """Descriptive image statistics only. These do not establish object visibility."""
    import numpy as np
    a = np.asarray(rgb)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3 or not a.size:
        raise ValueError('Expected nonempty RGB uint8 pixels')
    gray = a.astype(np.float32).mean(axis=2)
    dx = float(np.abs(np.diff(gray, axis=1)).mean()) if gray.shape[1] > 1 else 0.
    dy = float(np.abs(np.diff(gray, axis=0)).mean()) if gray.shape[0] > 1 else 0.
    return {'width': int(a.shape[1]), 'height': int(a.shape[0]),
            'mean_brightness_0_255': float(gray.mean()), 'std_brightness_0_255': float(gray.std()),
            'minimum_channel_value': int(a.min()), 'maximum_channel_value': int(a.max()),
            'fraction_pixels_all_channels_le_2': float(np.all(a <= 2, axis=2).mean()),
            'fraction_pixels_any_channel_ge_253': float(np.any(a >= 253, axis=2).mean()),
            'mean_absolute_neighbor_difference': (dx + dy) / 2,
            'interpretation': 'Descriptive statistics; object presence, occlusion and defect visibility unverified'}


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def observe_frame(detector: Any, rgb: Any, frame_id: str, captured_at: float, *,
                  now: float | None = None, max_age: float = 1.,
                  validated_evidence: dict | None = None) -> dict:
    """Receive full RGB pixels and a capture timestamp in the local monotonic clock.

    The detector owns ROI preprocessing. Physical predicates require explicit evidence
    associated with this frame. Ground-truth defect labels are never passed to infer.
    """
    import numpy as np
    started = time.perf_counter()
    now = time.monotonic() if now is None else now
    if not _number(max_age) or max_age <= 0 or not _number(now):
        raise ValueError('now and positive max_age must be finite')
    flags = []
    if not isinstance(frame_id, str) or not frame_id.strip():
        flags.append('INVALID_FRAME_ID')
    if not _number(captured_at) or captured_at < 0:
        flags.append('INVALID_CAPTURE_TIMESTAMP')
    elif not 0 <= now - captured_at <= max_age:
        flags.append('STALE_OR_FUTURE_FRAME')
    try:
        a = np.asarray(rgb)
    except (TypeError, ValueError):
        a = np.asarray(None)
    if a.ndim != 3 or (a.ndim == 3 and (a.shape[2] != 3 or min(a.shape[:2]) == 0)):
        flags.append('INVALID_RGB_SHAPE')
    if not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
        flags.append('NONFINITE_OR_NONNUMERIC_PIXELS')
    if a.dtype != np.uint8:
        flags.append('INVALID_RGB_DTYPE')
    stats = None
    if not flags or all(f in {'INVALID_FRAME_ID', 'INVALID_CAPTURE_TIMESTAMP', 'STALE_OR_FUTURE_FRAME'} for f in flags):
        stats = quality_stats(a)
        # Only near-zero signal and exact spatial uniformity are rejected here.
        # Task-specific blur, exposure, object and occlusion checks need validation.
        if stats['maximum_channel_value'] <= 2:
            flags.append('NO_USABLE_LIGHT_SIGNAL')
        if np.all(a == a[0, 0]):
            flags.append('SPATIALLY_UNIFORM_FRAME')
    physical = {name: {'status': 'UNVERIFIED', 'source': None} for name in
                ('object_presence', 'occlusion', 'defect_visibility')}
    if validated_evidence is not None:
        if (not isinstance(validated_evidence, dict) or validated_evidence.get('frame_id') != frame_id or
                validated_evidence.get('validated') is not True or
                not isinstance(validated_evidence.get('source'), str) or not validated_evidence['source'].strip()):
            raise ValueError('Validated evidence needs matching frame_id, validated=true and source')
        allowed = {'object_presence': {'PRESENT', 'ABSENT'}, 'occlusion': {'CLEAR', 'OCCLUDED'},
                   'defect_visibility': {'VISIBLE', 'NOT_VISIBLE'}}
        for name, values in allowed.items():
            value = validated_evidence.get(name)
            if value is not None:
                if value not in values:
                    raise ValueError(f'Invalid validated {name}')
                physical[name] = {'status': value, 'source': validated_evidence['source']}
    pixel_flags = list(flags)
    for name, bad_status in (('object_presence', 'ABSENT'), ('occlusion', 'OCCLUDED'),
                             ('defect_visibility', 'NOT_VISIBLE')):
        if physical[name]['status'] == bad_status:
            flags.append(f'{name.upper()}_{bad_status}')
    metadata = getattr(detector, 'metadata', {})
    artifact_id = metadata.get('artifact_id') if isinstance(metadata, dict) else None
    result = {'schema_version': 1, 'frame_id': frame_id, 'observation_id': frame_id,
              'captured_at': captured_at if _number(captured_at) else None,
              'capture_timestamp': captured_at if _number(captured_at) else None,
              'timestamp_source': 'publisher_monotonic', 'frame_validity': 'INVALID' if pixel_flags else 'VALID',
              'observation_validity': ('INVALID' if flags else 'UNVERIFIED' if
                  any(item['status'] == 'UNVERIFIED' for item in physical.values()) else 'VALID'),
              'quality_flags': flags, 'quality_stats': stats, **physical,
              'inference_executed': False, 'anomaly_status': 'NOT_EVALUATED',
              'score': None, 'threshold': None, 'disposition': 'UNKNOWN', 'anomaly_map': None,
              'artifact_id': artifact_id, 'model_artifact_id': artifact_id, 'uncertainty_band': None,
              'latency_ms': None, 'model_latency_ms': None}
    if not flags:
        try:
            raw = detector.infer(a)
        except ObservationInvalidError as error:
            # This exception is raised before the model runs. Other failures must
            # propagate so a broken model cannot be mistaken for bad input.
            flags.append('PREPROCESSING_REJECTED')
            result.update(observation_validity='INVALID', decision_reason=str(error))
        else:
            if not _number(raw.get('score')) or raw.get('disposition') not in {'NORMAL', 'ANOMALOUS', 'UNKNOWN', 'UNCERTAIN'}:
                raise ValueError('Detector returned malformed score or disposition')
            if raw.get('threshold') is not None and not _number(raw['threshold']):
                raise ValueError('Detector returned nonfinite threshold')
            anomaly_map = raw.get('anomaly_map')
            if anomaly_map is not None:
                values = np.asarray(anomaly_map)
                if values.ndim != 2 or not values.size or not np.isfinite(values).all():
                    raise ValueError('Detector returned malformed anomaly map')
            # Copy only detector fields; an implementation cannot overwrite association.
            fields = ('score', 'threshold', 'disposition', 'anomaly_map', 'latency_ms', 'model_latency_ms',
                      'backend', 'device', 'artifact_id', 'evidence_kind', 'execution_devices',
                      'inference_precision', 'uncertainty_band', 'roi', 'map_coordinates', 'uncertainty',
                      'localization_geometry', 'decision_reason')
            result.update({key: raw[key] for key in fields if key in raw})
            result.update(inference_executed=True, anomaly_status=raw['disposition'],
                          model_artifact_id=raw.get('artifact_id'))
    result.update(anomaly_score=result['score'], model_version=result['artifact_id'],
                  inference_ms=result['latency_ms'],
                  decision=('INVALID' if flags else {'NORMAL': 'GOOD', 'ANOMALOUS': 'DEFECT'}.get(result['disposition'], 'UNCERTAIN')),
                  localization_geometry=result.get('localization_geometry'))
    result['processing_ms'] = (time.perf_counter() - started) * 1000
    return result
