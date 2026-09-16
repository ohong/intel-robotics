"""Real Anomalib PatchCore with auditable splits and raw (non-probability) scores.

Heavy dependencies load only inside runtime methods. Import and manifest checks work
on the development Mac without installing or changing the sponsor environment.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time
from typing import Any


class AnomalyContractError(ValueError):
    """Input data or artifact cannot support the requested evidence claim."""


class ObservationInvalidError(AnomalyContractError):
    """Image does not meet the configured observation requirements."""


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_manifest(path: str | Path, *, engineering_smoke: bool = False) -> dict:
    """Validate one combined split manifest, including image content leakage.

    Every row needs path, role, specimen_id, label and ground_truth containing
    source and authoritative. Use label=unknown and authoritative=false for smoke.
    Paths are relative to the manifest. Adjacent frames share a specimen_id.
    """
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get('schema_version') != 1 or not isinstance(data.get('images'), list) or not data['images']:
        raise AnomalyContractError('Expected schema_version=1 and nonempty images list')
    split_policy = data.get('split_policy', 'specimen_and_session')
    if split_policy not in {'specimen_and_session', 'specimen_only_pilot'}:
        raise AnomalyContractError('Unknown split_policy')
    policy_reason = data.get('split_policy_reason')
    if split_policy == 'specimen_only_pilot' and (not isinstance(policy_reason, str) or not policy_reason.strip()):
        raise AnomalyContractError('specimen_only_pilot requires a nonempty split_policy_reason')
    specimens, sessions, contents, rows = {}, {}, {}, []
    shared_sessions = set()
    session_aware = split_policy == 'specimen_only_pilot' or any('session_id' in row for row in data['images'])
    for original in data['images']:
        row = dict(original)
        role, specimen = row.get('role'), row.get('specimen_id')
        if role not in {'train', 'validation', 'test'} or not isinstance(specimen, str) or not specimen.strip():
            raise AnomalyContractError('Each image needs a valid role and nonempty specimen_id')
        if specimen in specimens and specimens[specimen] != role:
            raise AnomalyContractError(f'Specimen overlaps splits: {specimen}')
        specimens[specimen] = role
        if session_aware:
            session = row.get('session_id')
            if not isinstance(session, str) or not session.strip():
                raise AnomalyContractError('When session_id is supplied, every image needs a nonempty session_id')
            if session in sessions and sessions[session] != role:
                if split_policy == 'specimen_and_session':
                    raise AnomalyContractError(f'Capture session overlaps splits: {session}')
                shared_sessions.add(session)
            sessions[session] = role
        label, truth = row.get('label'), row.get('ground_truth', {})
        if label not in {'normal', 'anomalous', 'unknown'}:
            raise AnomalyContractError('Label must be normal, anomalous, or unknown')
        if not isinstance(truth, dict) or not isinstance(truth.get('source'), str) or not truth['source'].strip() or type(truth.get('authoritative')) is not bool:
            raise AnomalyContractError('Ground-truth source and explicit authoritative boolean are required')
        if label == 'unknown' and truth['authoritative']:
            raise AnomalyContractError('Unknown labels cannot be authoritative')
        if not engineering_smoke and (label == 'unknown' or not truth['authoritative']):
            raise AnomalyContractError('Task data needs authoritative ground truth; use engineering smoke for unlabeled data')
        if role == 'train' and not engineering_smoke and label != 'normal':
            raise AnomalyContractError('PatchCore task training requires authoritative normal specimens')
        if role == 'train' and engineering_smoke and label == 'anomalous':
            raise AnomalyContractError('Known anomalous images cannot enter the reference bank')
        image = (path.parent / row.get('path', '')).resolve()
        if not image.is_file():
            raise AnomalyContractError(f'Image file missing: {image}')
        digest = _hash(image)
        if digest in contents:
            raise AnomalyContractError(f'Duplicate image content or split leakage: {image}')
        contents[digest] = role
        row.update(path=str(image), sha256=digest)
        rows.append(row)
    if not any(row['role'] == 'train' for row in rows):
        raise AnomalyContractError('At least one training image is required')
    # File locations are operational details; identity follows contents and provenance.
    canonical = json.dumps([{k: v for k, v in row.items() if k != 'path'} for row in rows], sort_keys=True).encode()
    limitations = [] if session_aware else ['Capture session independence is not recorded; specimen and content separation only']
    if split_policy == 'specimen_only_pilot':
        limitations.append('Specimen-only pilot: capture sessions may cross splits; results do not establish session-independent generalization')
    return {'schema_version': 1, 'manifest_sha256': _hash(path),
            'dataset_sha256': hashlib.sha256(canonical).hexdigest(), 'images': rows,
            'split_policy': split_policy, 'split_policy_reason': policy_reason,
            'shared_sessions': sorted(shared_sessions), 'split_limitations': limitations}


def _versions() -> dict:
    return {name: importlib.metadata.version(name) for name in ('anomalib', 'openvino', 'torch', 'timm', 'numpy', 'Pillow')}


def _roi(value=None) -> list[float]:
    value = [0., 0., 1., 1.] if value is None else value
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise AnomalyContractError('ROI requires normalized left, top, right, bottom')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise AnomalyContractError('ROI coordinates must be finite numbers')
    left, top, right, bottom = map(float, value)
    if not 0 <= left < right <= 1 or not 0 <= top < bottom <= 1:
        raise AnomalyContractError('ROI must have positive area within [0, 1]')
    return [left, top, right, bottom]


def _pixels(image: Any):
    import numpy as np
    from PIL import Image
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            return source.convert('RGB')
    else:
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) == 0:
            raise AnomalyContractError('Expected nonempty RGB uint8 image, not BGR or normalized data')
        return Image.fromarray(array)


def _foreground_config() -> dict:
    return {'saturation_min': .25, 'value_min': 50, 'min_area_fraction': .01,
            'min_bbox_side': 20, 'padding_fraction': .08, 'ambiguity_area_ratio': .25}


def _foreground_box(pixels, config: dict) -> tuple[list[int], dict]:
    """Find geometry from color, preserving every original pixel inside the box."""
    import cv2
    import numpy as np
    defaults = _foreground_config()
    if set(config) != set(defaults) or any(type(v) not in (int, float) or not math.isfinite(v) for v in config.values()):
        raise AnomalyContractError('Malformed foreground crop configuration')
    if not (0 <= config['saturation_min'] < 1 and 0 <= config['value_min'] < 255
            and 0 < config['min_area_fraction'] <= 1 and config['min_bbox_side'] >= 1
            and 0 <= config['padding_fraction'] <= 1 and 0 < config['ambiguity_area_ratio'] <= 1):
        raise AnomalyContractError('Invalid foreground crop thresholds')
    array = np.asarray(pixels)
    high = array.max(axis=2).astype(np.float32)
    low = array.min(axis=2).astype(np.float32)
    saturated = ((high - low) / np.maximum(high, 1) > config['saturation_min']) & (high > config['value_min'])
    count, _, stats, _ = cv2.connectedComponentsWithStats(saturated.astype(np.uint8), connectivity=8)
    components = sorted(stats[1:count], key=lambda stat: int(stat[cv2.CC_STAT_AREA]), reverse=True)
    height, width = array.shape[:2]
    if not components or components[0][cv2.CC_STAT_AREA] < width * height * config['min_area_fraction']:
        raise ObservationInvalidError('Invalid observation: colored foreground absent or too small')
    x, y, w, h, area = map(int, components[0])
    if min(w, h) < config['min_bbox_side']:
        raise ObservationInvalidError('Invalid observation: colored foreground bounding box too small')
    if len(components) > 1 and components[1][cv2.CC_STAT_AREA] > area * config['ambiguity_area_ratio']:
        raise ObservationInvalidError('Invalid observation: multiple comparable colored foreground components')
    padding = math.ceil(max(w, h) * config['padding_fraction'])
    box = [max(0, x - padding), max(0, y - padding), min(width, x + w + padding), min(height, y + h + padding)]
    return box, {'component_area': area, 'component_bbox_in_parent': [x, y, x + w, y + h],
                 'padding_pixels': padding, 'mask_applied_to_model_input': False}


def _prepare(image: Any, size: int, roi=None, foreground_crop=None):
    import numpy as np
    from PIL import Image
    pixels = _pixels(image)
    width, height = pixels.size
    left, top, right, bottom = _roi(roi)
    box = [math.floor(left * width), math.floor(top * height),
           math.ceil(right * width), math.ceil(bottom * height)]
    parent_box = box.copy()
    foreground_details = None
    if foreground_crop:
        relative, foreground_details = _foreground_box(pixels.crop(parent_box), foreground_crop)
        box = [relative[0] + parent_box[0], relative[1] + parent_box[1],
               relative[2] + parent_box[0], relative[3] + parent_box[1]]
    pixels = pixels.crop(box).resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(pixels, dtype=np.float32) / 255.0
    array = (array - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None]), {
        'image_size': [width, height], 'roi_normalized': [left, top, right, bottom],
        'roi_pixels': box, 'parent_roi_pixels': parent_box, 'foreground_crop': foreground_details,
        'map_coordinates': 'ROI-local; pixel box uses exclusive right/bottom'}


def _preprocess(image: Any, size: int, roi=None, foreground_crop=None):
    return _prepare(image, size, roi, foreground_crop)[0]


def _model(config: dict, *, pretrained: bool):
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel
    return PatchcoreModel(layers=config['layers'], backbone=config['backbone'],
                          pre_trained=pretrained, num_neighbors=config['num_neighbors'])


def fit(manifest: str | Path, output: str | Path, *, engineering_smoke: bool = False,
        image_size: int = 128, sampling_ratio: float = .02, max_images: int = 64, roi=None,
        foreground_crop: bool = False, num_neighbors: int = 1) -> dict:
    """Fit Anomalib's actual k-center memory bank on CPU, then export OpenVINO IR.

    Bounds avoid accidental unbounded work. No validation or test image is fitted.
    This method never opens cameras or robot devices.
    """
    if image_size < 64 or image_size > 512 or image_size % 32:
        raise AnomalyContractError('image_size must be a multiple of 32 in [64, 512]')
    if not 0 < sampling_ratio <= 1 or not 1 <= max_images <= 256:
        raise AnomalyContractError('Invalid coreset ratio or image bound')
    if type(num_neighbors) is not int or not 1 <= num_neighbors <= 20:
        raise AnomalyContractError('num_neighbors must be an integer in [1, 20]')
    roi = _roi(roi)
    foreground = _foreground_config() if foreground_crop else None
    dataset = validate_manifest(manifest, engineering_smoke=engineering_smoke)
    train = [row for row in dataset['images'] if row['role'] == 'train']
    if len(train) > max_images:
        raise AnomalyContractError('Training exceeds max_images; explicitly select a bounded manifest')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise AnomalyContractError('Output must be new or empty; existing artifacts are never overwritten')
    import torch
    import openvino as ov
    torch.set_num_threads(4)
    torch.manual_seed(17)
    config = {'backbone': 'wide_resnet50_2', 'layers': ['layer2', 'layer3'],
              'num_neighbors': num_neighbors, 'image_size': image_size, 'sampling_ratio': sampling_ratio,
              'roi': roi,
              'foreground_crop': foreground,
              'preprocessing': 'RGB; fixed normalized ROI (floor left/top, ceil right/bottom); optional colored foreground bbox retaining original pixels; PIL bilinear square resize; float32 /255; ImageNet mean/std',
              'seed': 17, 'torch_threads': 4}
    model = _model(config, pretrained=True)
    model.train()
    model.feature_extractor.eval()  # Freeze pretrained BatchNorm statistics during reference fitting.
    started = time.perf_counter()
    with torch.inference_mode():
        for row in train:
            model(torch.from_numpy(_preprocess(row['path'], image_size, roi, foreground)))
        model.subsample_embedding(sampling_ratio=sampling_ratio)
    if not len(model.memory_bank):
        raise AnomalyContractError('Coreset is empty; increase images or sampling_ratio')
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / 'model.pt')

    class Export(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixels):
            prediction = self.inner(pixels)
            return prediction.pred_score, prediction.anomaly_map

    example = torch.from_numpy(_preprocess(train[0]['path'], image_size, roi, foreground))
    converted = ov.convert_model(Export(model).eval(), example_input=example,
                                 input=[(1, 3, image_size, image_size)])
    converted.output(0).get_tensor().set_names({'score'})
    converted.output(1).get_tensor().set_names({'anomaly_map'})
    ov.save_model(converted, output / 'model.xml', compress_to_fp16=False)
    metadata = {'schema_version': 1, 'model': 'Anomalib PatchCore', 'config': config,
                'versions': _versions(), 'dataset': dataset,
                'evidence_kind': 'engineering_smoke' if engineering_smoke else 'task_model_uncalibrated',
                'threshold': None, 'calibration': None, 'evaluation': None,
                'train_images': len(train), 'memory_bank_shape': list(model.memory_bank.shape),
                'fit_and_export_seconds': time.perf_counter() - started,
                'precision': 'FP32 export; device compilation precision reported by benchmark',
                'files': {name: _hash(output / name) for name in ('model.pt', 'model.xml', 'model.bin')}}
    metadata['model_id'] = hashlib.sha256(json.dumps({'config': config, 'files': metadata['files']}, sort_keys=True).encode()).hexdigest()
    metadata['artifact_id'] = _artifact_identity(metadata)
    _write(output / 'metadata.json', metadata)
    return metadata


def _artifact_identity(metadata: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in metadata.items() if k != 'artifact_id'}, sort_keys=True).encode()).hexdigest()


def _metadata(directory: Path) -> dict:
    metadata = json.loads((directory / 'metadata.json').read_text())
    for name, digest in metadata['files'].items():
        if _hash(directory / name) != digest:
            raise AnomalyContractError(f'Artifact hash mismatch: {name}')
    threshold = metadata.get('threshold')
    if threshold is not None:
        if type(threshold) not in (int, float) or not math.isfinite(threshold):
            raise AnomalyContractError('Threshold must be finite')
        if metadata.get('evidence_kind') not in {'task_model_calibrated', 'task_model_evaluated'} or not metadata.get('calibration'):
            raise AnomalyContractError('Threshold requires authoritative validation calibration')
        if metadata.get('evidence_kind') == 'task_model_evaluated' and not metadata.get('evaluation'):
            raise AnomalyContractError('Threshold requires a held-out evaluation for evaluated evidence')
    margin = metadata.get('uncertainty_margin', 0.)
    if type(margin) not in (int, float) or not math.isfinite(margin) or margin < 0:
        raise AnomalyContractError('Uncertainty margin must be finite and nonnegative')
    if metadata.get('artifact_id') and metadata['artifact_id'] != _artifact_identity(metadata):
        raise AnomalyContractError('Artifact metadata identity mismatch')
    _roi(metadata.get('config', {}).get('roi'))
    return metadata


class AnomalyDetector:
    """Reusable runtime. Input contains pixels only; no ground-truth input exists."""
    def __init__(self, artifact_dir: str | Path, device: str = 'CPU', backend: str = 'openvino', precision: str | None = None):
        directory = Path(artifact_dir)
        self.metadata = _metadata(directory)
        self.backend, self.device = backend, device
        started = time.perf_counter()
        if backend == 'openvino':
            import openvino as ov
            core = ov.Core()
            compile_config = {'PERFORMANCE_HINT': 'LATENCY'}
            if precision is not None:
                if precision not in {'f32', 'f16'}:
                    raise AnomalyContractError('precision must be f32 or f16')
                compile_config['INFERENCE_PRECISION_HINT'] = getattr(ov.Type, precision)
            self.runtime = core.compile_model(str(directory / 'model.xml'), device, compile_config)
            devices = self.runtime.get_property('EXECUTION_DEVICES')
            self.execution_devices = [devices] if isinstance(devices, str) else list(devices)
            self.runtime_config = {'performance_hint': 'LATENCY', 'requested_precision': precision}
            for prop in ('INFERENCE_NUM_THREADS', 'NUM_STREAMS'):
                try:
                    self.runtime_config[prop] = str(self.runtime.get_property(prop))
                except RuntimeError:
                    pass
            self.inference_precision = str(self.runtime.get_property('INFERENCE_PRECISION_HINT'))
        elif backend == 'torch' and device == 'CPU':
            import torch
            torch.set_num_threads(4)
            self.runtime = _model(self.metadata['config'], pretrained=False)
            self.runtime.load_state_dict(torch.load(directory / 'model.pt', map_location='cpu', weights_only=True))
            self.runtime.eval()
            self.execution_devices, self.inference_precision = ['CPU'], 'float32'
            self.runtime_config = {'torch_num_threads': torch.get_num_threads()}
        else:
            raise AnomalyContractError('Supported backends: openvino on explicit device; torch on CPU')
        self.load_compile_ms = (time.perf_counter() - started) * 1000

    def _run(self, tensor):
        if self.backend == 'openvino':
            output = self.runtime(tensor)
            return output[self.runtime.output('score')], output[self.runtime.output('anomaly_map')]
        import torch
        with torch.inference_mode():
            output = self.runtime(torch.from_numpy(tensor))
        return output.pred_score.numpy(), output.anomaly_map.numpy()

    def infer(self, image: Any) -> dict:
        import numpy as np
        started = time.perf_counter()
        tensor, geometry = _prepare(image, self.metadata['config']['image_size'],
                                    self.metadata['config'].get('roi'), self.metadata['config'].get('foreground_crop'))
        preprocessed = time.perf_counter()
        score, anomaly_map = self._run(tensor)
        finished = time.perf_counter()
        score, anomaly_map = float(np.asarray(score).reshape(-1)[0]), np.asarray(anomaly_map).squeeze().copy()
        if not math.isfinite(score) or anomaly_map.ndim != 2 or not np.isfinite(anomaly_map).all():
            raise AnomalyContractError('Model returned malformed or nonfinite inference')
        threshold = self.metadata['threshold']
        margin = self.metadata.get('uncertainty_margin', 0.)
        disposition, reason = _decision(score, threshold, margin)
        geometry['map_size'] = [anomaly_map.shape[1], anomaly_map.shape[0]]
        return {'score': score, 'threshold': threshold, 'disposition': disposition,
                'uncertainty_band': [threshold - margin, threshold + margin] if threshold is not None else None,
                'decision_reason': reason, 'localization_geometry': geometry,
                'anomaly_map': anomaly_map, 'latency_ms': (finished - started) * 1000,
                'model_latency_ms': (finished - preprocessed) * 1000,
                'backend': self.backend, 'device': self.device,
                'artifact_id': self.metadata['artifact_id'], 'evidence_kind': self.metadata['evidence_kind']}


def _decision(score: float, threshold: float | None, margin: float = 0.) -> tuple[str, str]:
    if threshold is None:
        return 'UNKNOWN', 'uncalibrated'
    if margin > 0 and threshold - margin <= score <= threshold + margin:
        return 'UNKNOWN', 'within_uncertainty_band'
    if score >= threshold:
        return 'ANOMALOUS', 'above_uncertainty_band'
    return 'NORMAL', 'below_uncertainty_band'


def _metrics(scores: list[tuple[float, str]], threshold: float) -> dict:
    tp = sum(s >= threshold and label == 'anomalous' for s, label in scores)
    fp = sum(s >= threshold and label == 'normal' for s, label in scores)
    fn = sum(s < threshold and label == 'anomalous' for s, label in scores)
    tn = sum(s < threshold and label == 'normal' for s, label in scores)
    return {'count': len(scores), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def _distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {'count': 0}
    def quantile(q):
        index = (len(ordered) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return {'count': len(ordered), 'min': ordered[0], 'p05': quantile(.05),
            'median': quantile(.5), 'p95': quantile(.95), 'max': ordered[-1]}


def _report(records: list[dict], threshold: float, margin: float) -> dict:
    scores = [(r['score'], r['label']) for r in records]
    decisions = [{**r, 'decision': _decision(r['score'], threshold, margin)[0]} for r in records]
    counts = {label: {decision: sum(r['label'] == label and r['decision'] == decision for r in decisions)
                     for decision in ('NORMAL', 'ANOMALOUS', 'UNKNOWN')}
              for label in ('normal', 'anomalous')}
    normal_scores = [s for s, label in scores if label == 'normal']
    defect_scores = [s for s, label in scores if label == 'anomalous']
    return {'count': len(records), 'threshold_confusion': _metrics(scores, threshold),
            'decision_counts': counts, 'missed_defects': counts['anomalous']['NORMAL'],
            'rejected_good_parts': counts['normal']['ANOMALOUS'],
            'uncertain_defects': counts['anomalous']['UNKNOWN'],
            'uncertain_good_parts': counts['normal']['UNKNOWN'],
            'score_distributions': {label: _distribution([s for s, label_ in scores if label_ == label])
                                    for label in ('normal', 'anomalous')},
            'normal_scores_strictly_below_defect_scores': (max(normal_scores) < min(defect_scores))
                if normal_scores and defect_scores else None,
            'records': decisions,
            'unit': 'images; repeated views of a specimen are not independent trials',
            'statistical_limit': 'Descriptive counts only; small specimen counts do not establish deployment error rates',
            'specimen_counts_by_label': {label: len({r['specimen_id'] for r in records if r['label'] == label})
                                         for label in ('normal', 'anomalous')},
            'specimen_count': len({r['specimen_id'] for r in records}),
            'session_count': len({r['session_id'] for r in records if r.get('session_id')})}


def _split_rows(metadata: dict, manifest: str | Path, role: str) -> tuple[dict, list[dict]]:
    dataset = validate_manifest(manifest)
    original = metadata['dataset']
    if (dataset['split_policy'] != original.get('split_policy', 'specimen_and_session') or
            dataset.get('split_policy_reason') != original.get('split_policy_reason')):
        raise AnomalyContractError('Split policy and its recorded reason must remain unchanged after fit')
    if role == 'test':
        # Only append newly collected tests; preserve every fitted/selected/evaluation row.
        def identity(row):
            return json.dumps({k: v for k, v in row.items() if k != 'path'}, sort_keys=True)
        original_rows = {identity(row) for row in original['images']}
        current_rows = {identity(row) for row in dataset['images']}
        retained = original_rows.issubset(current_rows)
        added_rows = [row for row in dataset['images'] if identity(row) not in original_rows]
        if not retained or any(row['role'] != 'test' for row in added_rows):
            raise AnomalyContractError('Final test requires unchanged images/provenance and only appended test rows')
    elif dataset['dataset_sha256'] != original['dataset_sha256']:
        raise AnomalyContractError('Requires the complete original split manifest and unchanged images')
    rows = [r for r in dataset['images'] if r['role'] == role]
    if {r['label'] for r in rows} != {'normal', 'anomalous'}:
        raise AnomalyContractError(f'{role} requires independent normal and anomalous specimens')
    return dataset, rows


def _score_rows(detector, rows: list[dict]) -> list[dict]:
    records = []
    for row in rows:
        score = float(detector.infer(row['path'])['score'])
        if not math.isfinite(score):
            raise AnomalyContractError('Cannot select or evaluate a threshold from a nonfinite score')
        records.append({**{k: row.get(k) for k in ('path', 'sha256', 'specimen_id', 'session_id', 'label')}, 'score': score})
    return records


def calibrate(artifact_dir: str | Path, manifest: str | Path, *, uncertainty_fraction: float = .05) -> dict:
    """Freeze a validation-only threshold and uncertainty policy. Never infer test images.

    The uncertainty width is a declared heuristic, not an estimated probability.
    Once selected, these settings cannot be changed on this fitted artifact.
    """
    directory = Path(artifact_dir)
    metadata = _metadata(directory)
    if metadata['evidence_kind'] == 'engineering_smoke':
        raise AnomalyContractError('Smoke artifacts cannot be calibrated; refit authoritative training data')
    if metadata.get('calibration') or metadata.get('evaluation') or metadata.get('threshold') is not None:
        raise AnomalyContractError('Calibration is frozen; create a new fitted artifact for a new protocol')
    if not math.isfinite(uncertainty_fraction) or not 0 <= uncertainty_fraction <= .5:
        raise AnomalyContractError('uncertainty_fraction must be finite in [0, .5]')
    dataset, rows = _split_rows(metadata, manifest, 'validation')
    detector = AnomalyDetector(directory, device='CPU', precision='f32')
    records = _score_rows(detector, rows)
    scores = [(r['score'], r['label']) for r in records]
    distinct = sorted({score for score, _ in scores})
    candidates = distinct + [(a + b) / 2 for a, b in zip(distinct, distinct[1:])]
    # Prefer the center of an equally accurate gap to avoid unstable boundary decisions.
    threshold = max(candidates, key=lambda value: (_metrics(scores, value)['f1'],
                    min(abs(value - score) for score in distinct), value))
    margin = (distinct[-1] - distinct[0]) * uncertainty_fraction
    # Equal validation scores contain no separation; retain explicit abstention at the threshold.
    if uncertainty_fraction > 0 and margin == 0:
        margin = max(math.ulp(threshold), 1e-12)
    metadata['threshold'], metadata['uncertainty_margin'] = threshold, margin
    metadata['evidence_kind'] = 'task_model_calibrated'
    metadata['calibration'] = {
        'method': 'maximum validation F1; widest observed score gap, then highest threshold break ties',
        'backend': 'openvino', 'device': 'CPU', 'precision': 'f32',
        'uncertainty_policy': 'margin = fraction of validation score range; closed interval abstains; not calibrated probability',
        'uncertainty_fraction': uncertainty_fraction, 'threshold': threshold, 'uncertainty_margin': margin,
        'dataset_sha256': dataset['dataset_sha256'], 'test_inference_performed': False,
        'split_limitations': dataset['split_limitations'], **_report(records, threshold, margin)}
    metadata['calibration']['calibration_id'] = hashlib.sha256(
        json.dumps(metadata['calibration'], sort_keys=True).encode()).hexdigest()
    metadata['artifact_id'] = _artifact_identity(metadata)
    _write(directory / 'metadata.json', metadata)
    return {'artifact_id': metadata['artifact_id'], 'threshold': threshold,
            'uncertainty_margin': margin, 'calibration': metadata['calibration'], 'evaluation': None}


def evaluate(artifact_dir: str | Path, manifest: str | Path, *, parity_atol: float = .1,
             parity_rtol: float = .01) -> dict:
    """Run final test once with frozen settings and compare CPU OpenVINO with Torch.

    The report records failed parity; evaluation is not itself an accuracy acceptance gate.
    Test labels never select thresholds or alter the fitted normal memory bank.
    """
    directory = Path(artifact_dir)
    metadata = _metadata(directory)
    if metadata.get('evaluation'):
        raise AnomalyContractError('Final test is already recorded; use its saved report, not repeated tuning')
    if metadata.get('threshold') is None or not metadata.get('calibration'):
        raise AnomalyContractError('Calibrate using validation data before final test evaluation')
    if any(not math.isfinite(v) or v < 0 for v in (parity_atol, parity_rtol)):
        raise AnomalyContractError('Parity tolerances must be finite and nonnegative')
    dataset, rows = _split_rows(metadata, manifest, 'test')
    detector = AnomalyDetector(directory, device='CPU', precision='f32')
    original = AnomalyDetector(directory, backend='torch')
    import numpy as np
    records, parity = [], []
    threshold, margin = metadata['threshold'], metadata.get('uncertainty_margin', 0.)
    for row in rows:
        exported, reference = detector.infer(row['path']), original.infer(row['path'])
        records.append({**{k: row.get(k) for k in ('path', 'sha256', 'specimen_id', 'session_id', 'label')},
                        'score': exported['score']})
        score_pass = bool(np.isclose(exported['score'], reference['score'], atol=parity_atol, rtol=parity_rtol))
        map_pass = bool(np.allclose(exported['anomaly_map'], reference['anomaly_map'], atol=parity_atol, rtol=parity_rtol))
        decision_pass = exported['disposition'] == reference['disposition']
        parity.append({'sha256': row['sha256'], 'score_abs_error': abs(exported['score'] - reference['score']),
                       'map_max_abs_error': float(np.max(np.abs(exported['anomaly_map'] - reference['anomaly_map']))),
                       'torch_score': reference['score'], 'openvino_score': exported['score'],
                       'torch_decision': reference['disposition'], 'openvino_decision': exported['disposition'],
                       'score_pass': score_pass, 'map_pass': map_pass, 'decision_pass': decision_pass,
                       'pass': score_pass and map_pass and decision_pass})
    metadata['evaluation'] = {
        'split': 'test', 'dataset_sha256': dataset['dataset_sha256'],
        'dataset': dataset,
        'calibration_id': metadata['calibration'].get('calibration_id'),
        'evaluated_artifact_id': metadata['artifact_id'], 'threshold': threshold, 'uncertainty_margin': margin,
        'backend': 'openvino', 'device': 'CPU', 'precision': 'f32',
        'split_limitations': dataset['split_limitations'], **_report(records, threshold, margin),
        'export_parity': {'atol': parity_atol, 'rtol': parity_rtol, 'records': parity,
                          'pass': all(r['pass'] for r in parity)}}
    metadata['evidence_kind'] = 'task_model_evaluated'
    metadata['artifact_id'] = _artifact_identity(metadata)
    _write(directory / 'metadata.json', metadata)
    return {'artifact_id': metadata['artifact_id'], 'evaluation': metadata['evaluation']}


def benchmark(artifact_dir: str | Path, image: str | Path, *, devices: tuple[str, ...] = ('CPU', 'GPU', 'NPU'),
              iterations: int = 10, warmup: int = 3, atol: float = .1, rtol: float = .01,
              precision: str | None = None) -> dict:
    """Measure the actual exported model, including explicit unsupported-device errors.

    Model timing excludes preprocessing. All backends receive identical tensors.
    Parity measures scores, maps, and frozen decisions against the saved Torch model.
    """
    import numpy as np
    if not 1 <= iterations <= 1000 or not 1 <= warmup <= 100:
        raise AnomalyContractError('Benchmark iteration bounds exceeded')
    reference = AnomalyDetector(artifact_dir, backend='torch')
    tensor = _preprocess(image, reference.metadata['config']['image_size'],
                         reference.metadata['config'].get('roi'), reference.metadata['config'].get('foreground_crop'))
    expected_score, expected_map = reference._run(tensor)
    threshold, margin = reference.metadata['threshold'], reference.metadata.get('uncertainty_margin', 0.)
    expected_decision = _decision(float(np.asarray(expected_score).reshape(-1)[0]), threshold, margin)[0]
    report = {'artifact_id': reference.metadata['artifact_id'], 'image_sha256': _hash(Path(image)),
              'versions': _versions(), 'evidence_kind': reference.metadata['evidence_kind'],
              'iterations': iterations, 'warmup': warmup, 'parity_atol': atol, 'parity_rtol': rtol,
              'timing_scope': 'batch 1 synchronous model call only; same preprocessed input', 'results': []}
    for backend, device in [('torch', 'CPU')] + [('openvino', d) for d in devices]:
        try:
            detector = reference if backend == 'torch' else AnomalyDetector(artifact_dir, device=device, precision=precision)
            for _ in range(warmup):
                detector._run(tensor)
            times = []
            for _ in range(iterations):
                started = time.perf_counter()
                score, anomaly_map = detector._run(tensor)
                times.append((time.perf_counter() - started) * 1000)
            decision = _decision(float(np.asarray(score).reshape(-1)[0]), threshold, margin)[0]
            report['results'].append({'backend': backend, 'requested_device': device, 'status': 'measured',
                'execution_devices': detector.execution_devices, 'inference_precision': detector.inference_precision,
                'runtime_config': detector.runtime_config,
                'load_compile_ms': detector.load_compile_ms, 'median_ms': float(np.median(times)),
                'p95_ms': float(np.percentile(times, 95)), 'samples_ms': times,
                'score': float(np.asarray(score).reshape(-1)[0]),
                'score_abs_error': float(np.max(np.abs(score - expected_score))),
                'map_max_abs_error': float(np.max(np.abs(anomaly_map - expected_map))),
                'decision': decision, 'reference_decision': expected_decision,
                'decision_parity_pass': decision == expected_decision,
                'parity_pass': bool(np.allclose(score, expected_score, atol=atol, rtol=rtol) and
                                    np.allclose(anomaly_map, expected_map, atol=atol, rtol=rtol) and
                                    decision == expected_decision)})
        except Exception as exc:
            report['results'].append({'backend': backend, 'requested_device': device, 'status': 'failed_or_unsupported',
                                      'error': f'{type(exc).__name__}: {exc}'})
    return report
