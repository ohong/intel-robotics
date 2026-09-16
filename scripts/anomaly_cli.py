#!/usr/bin/env python3
"""Bounded offline detector commands; never access cameras or robot devices."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from secondlook.anomaly import AnomalyDetector, benchmark, calibrate, evaluate, fit, validate_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('validate')
    check.add_argument('--manifest', required=True,
                       help='JSON manifest; explicit specimen_only_pilot policy allows recorded shared sessions')
    check.add_argument('--engineering-smoke', action='store_true')
    train = commands.add_parser('fit')
    train.add_argument('--manifest', required=True)
    train.add_argument('--output', required=True)
    train.add_argument('--engineering-smoke', action='store_true')
    train.add_argument('--image-size', type=int, default=128)
    train.add_argument('--sampling-ratio', type=float, default=.02)
    train.add_argument('--num-neighbors', type=int, default=1,
                       help='PatchCore score weighting neighborhood size, integer 1..20')
    train.add_argument('--max-images', type=int, default=64)
    train.add_argument('--roi', nargs=4, type=float, metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'))
    train.add_argument('--foreground-crop', action='store_true',
                       help='Within fixed ROI, crop dominant colored component bbox; preserve all RGB pixels inside')
    cal = commands.add_parser('calibrate')
    cal.add_argument('--artifact', required=True)
    cal.add_argument('--manifest', required=True)
    cal.add_argument('--uncertainty-fraction', type=float, default=.05,
                     help='Abstention half-width as fraction of validation score range; not probability')
    test = commands.add_parser('evaluate', help='One final held-out test using frozen calibration')
    test.add_argument('--artifact', required=True)
    test.add_argument('--manifest', required=True,
                      help='Original manifest or unchanged original rows plus appended unseen test rows')
    test.add_argument('--parity-atol', type=float, default=.1)
    test.add_argument('--parity-rtol', type=float, default=.01)
    infer = commands.add_parser('infer')
    infer.add_argument('--artifact', required=True)
    infer.add_argument('--image', required=True)
    infer.add_argument('--device', default='CPU')
    infer.add_argument('--precision', choices=['f32', 'f16'])
    infer.add_argument('--map-output', required=True, help='Save raw anomaly map as .npy')
    bench = commands.add_parser('benchmark')
    bench.add_argument('--artifact', required=True)
    bench.add_argument('--image', required=True)
    bench.add_argument('--devices', nargs='+', default=['CPU', 'GPU', 'NPU'])
    bench.add_argument('--iterations', type=int, default=10)
    bench.add_argument('--precision', choices=['f32', 'f16'])
    bench.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'validate':
        result = validate_manifest(args.manifest, engineering_smoke=args.engineering_smoke)
    elif args.command == 'fit':
        result = fit(args.manifest, args.output, engineering_smoke=args.engineering_smoke,
                     image_size=args.image_size, sampling_ratio=args.sampling_ratio,
                     max_images=args.max_images, roi=args.roi, foreground_crop=args.foreground_crop,
                     num_neighbors=args.num_neighbors)
    elif args.command == 'calibrate':
        result = calibrate(args.artifact, args.manifest, uncertainty_fraction=args.uncertainty_fraction)
    elif args.command == 'evaluate':
        result = evaluate(args.artifact, args.manifest, parity_atol=args.parity_atol, parity_rtol=args.parity_rtol)
    elif args.command == 'infer':
        import numpy as np
        result = AnomalyDetector(args.artifact, device=args.device, precision=args.precision).infer(args.image)
        np.save(args.map_output, result.pop('anomaly_map'))
        result['anomaly_map_path'] = str(Path(args.map_output).resolve())
    else:
        result = benchmark(args.artifact, args.image, devices=tuple(args.devices), iterations=args.iterations, precision=args.precision)
        Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
