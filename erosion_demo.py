"""Synthetic longitudinal bone-defect quantification. Educational, not clinical."""
from __future__ import annotations

import argparse
from collections import deque
import csv
import html
import json
from pathlib import Path
import platform

import numpy as np


def validate_spacing(spacing):
    values = np.asarray(spacing, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('spacing must contain three finite positive values in x,y,z order')
    return values


def phantom(seed=7):
    """Return aligned z,y,x images and known ideal bone/defect masks."""
    z, y, x = np.indices((64, 64, 64))
    ideal = ((x - 32) / 21)**2 + ((y - 32) / 18)**2 + ((z - 32) / 25)**2 <= 1
    rng = np.random.default_rng(seed)
    images, truths = [], []
    for radius in (4, 6):
        defect = ideal & (((x - 50)**2 + (y - 32)**2 + (z - 32)**2) <= radius**2)
        defect |= ideal & (((x - 19)**2 + (y - 23)**2 + (z - 39)**2) <= 3**2)
        bone = ideal & ~defect
        images.append(np.where(bone, 1000., 100.) + rng.normal(0, 35, ideal.shape))
        truths.append(defect)
    return images, ideal, truths


def segment_defects(image, reference, threshold=550.):
    """Missing high-intensity material inside a supplied intact-bone reference."""
    image, reference = np.asarray(image), np.asarray(reference)
    if image.ndim != 3 or image.shape != reference.shape or reference.dtype != bool:
        raise ValueError('image must be 3D and reference must be a matching boolean mask')
    if not np.all(np.isfinite(image)) or not np.isfinite(threshold):
        raise ValueError('image and threshold must be finite')
    return reference & (image < threshold)


def measure_components(mask, spacing=(0.2, 0.2, 0.3), min_voxels=8):
    """6-connected components; volume mm³ and centroid xyz mm from origin zero."""
    spacing = validate_spacing(spacing)
    mask = np.asarray(mask)
    if mask.ndim != 3 or mask.dtype != bool:
        raise ValueError('mask must be a 3D boolean array')
    if not isinstance(min_voxels, int) or min_voxels < 1:
        raise ValueError('min_voxels must be a positive integer')
    labels = np.zeros(mask.shape, dtype=np.int32)
    visited = np.zeros(mask.shape, dtype=bool)
    rows = []
    for start_array in np.argwhere(mask):
        start = tuple(start_array)
        if visited[start]:
            continue
        visited[start] = True
        queue, points = deque([start]), []
        while queue:
            point = queue.popleft()
            points.append(point)
            for axis in range(3):
                for step in (-1, 1):
                    neighbor = list(point)
                    neighbor[axis] += step
                    neighbor = tuple(neighbor)
                    if (0 <= neighbor[axis] < mask.shape[axis]
                            and mask[neighbor] and not visited[neighbor]):
                        visited[neighbor] = True
                        queue.append(neighbor)
        if len(points) < min_voxels:
            continue
        coords = np.array(points)
        label = len(rows) + 1
        labels[tuple(coords.T)] = label
        centroid = coords.mean(axis=0)[::-1] * spacing
        rows.append(dict(component_id=label, voxels=len(points),
                         volume_mm3=float(len(points) * np.prod(spacing)),
                         centroid_x_mm=float(centroid[0]), centroid_y_mm=float(centroid[1]),
                         centroid_z_mm=float(centroid[2])))
    return labels, rows


def dice(a, b):
    total = int(a.sum() + b.sum())
    return 1.0 if total == 0 else float(2 * np.count_nonzero(a & b) / total)


def write_nrrd(path, array, spacing):
    """Write little-endian float32 NRRD; numpy zyx maps to NRRD xyz."""
    sx, sy, sz = validate_spacing(spacing)
    array = np.asarray(array, dtype='<f4')
    if array.ndim != 3:
        raise ValueError('NRRD export requires a 3D array')
    z, y, x = array.shape
    header = (f'NRRD0005\ntype: float\ndimension: 3\nsizes: {x} {y} {z}\n'
              'space: left-posterior-superior\n'
              f'space directions: ({sx},0,0) (0,{sy},0) (0,0,{sz})\n'
              'space origin: (0,0,0)\nspace units: "mm" "mm" "mm"\n'
              'kinds: domain domain domain\nencoding: raw\nendian: little\n\n')
    Path(path).write_bytes(header.encode('ascii') + array.tobytes(order='C'))


def slice_svg(image, mask, z=32):
    """Self-contained pixel visualization; red marks retained defect voxels."""
    pixels = []
    for y, x in np.ndindex(image.shape[1:]):
        value = int(np.clip(image[z, y, x] / 1200 * 255, 0, 255))
        color = '#ff657a' if mask[z, y, x] else f'rgb({value},{value},{value})'
        pixels.append(f'<rect x="{x}" y="{y}" width="1" height="1" fill="{color}"/>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" '
            'role="img" aria-label="Synthetic axial slice; defects highlighted in red">'
            + ''.join(pixels) + '</svg>')


def run(out, seed=7, threshold=550., min_voxels=8):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    spacing = (0.2, 0.2, 0.3)
    images, reference, truths = phantom(seed)
    results, all_rows, panels = [], [], []
    write_nrrd(out / 'reference.nrrd', reference, spacing)
    for name, image, truth in zip(('baseline', 'followup'), images, truths):
        candidates = segment_defects(image, reference, threshold)
        labels, rows = measure_components(candidates, spacing, min_voxels)
        retained = labels > 0
        volume = sum(row['volume_mm3'] for row in rows)
        results.append(dict(timepoint=name, total_volume_mm3=volume,
                            component_count=len(rows), dice_vs_synthetic_truth=dice(retained, truth),
                            truth_volume_mm3=float(truth.sum() * np.prod(spacing))))
        all_rows.extend(dict(timepoint=name, **row) for row in rows)
        write_nrrd(out / f'{name}.nrrd', image, spacing)
        write_nrrd(out / f'{name}_defects.nrrd', retained, spacing)
        svg = slice_svg(image, retained)
        (out / f'{name}.svg').write_text(svg)
        panels.append(f'<section><h2>{name.title()}</h2>{svg}<p>{volume:.3f} mm³ · {len(rows)} components</p></section>')
    base, follow = (r['total_volume_mm3'] for r in results)
    summary = dict(seed=seed, threshold=threshold, min_voxels=min_voxels,
                   spacing_xyz_mm=spacing, shape_zyx=list(reference.shape),
                   python=platform.python_version(), numpy=np.__version__,
                   results=results, change_mm3=follow-base,
                   change_percent=None if base == 0 else 100*(follow-base)/base)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    fields = ['timepoint', 'component_id', 'voxels', 'volume_mm3',
              'centroid_x_mm', 'centroid_y_mm', 'centroid_z_mm']
    with (out / 'measurements.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    report = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Bone erosion demo</title>
<style>body{font:17px system-ui;background:#101827;color:#e7edf7;max-width:960px;margin:40px auto;padding:20px}h1{font-size:40px}p{line-height:1.6;color:#bdcadf}.panels{display:flex;gap:24px}section{flex:1;background:#1a2639;padding:24px;border-radius:16px}svg{width:100%;image-rendering:pixelated}strong{color:#7ee0cf}@media(max-width:600px){.panels{display:block}}</style>
<h1>Measuring synthetic bone defects</h1><p>Python portfolio demo · two aligned 3D phantoms · no patient data</p>'''
    report += '<div class="panels">' + ''.join(panels) + '</div>'
    report += f'<p><strong>Total defect volume change: {follow-base:+.3f} mm³</strong></p>'
    report += '<p>Axial slice z=32. Red indicates detected defects; volumes use all 64 slices. An ideal intact-bone mask is supplied by the simulator. This is not automatic clinical erosion detection. Images are already aligned; registration is not implemented.</p>'
    report += '<details><summary>Reproducibility and validation</summary><pre>' + html.escape(json.dumps(summary, indent=2)) + '</pre></details></html>'
    (out / 'report.html').write_text(report)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('demo_output'))
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--threshold', type=float, default=550.)
    parser.add_argument('--min-voxels', type=int, default=8)
    args = parser.parse_args()
    if not np.isfinite(args.threshold) or args.min_voxels < 1 or args.seed < 0:
        parser.error('use a finite threshold, positive min-voxels, and nonnegative seed')
    result = run(args.out, args.seed, args.threshold, args.min_voxels)
    print(f"Report: {args.out / 'report.html'}\nVolume change: {result['change_mm3']:+.3f} mm³")


if __name__ == '__main__':
    main()
