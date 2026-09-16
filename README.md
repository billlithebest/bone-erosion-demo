# Bone Erosion Demo

A small Python project that quantifies simulated bone defects in two aligned 3D images. An educational example of research software development in musculoskeletal imaging.

![Baseline axial slice](examples/baseline.svg)

**Demo result:** baseline defect volume **4.308 mm³**, follow-up **10.236 mm³**, change **+5.928 mm³**. Open `examples/report.html` in a browser for the side-by-side report. Red overlays show defects on one axial slice; measurements use the whole volume.

## Quick start

Requires Python 3.10 or newer. NumPy is the only external dependency.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python erosion_demo.py
python -m unittest discover -s tests -v
```

On Windows, activate with `.venv\Scripts\activate` instead. Open `demo_output/report.html` in your browser. The checked-in `examples/report.html` works without installing anything.

Try changing the threshold or component size filter:

```bash
python erosion_demo.py --threshold 950 --min-voxels 20 --out experiment
```

A high threshold deliberately misclassifies some bone voxels; inspect how the measurements and Dice score change. Output folders are overwritten when reused.

## How it works

1. Generate a 64 × 64 × 64 ellipsoid representing ideal intact bone.
2. Remove two spherical regions, one increasing in radius between timepoints. Add seeded Gaussian intensity noise to each image. These intensities are arbitrary, not calibrated Hounsfield units.
3. Classify voxels below a threshold as missing bone **inside the supplied ideal reference mask**. This reference comes from the simulator, not from an automatic reconstruction method.
4. Group candidates using 6-connectivity (face-sharing neighbors), implemented with a queue. Discard components smaller than a configurable voxel count.
5. Compute each component's volume and centroid. Compare total defect volume between timepoints and evaluate masks against synthetic ground truth.
6. Export an HTML visual report, CSV table, JSON parameters/results, and NRRD images and masks.

Volume = voxel count × spacing_x × spacing_y × spacing_z. Here spacing is (0.2, 0.2, 0.3) mm, so each voxel occupies 0.012 mm³. Arrays use **z,y,x** order; spacing and reported centroids use **x,y,z**. NRRD exports use a zero origin with axis-aligned LPS coordinates. Slice plots use array coordinates, not radiological display conventions.

Component IDs are local to each timepoint and are **not longitudinal lesion matches**. The reported change compares total volumes. Percentage change is null when the baseline is zero.

## Files

| File | Purpose |
| --- | --- |
| `erosion_demo.py` | Synthetic data, segmentation, measurement, exports, command line |
| `tests/test_demo.py` | Six tests covering physical units, axes, connectivity, filtering, errors, reproducibility, exports and end-to-end execution |
| `examples/report.html` | Ready-to-open visual report |
| `examples/measurements.csv` | Per-component measurements |
| `examples/summary.json` | Parameters, versions, ground-truth comparison and total change |
| `examples/*.nrrd` | Volumes for external image viewers |
| `.github/workflows/tests.yml` | GitHub Actions test workflow for Python 3.10 and 3.12 |

## View in 3D Slicer

1. Open 3D Slicer and use **File → Add Data** to load `examples/baseline.nrrd` and `examples/followup.nrrd`.
2. Load `examples/baseline_defects.nrrd` or `examples/followup_defects.nrrd` as a segmentation using the file-type option in the Add Data dialog; expand options if needed.
3. Compare slices and inspect the defect overlay. The two volumes share the same grid and physical coordinates.

The project writes standard NRRD files; it is not a Slicer extension and does not use ITK or VTK. NRRD headers and binary axis ordering are tested locally. Interactive import into Slicer has not been tested in this environment.

## Validation and limits

The default run produces two components at each timepoint, with Dice = 1.0 against the generated defect masks. This perfect score reflects a deliberately simple phantom with widely separated intensity distributions and a known reference. It is a correctness check, not evidence of real-world accuracy.

No patient data, real anatomy, clinical validation, registration, DICOM import, partial-volume modeling, or motion correction is included. A solid ellipsoid does not model cortical/trabecular anatomy or marrow. Missing material relative to a known intact reference is a simplified defect definition, not a clinical erosion criterion. Both images are already aligned; real longitudinal measurements require registration and review of segmentation quality.

A useful next step would be a Slicer scripted module, followed by testing SimpleITK registration on deliberately shifted phantoms and assessing threshold sensitivity. Real-image evaluation would require appropriate datasets and expert reference annotations.

## References

- [Manske Lab](https://www.manskelab.ca/): research context; the lab describes CT, MR and ultrasound work on musculoskeletal health. This independent demo is not affiliated with the lab.
- [3D Slicer data loading documentation](https://slicer.readthedocs.io/en/latest/user_guide/data_loading_and_saving.html): NRRD support and loading workflow.

Local verification: Python 3.12.14 and NumPy 2.3.5; six tests passed. GitHub Actions also passed on Python 3.10 and 3.12.
