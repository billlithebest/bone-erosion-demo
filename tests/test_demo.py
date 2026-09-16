import tempfile
import unittest
from pathlib import Path
import numpy as np
from erosion_demo import measure_components, segment_defects, phantom, dice, write_nrrd, run


class PipelineTests(unittest.TestCase):
    def test_volume_and_axis_order(self):
        mask = np.zeros((5, 6, 7), bool)
        mask[1:3, 2:4, 3:5] = True
        _, rows = measure_components(mask, (0.2, 0.3, 0.5), 1)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]['volume_mm3'], 8 * 0.2 * 0.3 * 0.5)
        self.assertAlmostEqual(rows[0]['centroid_x_mm'], 3.5 * 0.2)
        self.assertAlmostEqual(rows[0]['centroid_z_mm'], 1.5 * 0.5)

    def test_connectivity_and_filter(self):
        mask = np.zeros((5, 5, 5), bool)
        mask[0, 0, 0] = mask[1, 1, 1] = mask[1, 1, 2] = True
        labels, rows = measure_components(mask, min_voxels=2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(np.count_nonzero(labels), 2)
        self.assertEqual(labels[0, 0, 0], 0)

    def test_empty_and_invalid(self):
        mask = np.zeros((2, 2, 2), bool)
        self.assertEqual(measure_components(mask)[1], [])
        self.assertEqual(dice(mask, mask), 1.)
        for spacing in ((0, 1, 1), (-1, 1, 1), (1, 2), (1, np.nan, 1)):
            with self.assertRaises(ValueError):
                measure_components(mask, spacing)
        with self.assertRaises(ValueError):
            segment_defects(np.zeros((3, 3)), mask)

    def test_phantom_truth_and_growth(self):
        images, reference, truths = phantom()
        for image, truth in zip(images, truths):
            np.testing.assert_array_equal(segment_defects(image, reference), truth)
        self.assertGreater(truths[1].sum(), truths[0].sum())
        np.testing.assert_array_equal(images[0], phantom()[0][0])

    def test_nrrd_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.nrrd'
            array = np.arange(24).reshape(2, 3, 4)
            write_nrrd(path, array, (0.2, 0.3, 0.5))
            header, raw = path.read_bytes().split(b'\n\n', 1)
            self.assertIn(b'sizes: 4 3 2', header)
            self.assertIn(b'(0.2,0,0) (0,0.3,0) (0,0,0.5)', header)
            np.testing.assert_array_equal(np.frombuffer(raw, '<f4').reshape(2, 3, 4), array)

    def test_end_to_end_and_empty_report(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run(folder)
            self.assertGreater(result['change_mm3'], 0)
            self.assertTrue((Path(folder) / 'report.html').exists())
            empty = run(folder, threshold=-1e6)
            self.assertIsNone(empty['change_percent'])
            self.assertEqual(len((Path(folder) / 'measurements.csv').read_text().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
