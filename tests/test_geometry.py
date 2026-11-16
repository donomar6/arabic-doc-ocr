import unittest

import numpy as np
from PIL import Image

from arabic_ocr import _geometry as geo


class ComponentsTest(unittest.TestCase):
    def test_two_blobs_and_diagonal_connectivity(self):
        m = np.zeros((10, 10), dtype=bool)
        m[1:3, 1:4] = True  # blob A
        m[3, 4] = True  # touches A diagonally -> same component (8-connectivity)
        m[6:9, 6:9] = True  # blob B
        comps = geo.connected_components(m)
        self.assertEqual(len(comps), 2)
        areas = sorted(c.area for c in comps)
        self.assertEqual(areas, [7, 9])

    def test_empty(self):
        self.assertEqual(geo.connected_components(np.zeros((5, 5), bool)), [])

    def test_dilate(self):
        m = np.zeros((4, 4), bool)
        m[1, 1] = True
        d = geo.dilate2x2(m)
        self.assertTrue(d[1, 1] and d[1, 2] and d[2, 1] and d[2, 2])
        self.assertEqual(int(d.sum()), 4)


class RectTest(unittest.TestCase):
    def test_axis_aligned(self):
        pts = np.array([[0, 0], [10, 0], [10, 4], [0, 4]], dtype=float)
        r = geo.min_area_rect(pts)
        self.assertAlmostEqual(sorted([r.w, r.h])[0], 4.0, places=6)
        self.assertAlmostEqual(sorted([r.w, r.h])[1], 10.0, places=6)
        box = geo.order_box(r.corners())
        np.testing.assert_allclose(box, [[0, 0], [10, 0], [10, 4], [0, 4]], atol=1e-6)

    def test_rotated(self):
        rng = np.random.default_rng(0)
        base = np.stack([rng.uniform(0, 20, 200), rng.uniform(0, 5, 200)], 1)
        ang = np.deg2rad(30)
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        pts = base @ rot.T + 50
        r = geo.min_area_rect(pts)
        self.assertLess(r.short_side, 5.2)
        self.assertLess(max(r.w, r.h), 20.5)
        self.assertGreater(max(r.w, r.h), 19.0)

    def test_expand_and_mask(self):
        r = geo.Rect((5, 5), (1, 0), (0, 1), 4, 2)
        big = r.expanded(1)
        self.assertEqual((big.w, big.h), (6, 4))
        prob = np.zeros((11, 11), dtype=np.float32)
        prob[4:7, 3:8] = 1.0
        self.assertGreater(geo.mean_in_polygon(prob, geo.order_box(r.corners())), 0.9)
        area, perim = geo.polygon_area_perimeter(geo.order_box(r.corners()))
        self.assertAlmostEqual(area, 8.0)
        self.assertAlmostEqual(perim, 12.0)


class CropTest(unittest.TestCase):
    def test_axis_aligned_crop_matches_slicing(self):
        arr = (np.random.default_rng(1).random((60, 80, 3)) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        box = np.array([[10, 5], [50, 5], [50, 25], [10, 25]])
        crop = np.asarray(geo.crop_quad(img, box))
        self.assertEqual(crop.shape, (20, 40, 3))
        ref = arr[5:25, 10:50]
        self.assertLess(np.abs(crop.astype(int) - ref.astype(int)).mean(), 3.0)

    def test_tall_crop_is_rotated(self):
        img = Image.new("RGB", (100, 100))
        crop = geo.crop_quad(img, np.array([[10, 10], [20, 10], [20, 60], [10, 60]]))
        self.assertEqual(crop.size, (50, 10))


class SplitTest(unittest.TestCase):
    def test_wide_crop_is_split_at_gaps(self):
        from arabic_ocr._split import split_wide_crop

        img = Image.new("RGB", (1200, 40), "white")
        px = img.load()
        # five "words" of 200 px separated by 50 px of white
        for w in range(5):
            for x in range(w * 250, w * 250 + 200):
                for y in range(8, 32):
                    px[x, y] = (0, 0, 0)
        pieces = split_wide_crop(img, max_ratio=8.0)
        self.assertGreater(len(pieces), 1)
        self.assertEqual(sum(p.width for p in pieces), 1200)
        for p in pieces:
            self.assertLessEqual(p.width / p.height, 8.0 + 1e-9)
        # cuts land in white space: columns on both sides of every cut are white
        for left, right in zip(pieces[:-1], pieces[1:]):
            self.assertEqual(int(np.asarray(left.convert("L"))[:, -1].min()), 255)
            self.assertEqual(int(np.asarray(right.convert("L"))[:, 0].min()), 255)

    def test_narrow_crop_untouched(self):
        from arabic_ocr._split import split_wide_crop

        img = Image.new("RGB", (200, 40), "white")
        self.assertEqual(len(split_wide_crop(img)), 1)


class EditDistanceTest(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(geo.edit_distance("kitten", "sitting"), 3)
        self.assertEqual(geo.edit_distance("", "abc"), 3)
        self.assertEqual(geo.edit_distance("abc", "abc"), 0)


if __name__ == "__main__":
    unittest.main()
