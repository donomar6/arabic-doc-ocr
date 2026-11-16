import unittest

import numpy as np

from arabic_ocr._layout import group_lines


def box(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


class GroupLinesTest(unittest.TestCase):
    def test_lines_top_to_bottom_and_rtl_inside(self):
        boxes = [box(0, 50, 100, 70), box(300, 10, 400, 30), box(0, 12, 100, 32), box(150, 48, 250, 72)]
        lines = group_lines(boxes, rtl=True)
        self.assertEqual(lines, [[1, 2], [3, 0]])

    def test_empty(self):
        self.assertEqual(group_lines([]), [])


if __name__ == "__main__":
    unittest.main()
