import unittest

from arabic_ocr._bidi import fix_trailing_period, normalize_chars, normalize_digits, visual_to_logical


class VisualToLogicalTest(unittest.TestCase):
    def test_pure_arabic_is_reversed(self):
        self.assertEqual(visual_to_logical("دوكلا"), "الكود")

    def test_number_stays_ltr_and_moves_to_logical_end(self):
        # visual (as drawn): number on the left, label on the right
        self.assertEqual(visual_to_logical("٤٦١٠٠٢ : دوكلا"), "الكود : ٤٦١٠٠٢")

    def test_date_with_separators(self):
        self.assertEqual(visual_to_logical("٢٠١٢/٠٦/٢٥ : طاشنلا ءدب خيرات"), "تاريخ بدء النشاط : ٢٠١٢/٠٦/٢٥")

    def test_latin_words_keep_order(self):
        self.assertEqual(visual_to_logical("Hello World ابحرم"), "مرحبا Hello World")

    def test_two_numbers_swap(self):
        # logical "١٢ ٣٤" is displayed as "٣٤ ١٢"
        self.assertEqual(visual_to_logical("٣٤ ١٢"), "١٢ ٣٤")

    def test_brackets_are_mirrored_back(self):
        self.assertEqual(visual_to_logical("(صن)"), "(نص)")

    def test_decimal_and_percent(self):
        self.assertEqual(visual_to_logical("%12.5 ةبسن"), "نسبة 12.5%")

    def test_empty(self):
        self.assertEqual(visual_to_logical(""), "")


class NormalizeTest(unittest.TestCase):
    def test_mixed_digit_run_uses_majority_system(self):
        self.assertEqual(normalize_digits("٤٦١00٢"), "٤٦١٠٠٢")
        self.assertEqual(normalize_digits("2012/٠6/25"), "2012/06/25")

    def test_pure_runs_untouched(self):
        self.assertEqual(normalize_digits("abc 123 ٤٥٦"), "abc 123 ٤٥٦")

    def test_trailing_period_read_as_zero(self):
        self.assertEqual(fix_trailing_period("الكود : ٠٤٦١٠٠٢"), "الكود : ٤٦١٠٠٢.")
        self.assertEqual(fix_trailing_period("التاريخ ٠٢٠١٢/٠٦/٢٥"), "التاريخ ٢٠١٢/٠٦/٢٥.")
        self.assertEqual(fix_trailing_period("٠١٢٣٤٥٦٧٨٩٠"), "٠١٢٣٤٥٦٧٨٩٠")  # plain number: untouched
        self.assertEqual(fix_trailing_period("٠٢٠١٢/٠٦/٢٥"), "٢٠١٢/٠٦/٢٥.")  # date-only line
        self.assertEqual(fix_trailing_period("هاتف ٠١٢٣٤٥٦٧٨٩٠"), "هاتف ٠١٢٣٤٥٦٧٨٩٠")  # 11 digits: phone number

    def test_persian_forms(self):
        self.assertEqual(normalize_chars("الکود ۲۰۱۲"), "الكود ٢٠١٢")


if __name__ == "__main__":
    unittest.main()
