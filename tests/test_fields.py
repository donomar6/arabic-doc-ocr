import datetime as dt
import unittest

from arabic_ocr.fields import Field, best_match, normalize_arabic, parse_date, repair_code, validate, _match_label


class ValidatorTest(unittest.TestCase):
    def test_dates(self):
        self.assertEqual(parse_date("٢٠١٢/٠٦/٢٥"), dt.date(2012, 6, 25))
        self.assertEqual(parse_date("25/06/2012"), dt.date(2012, 6, 25))
        self.assertEqual(parse_date("2012-06-25"), dt.date(2012, 6, 25))
        self.assertIsNone(parse_date("٢٠١٢٠٦٢٥"))
        self.assertIsNone(parse_date("٣٥/٠٦/٢٠١٢"))

    def test_code(self):
        fld = Field("code", ["الكود"], kind="code", length=6)
        self.assertEqual(validate(fld, "٤٦١٠٠٢.")[:2], ("461002", True))
        self.assertEqual(validate(fld, "٤٦١٠٠")[1], False)

    def test_code_table_repair(self):
        table = {"461002": {}, "461006": {}}
        self.assertEqual(repair_code("461007", table), None)  # 7->1 gives 461001, not in table
        self.assertEqual(repair_code("461003", table), "461002")  # 3->2 gives a known code
        self.assertEqual(repair_code("461009", {"461002": {}, "461006": {}}), "461006")  # 9->6
        fld = Field("code", ["الكود"], kind="code", length=6)
        value, valid, reasons = validate(fld, "٤٦١٠٠٢", table)
        self.assertEqual((value, valid), ("461002", True))
        value, valid, reasons = validate(fld, "٩٩٩٩٩٩", table)
        self.assertEqual(valid, False)

    def test_name_and_digits(self):
        self.assertFalse(validate(Field("n", ["x"], kind="name"), "محمد ٣")[1])
        self.assertTrue(validate(Field("n", ["x"], kind="name"), "محمد عبد الرحمن")[1])
        self.assertEqual(validate(Field("t", ["x"], kind="digits", length=9), "300012345")[1], True)

    def test_vocabulary(self):
        vocab = ["مكاتب التصدير والاستيراد", "التسويق التعاوني للمنتجات الزراعية"]
        self.assertEqual(best_match("مكاتب التصدير والأستيراد", vocab)[0], vocab[0])
        self.assertIsNone(best_match("شيء آخر تماما", vocab)[0])

    def test_label_match(self):
        self.assertEqual(_match_label("الكود : ٤٦١٠٠٢.", ["الكود"]), ("الكود", "٤٦١٠٠٢."))
        self.assertEqual(_match_label("تاريخ بده التشاط :", ["تاريخ بدء النشاط"])[1], "")
        self.assertIsNone(_match_label("نص عادي", ["الكود"]))

    def test_normalize(self):
        self.assertEqual(normalize_arabic("الأنشطة الإضافية"), "الانشطه الاضافيه")


if __name__ == "__main__":
    unittest.main()
