# Third-party notices

`arabic-doc-ocr` redistributes the following third-party components inside the
wheel. Their full license texts are shipped in the package:

| Component | Version | License | Location in the wheel |
|---|---|---|---|
| ONNX Runtime (CPU) | see `arabic_ocr/_vendor/onnxruntime/VENDORED.txt` | MIT | `arabic_ocr/_vendor/onnxruntime/LICENSE`, `ThirdPartyNotices.txt`, `Privacy.md`; copies in `arabic_ocr/_vendor/licenses/onnxruntime/` |
| PP-OCRv4 text detection model (`ch_PP-OCRv4_det_infer.onnx`) | PP-OCRv4 mobile | Apache-2.0 | `arabic_ocr/models/LICENSE`, `arabic_ocr/models/NOTICE.txt` |
| PP-OCRv5 Arabic recognition model (`arabic_PP-OCRv5_rec_infer.onnx`) + `arabic_dict_v5.txt` | PP-OCRv5 mobile | Apache-2.0 | `arabic_ocr/models/LICENSE`, `arabic_ocr/models/NOTICE.txt` |
| PP-OCRv4 Arabic recognition model (`arabic_PP-OCRv4_rec_infer.onnx`) + `arabic_dict.txt` | PP-OCRv4 mobile | Apache-2.0 | `arabic_ocr/models/LICENSE`, `arabic_ocr/models/NOTICE.txt` |

## ONNX Runtime

Copyright (c) Microsoft Corporation. Licensed under the MIT License.
https://github.com/microsoft/onnxruntime

The copy in `arabic_ocr/_vendor/onnxruntime` is trimmed to the inference
parts, its imports are rewritten to the private `arabic_ocr._vendor` namespace,
and the pybind11 internals identifier inside the extension module is renamed
so the copy can coexist with any onnxruntime the user installs. No other
changes were made. ONNX Runtime's own third-party notices (protobuf, Eigen,
abseil, pybind11, ...) are in `ThirdPartyNotices.txt`.

## OCR models

The ONNX models were converted and published by the RapidOCR project
(RapidAI, https://github.com/RapidAI/RapidOCR, model repository
https://www.modelscope.cn/models/RapidAI/RapidOCR) from the PP-OCR models
trained and released by PaddleOCR (https://github.com/PaddlePaddle/PaddleOCR).
Both projects are licensed under the Apache License, Version 2.0. The models
are redistributed unchanged; file hashes are listed in
`arabic_ocr/models/NOTICE.txt`.

## Re-implemented code

The DB detection post-processing, CTC decoding, crop rectification and line
ordering in `arabic_ocr` are new implementations in numpy/Pillow that follow
the algorithms of RapidOCR / PaddleOCR (Apache-2.0). No OpenCV, Shapely or
pyclipper code is included.
