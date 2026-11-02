"""OCR backend adapter — isolates the rest of the pipeline from the engine.

Exposes one small surface:

    engine = build_ocr_engine(prefer="paddle", device="cpu")
    results = engine.read(image_bgr)   # -> list[OcrLine]

`OcrLine` is (text, confidence, box) where box is [x1, y1, x2, y2] in the
coordinate space of the image passed in.

Two backends are supported and auto-selected:
  * PaddleOCR (preferred per project decision) — PP-OCRv5, CPU by default.
  * EasyOCR  (fallback) — torch-based, reuses the inherited GPU build.

The PaddleOCR 3.x API changed shape across minor versions, so the adapter tries
the modern `.predict()` first and falls back to `.ocr()`, normalizing either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class OcrLine:
    text: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2]


def _poly_to_xyxy(poly) -> List[float]:
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


class _PaddleBackend:
    name = "paddleocr"

    def __init__(self, lang: str = "en"):
        from paddleocr import PaddleOCR
        # Keep init tolerant across 3.x minor versions: only pass widely-supported args.
        try:
            self._ocr = PaddleOCR(lang=lang, use_textline_orientation=True)
        except TypeError:
            self._ocr = PaddleOCR(lang=lang)

    def read(self, image_bgr) -> List[OcrLine]:
        # Prefer modern predict(); fall back to legacy ocr().
        try:
            out = self._ocr.predict(image_bgr)
            return self._parse_predict(out)
        except (AttributeError, TypeError):
            out = self._ocr.ocr(image_bgr)
            return self._parse_ocr(out)

    @staticmethod
    def _parse_predict(out) -> List[OcrLine]:
        lines: List[OcrLine] = []
        if not out:
            return lines
        for res in out:
            d = res.get("res", res) if isinstance(res, dict) else getattr(res, "res", res)
            texts = d.get("rec_texts") or d.get("rec_text") or []
            scores = d.get("rec_scores") or d.get("rec_score") or []
            polys = d.get("rec_polys") or d.get("dt_polys") or d.get("rec_boxes") or []
            for i, t in enumerate(texts):
                sc = float(scores[i]) if i < len(scores) else 0.0
                box = _poly_to_xyxy(polys[i]) if i < len(polys) else [0, 0, 0, 0]
                if t and t.strip():
                    lines.append(OcrLine(t.strip(), sc, box))
        return lines

    @staticmethod
    def _parse_ocr(out) -> List[OcrLine]:
        lines: List[OcrLine] = []
        if not out:
            return lines
        page = out[0] if out and isinstance(out[0], list) else out
        for item in page or []:
            try:
                poly, (text, score) = item
            except (ValueError, TypeError):
                continue
            if text and text.strip():
                lines.append(OcrLine(text.strip(), float(score), _poly_to_xyxy(poly)))
        return lines


class _EasyBackend:
    name = "easyocr"

    def __init__(self, lang: str = "en", gpu: bool = True):
        import easyocr
        # verbose=False suppresses the Unicode progress bar that crashes on the
        # Windows cp1252 console (the solid-block char can't be encoded there).
        self._reader = easyocr.Reader([lang], gpu=gpu, verbose=False)

    def read(self, image_bgr) -> List[OcrLine]:
        out = self._reader.readtext(image_bgr)
        lines: List[OcrLine] = []
        for box, text, conf in out:
            if text and text.strip():
                lines.append(OcrLine(text.strip(), float(conf), _poly_to_xyxy(box)))
        return lines


def build_ocr_engine(prefer: str = "paddle", device: str = "cpu"):
    """Return the first OCR backend that imports successfully.

    prefer: "paddle" or "easy". device: "cpu"/"gpu" (only affects EasyOCR;
    PaddleOCR here is intentionally CPU to stay isolated from the torch/CUDA env).
    """
    order = ["paddle", "easy"] if prefer == "paddle" else ["easy", "paddle"]
    errors = {}
    for backend in order:
        try:
            if backend == "paddle":
                eng = _PaddleBackend()
            else:
                eng = _EasyBackend(gpu=(device != "cpu"))
            print(f"[ocr] using backend: {eng.name}")
            return eng
        except Exception as e:  # noqa: BLE001 - want to try the next backend
            errors[backend] = repr(e)
            print(f"[ocr] backend '{backend}' unavailable: {e}")
    raise RuntimeError(f"No OCR backend available. Tried: {errors}")
