"""Command-line entry point for ApexView.

THIN CLIENT. This module orchestrates and presents; it performs no image
analysis of its own. It reads files through
:func:`apexview.io.dicom_reader.load_dicom` and classifies pairs through
:func:`apexview.engine.pair_classifier.classify_pair`, then formats whatever
those return for the terminal.

Every number printed here comes from the objects the engine and reader already
produced (``RadiographImage``, ``ClassificationResult``). The CLI never runs
SIFT, computes a homography, recomputes an inlier count or reprojection error,
re-derives an 8-bit image, or fabricates a pixel spacing. It deliberately does
NOT import cv2: if presentation code needed OpenCV it would mean computation
had leaked out of the engine. PNG saving of an engine-produced array is done
with Pillow, which is image I/O, not image analysis.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Sequence

from apexview.engine.pair_classifier import (
    ClassificationResult,
    PairType,
    classify_pair,
)
from apexview.io.dicom_reader import (
    InvalidDicomError,
    RadiographImage,
    load_dicom,
)

_PROG = "apexview"


class _UserError(Exception):
    """A user-facing error: prints a clean message to stderr, never a
    traceback. Carries the process exit code to return."""

    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.code = code


def _load(path: str) -> RadiographImage:
    """Load a DICOM, converting reader exceptions into clean _UserErrors."""
    try:
        return load_dicom(path)
    except FileNotFoundError:
        raise _UserError(f"file not found: {path}")
    except InvalidDicomError as exc:
        raise _UserError(f"not a valid DICOM file: {path} ({exc})")


def _format_spacing(img: RadiographImage) -> str:
    """Render the pixel-spacing status of an image. Honest about absence:
    never substitutes a default when the DICOM carried no calibration."""
    if img.pixel_spacing_mm is None:
        return (
            "UNAVAILABLE (this radiograph carries no calibration; "
            "real-world measurements are not possible without it)"
        )
    row, col = img.pixel_spacing_mm
    return f"{row:.3f} mm x {col:.3f} mm  (from {img.pixel_spacing_source})"


def _format_reproj(value: float) -> str:
    if math.isnan(value):
        return "n/a (matching failed)"
    return f"{value:.3f} px (mean)"


def _save_png(array, path: str) -> None:
    """Write an engine-produced uint8 grayscale array to a PNG.

    Pillow is used purely as an image-file encoder; the array is produced by
    the engine, not here. cv2 is intentionally avoided in this module.
    """
    from PIL import Image

    Image.fromarray(array).save(path)


def _run_inspect(path: str) -> int:
    img = _load(path)
    rows, cols = img.pixels_raw.shape
    raw_min = int(img.pixels_raw.min())
    raw_max = int(img.pixels_raw.max())

    print(f"{_PROG} — DICOM inspection")
    print(f"  File:          {path}")
    print(f"  Dimensions:    {rows} x {cols} (rows x cols)")
    print(f"  Bit depth:     {img.bit_depth} bits")
    print(f"  Pixel spacing: {_format_spacing(img)}")
    print(f"  Intensity:     min {raw_min}, max {raw_max} (raw array)")
    return 0


def _run_analyze(file_a: str, file_b: str, out: str | None) -> int:
    img_a = _load(file_a)
    img_b = _load(file_b)

    try:
        result: ClassificationResult = classify_pair(
            img_a.pixels_u8, img_b.pixels_u8
        )
    except ValueError as exc:
        raise _UserError(f"invalid image data for analysis: {exc}")

    print(f"{_PROG} — pair analysis")
    print(f"  File A: {file_a}")
    print(f"  File B: {file_b}")
    print()
    print(f"  Verdict:        {result.pair_type.name}")
    print(f"  Inlier count:   {result.inlier_count}")
    print(f"  Reprojection:   {_format_reproj(result.mean_reprojection_error)}")
    print(f"  Engine message: {result.message}")
    print()
    print("  Pixel spacing:")
    print(f"    File A: {_format_spacing(img_a)}")
    print(f"    File B: {_format_spacing(img_b)}")
    print()

    if result.pair_type is PairType.EXTENSION and result.stitched_image is not None:
        if out is not None:
            _save_png(result.stitched_image, out)
            print(f"  Stitched image saved to: {out}")
        else:
            print(
                "  Stitched image available — pass --out <path> to save it "
                "(not dumped to the terminal)."
            )
    else:
        print(
            "  No stitched image (angulation; correction not yet implemented)."
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description=(
            "ApexView: headless dental radiograph engine. Inspect a single "
            "DICOM, or analyze two DICOMs to classify them as an extension "
            "pair (stitchable) or an angulation pair."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    inspect_p = subparsers.add_parser(
        "inspect",
        help="Load one DICOM and report its contents (dimensions, bit depth, "
        "pixel spacing, intensity range).",
    )
    inspect_p.add_argument("path", help="Path to a single DICOM file.")

    analyze_p = subparsers.add_parser(
        "analyze",
        help="Load two DICOMs and classify them as EXTENSION or ANGULATION.",
    )
    analyze_p.add_argument("file_a", help="First DICOM file (image A).")
    analyze_p.add_argument("file_b", help="Second DICOM file (image B).")
    analyze_p.add_argument(
        "--out",
        default=None,
        help="If the pair is an extension, save the stitched PNG to this path.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help(sys.stderr)
        return 2

    try:
        if args.command == "inspect":
            return _run_inspect(args.path)
        if args.command == "analyze":
            return _run_analyze(args.file_a, args.file_b, args.out)
    except _UserError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.code
    except Exception as exc:  # top-level safety net, not a bare except
        print(f"unexpected error: {exc}", file=sys.stderr)
        return 3

    # Unreachable: argparse only yields known subcommands.
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
