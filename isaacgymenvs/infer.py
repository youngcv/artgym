from __future__ import annotations

import argparse
import sys


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Unified inference entrypoint for teacher and distilled student policies."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["teacher", "student"],
        help="Inference mode: teacher or student.",
    )
    return parser


def main():
    parser = _build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return

    if sys.argv[1] in {"-h", "--help"}:
        parser.parse_args()
        return

    args = parser.parse_args(sys.argv[1:2])
    remainder = sys.argv[2:]
    sys.argv = [sys.argv[0]] + remainder
    if args.mode == "teacher":
        from isaacgymenvs.infer_teacher_impl import main as teacher_main

        teacher_main()
        return
    if args.mode == "student":
        from isaacgymenvs.infer_student_impl import main as student_main

        student_main()
        return
    raise ValueError(f"Unsupported infer mode: {args.mode!r}")


if __name__ == "__main__":
    main()
