from __future__ import annotations

import argparse
import sys


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Unified deployment entrypoint for real-hand deploy modes."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["server", "client", "test-grasp", "replay"],
        help="Deployment mode: server, client, test-grasp, or replay.",
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
    if args.mode == "server":
        from isaacgymenvs.deploy._server_impl import main as server_main

        server_main()
        return
    if args.mode == "client":
        from isaacgymenvs.deploy._client_impl import main as client_main

        client_main()
        return
    if args.mode == "test-grasp":
        from isaacgymenvs.deploy.load_selected_grasp_real import main as test_grasp_main

        test_grasp_main()
        return
    if args.mode == "replay":
        from isaacgymenvs.deploy.replay_cur_targets_real import main as replay_main

        replay_main()
        return
    raise ValueError(f"Unsupported deploy mode: {args.mode!r}")


if __name__ == "__main__":
    main()
