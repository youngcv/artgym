from __future__ import annotations

import argparse
import socketserver
import threading
import traceback

from isaacgymenvs.deploy.policy_rpc import recv_json_line, send_json_line


def parse_args():
    parser = argparse.ArgumentParser(description="Student policy inference server for sim-to-real deployment.")
    parser.add_argument("--student-artifact", required=True, help="Path to the distilled student artifact.")
    parser.add_argument("--checkpoint", default="", help="Optional teacher checkpoint override.")
    parser.add_argument("--task", default="artmanip", help="Task config name.")
    parser.add_argument("--train", default="artmanipSAPGPrivLSTMPPO", help="Train config name.")
    parser.add_argument("--hand", default="sharpa", help="Hand config name.")
    parser.add_argument("--object", default="knife", help="Object config name.")
    parser.add_argument("--asset-dir", default="", help="Optional asset directory under assets/objects, e.g. knife_multi.")
    parser.add_argument("--rl-device", default="cuda:0", help="Torch device used for inference.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--expl-block-idx", type=int, default=-1, help="Override the teacher exploration block when applicable.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=5555, help="Bind port.")
    return parser.parse_args()


class StudentPolicyTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, runtime):
        self.runtime = runtime
        super().__init__(server_address, StudentPolicyRequestHandler)



def _start_goal_input_thread(runtime):
    def _goal_input_loop():
        print("Interactive server goal input enabled.")
        if getattr(runtime, "goal_sequence", None):
            print("Press Enter to switch to the next configured goal in the server sequence.")
        print("Enter a float goal offset to set an explicit override.")
        print("Type 'clear' to remove the override, or 'quit' to stop the input thread.")
        while True:
            try:
                raw = input("server_goal_offset> ").strip()
            except EOFError:
                return
            except KeyboardInterrupt:
                print("\nServer goal input stopped.")
                return
            if raw == "":
                try:
                    goal_index, goal_offset = runtime.cycle_goal_override()
                except ValueError as exc:
                    print(str(exc))
                    continue
                print(f"Cycled server goal_override -> idx={goal_index} goal_offset={goal_offset:.6f}")
                continue
            lowered = raw.lower()
            if lowered in {"quit", "exit", "q"}:
                print("Server goal input stopped.")
                return
            if lowered in {"clear", "none", "null"}:
                runtime.clear_goal_override()
                print("Cleared server goal override.")
                continue
            try:
                goal_offset = float(raw)
            except ValueError:
                print(f"Invalid goal offset: {raw!r}. Please enter a float or 'clear'.")
                continue
            runtime.set_goal_override(goal_offset)
            print(f"Updated server goal_override to {goal_offset:.6f}")

    thread = threading.Thread(target=_goal_input_loop, name="server_goal_input_thread", daemon=True)
    thread.start()
    return thread


class StudentPolicyRequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        reader = self.connection.makefile("r", encoding="utf-8")
        writer = self.connection.makefile("w", encoding="utf-8")
        try:
            while True:
                try:
                    request = recv_json_line(reader)
                except EOFError:
                    break

                try:
                    response = self._dispatch(request)
                    send_json_line(writer, {"status": "ok", **response})
                except Exception as exc:
                    traceback.print_exc()
                    send_json_line(writer, {"status": "error", "error": str(exc)})
        finally:
            writer.close()
            reader.close()

    def _dispatch(self, request):
        request_type = request.get("type", "")
        if request_type == "ping":
            return {"message": "pong"}
        if request_type == "init_session":
            session_id = str(request.get("session_id", "default"))
            context = self.server.runtime.init_session(session_id)
            return {"session_id": session_id, "context": context}
        if request_type == "close_session":
            session_id = str(request.get("session_id", "default"))
            self.server.runtime.close_session(session_id)
            return {"session_id": session_id, "closed": True}
        if request_type == "infer":
            session_id = str(request.get("session_id", "default"))
            return self.server.runtime.infer(
                session_id=session_id,
                policy_obs=request["policy_obs"],
                student_obs=request["student_obs"],
                expl_features=request.get("expl_features"),
                deterministic=bool(request.get("deterministic", False)),
                reset_rnn=bool(request.get("reset_rnn", False)),
            )
        raise ValueError(f"Unsupported request type: {request_type!r}")


def main():
    args = parse_args()

    from isaacgymenvs.deploy.student_policy_runtime import load_student_policy_runtime

    runtime = load_student_policy_runtime(
        student_artifact_path=args.student_artifact,
        checkpoint_path=args.checkpoint,
        task=args.task,
        train=args.train,
        hand=args.hand,
        object_name=args.object,
        asset_dir=args.asset_dir,
        rl_device=args.rl_device,
        seed=args.seed,
        expl_block_idx=args.expl_block_idx,
    )

    print(
        "Student policy server ready on "
        f"{args.host}:{args.port} | task={args.task} train={args.train} hand={args.hand} object={args.object}"
    )
    runtime.interactive_goal_input_enabled = True
    if getattr(runtime, "goal_sequence", None):
        goal_index, goal_offset = runtime.cycle_goal_override()
        print(f"Initialized server goal_override -> idx={goal_index} goal_offset={goal_offset:.6f}")
    else:
        print("Server goal input enabled, but no configured goal sequence was found.")
    _start_goal_input_thread(runtime)
    with StudentPolicyTCPServer((args.host, args.port), runtime) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
