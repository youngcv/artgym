from __future__ import annotations

import json
import socket
from typing import Any


class PolicyRPCError(RuntimeError):
    pass


def _json_default(value: Any):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def send_json_line(writer, payload: dict[str, Any]) -> None:
    writer.write(json.dumps(payload, default=_json_default))
    writer.write("\n")
    writer.flush()


def recv_json_line(reader) -> dict[str, Any]:
    line = reader.readline()
    if line == "":
        raise EOFError("Remote peer closed the connection.")
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise PolicyRPCError(f"Invalid JSON payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise PolicyRPCError(f"Expected a JSON object, got {type(payload).__name__}")
    return payload


class JsonLineSocketClient:
    def __init__(self, host: str, port: int, timeout_sec: float = 5.0):
        self.host = host
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)
        self.sock = None
        self.reader = None
        self.writer = None

    def connect(self) -> None:
        if self.sock is not None:
            return
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout_sec)
        self.sock.settimeout(self.timeout_sec)
        self.reader = self.sock.makefile("r", encoding="utf-8")
        self.writer = self.sock.makefile("w", encoding="utf-8")

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.connect()
        send_json_line(self.writer, payload)
        response = recv_json_line(self.reader)
        if response.get("status") == "error":
            raise PolicyRPCError(response.get("error", "Unknown RPC error"))
        return response

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
