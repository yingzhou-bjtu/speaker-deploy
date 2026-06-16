"""TCP PCM 客户端，对接 robotmedia 默认 5001 口。"""
from __future__ import annotations

import socket
import time
from typing import Optional


def current_time_ms() -> int:
    try:
        from cy_audio_node.time_utils import current_time_ms as _t

        return int(_t())
    except ImportError:
        return int(time.time() * 1000)


class TCPAudioClient:
    """按固定字节数从 TCP 取 PCM 帧。"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5001,
        *,
        timeout_sec: float = 10.0,
    ):
        self.host = host
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)
        self.sock: Optional[socket.socket] = None
        self.buffer = b""

    def connect(self, *, quiet: bool = False) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn_to = self.timeout_sec if self.timeout_sec > 0 else 15.0
        self.sock.settimeout(conn_to)
        self.sock.connect((self.host, self.port))
        # 连上后阻塞等数据，静音时不该周期性超时断连。
        self.sock.settimeout(None)
        if not quiet:
            print(f"Connected to TCP audio server {self.host}:{self.port}", flush=True)

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def recv_chunk(self, max_size: int = 4096, *, idle_timeout: float = 0.5) -> bytes:
        """收数据，超时内没数据就返回空字节。"""
        if self.sock is None:
            raise RuntimeError("not connected")
        self.sock.settimeout(float(idle_timeout))
        try:
            data = self.sock.recv(int(max_size))
        except socket.timeout:
            return b""
        if not data:
            raise ConnectionError("Server disconnected")
        return data

    def recv_pcm(self, frame_size: int) -> bytes:
        if self.sock is None:
            raise RuntimeError("not connected")
        frame_size = int(frame_size)
        while len(self.buffer) < frame_size:
            chunk = self.recv_chunk(4096, idle_timeout=60.0)
            if not chunk:
                raise socket.timeout("no PCM within idle window")
            self.buffer += chunk
        frame = self.buffer[:frame_size]
        self.buffer = self.buffer[frame_size:]
        return frame

    def recv_pcm_with_time(self, frame_size: int) -> tuple[bytes, int]:
        return self.recv_pcm(frame_size), current_time_ms()
