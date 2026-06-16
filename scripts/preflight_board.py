#!/usr/bin/env python3
"""板端启动前检查 venv、ONNX 和 robotmedia 配置。"""
from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

_DEPLOY = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = Path("/usr/share/robotmedia/config/config.ini")
_DEFAULT_ONNX = _DEPLOY / "pretrained" / "eres2net_sv.onnx"


def _vad_pcm_socket_enabled(ini_path: Path) -> bool:
    if not ini_path.is_file():
        return False
    in_sec = False
    for raw in ini_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line == "[vad_pcm_socket]":
            in_sec = True
            continue
        if in_sec and line.startswith("[") and line.endswith("]"):
            break
        if in_sec and line.startswith("enable"):
            val = line.split("=", 1)[-1].strip()
            return val in ("1", "true", "yes", "on")
    return False


def _port_listening(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, int(port)))
        s.close()
        return True
    except OSError:
        return False


def run_checks(
    *,
    config_ini: Path,
    pcm_host: str,
    pcm_port: int,
    onnx_path: Path,
    venv_python: Path | None,
    require_pcm: bool,
) -> list[str]:
    errors: list[str] = []
    warns: list[str] = []

    py = venv_python or (_DEPLOY / ".venv" / "bin" / "python")
    if not py.is_file():
        errors.append(f"缺少 venv: {py}（请 ./setup_board_deps.sh）")
    else:
        import subprocess

        r = subprocess.run(
            [str(py), "-c", "import onnxruntime, hdbscan, numpy"],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            errors.append(f"venv 依赖不可用: {r.stderr.strip() or r.stdout}")

    if not onnx_path.is_file():
        errors.append(f"缺少 ONNX: {onnx_path}")

    if not config_ini.is_file():
        warns.append(f"未找到 robotmedia 配置: {config_ini}")
    elif not _vad_pcm_socket_enabled(config_ini):
        errors.append(
            f"{config_ini} 中 [vad_pcm_socket] enable 未开启；"
            "唤醒后 TCP 不会推 PCM。请设 enable=1 并 systemctl restart robotmedia"
        )

    if require_pcm:
        if not _port_listening(pcm_host, pcm_port):
            errors.append(
                f"TCP {pcm_host}:{pcm_port} 不可连接（robotmedia 未监听或未启动）"
            )

    for w in warns:
        print(f"WARN: {w}", file=sys.stderr)
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="板端声纹服务启动前检查")
    ap.add_argument("--config-ini", type=Path, default=_DEFAULT_CONFIG)
    ap.add_argument("--pcm-host", default="127.0.0.1")
    ap.add_argument("--pcm-port", type=int, default=5001)
    ap.add_argument("--onnx-path", type=Path, default=_DEFAULT_ONNX)
    ap.add_argument("--venv-python", type=Path, default=None)
    ap.add_argument("--no-pcm", action="store_true", help="不检查 PCM 端口")
    args = ap.parse_args()

    errs = run_checks(
        config_ini=args.config_ini,
        pcm_host=str(args.pcm_host),
        pcm_port=int(args.pcm_port),
        onnx_path=args.onnx_path,
        venv_python=args.venv_python,
        require_pcm=not args.no_pcm,
    )
    if errs:
        print("preflight FAILED:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("preflight OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
