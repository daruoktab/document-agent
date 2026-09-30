"""Pastikan seluruh tes pytest berjalan tanpa koneksi ke server atau API nyata."""

from __future__ import annotations

import inspect
import socket
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def isolate_local_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Developer .env must not select real endpoints or alter test expectations."""
    import app.config

    monkeypatch.setattr(app.config, "_DOTENV_VALUES", {})
    monkeypatch.setattr(app.config, "_LOCAL_ENV_PATH", tmp_path / "absent.env")


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def is_local_socketpair() -> bool:
        return any(
            frame.function in {"socketpair", "_fallback_socketpair"}
            and frame.filename.endswith("socket.py")
            for frame in inspect.stack(context=0)
        )

    def reject_connection(sock: socket.socket, address: Any) -> None:
        if is_local_socketpair():
            return original_connect(sock, address)
        raise AssertionError("Tes mencoba membuka koneksi jaringan nyata")

    def reject_connection_ex(sock: socket.socket, address: Any) -> int:
        if is_local_socketpair():
            return original_connect_ex(sock, address)
        raise AssertionError("Tes mencoba membuka koneksi jaringan nyata")

    monkeypatch.setattr(socket.socket, "connect", reject_connection)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_connection_ex)
