# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only socket boundary: permit loopback, reject external connections."""

import ipaddress
import socket

_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex
_getaddrinfo = socket.getaddrinfo


def local(host):
    if isinstance(host, bytes):
        host = host.decode("ascii")
    if host == "localhost":
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise OSError(f"E2E blocks external network: {host}")


def connect(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        local(address[0])
    return _connect(sock, address)


def connect_ex(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        local(address[0])
    return _connect_ex(sock, address)


def getaddrinfo(
    host: bytes | str | None,
    port: bytes | str | int | None,
    family: int = 0,
    type: int = 0,
    proto: int = 0,
    flags: int = 0,
):
    if host is not None:
        local(host)
    return _getaddrinfo(host, port, family, type, proto, flags)


socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
# The wrapper preserves the stdlib call signature and forwards its return unchanged.
# ty cannot infer the stdlib alias return through this monkeypatch.
socket.getaddrinfo = getaddrinfo  # ty: ignore[invalid-assignment]
