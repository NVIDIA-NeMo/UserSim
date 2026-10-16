#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Verify the versioned upstream installer before installing the registry's Mise release.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
version="$(sed -n 's/^min_version = "\([^"]*\)"/\1/p' "$root/.mise.toml")"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo 'Invalid Mise min_version' >&2; exit 1; }
install_path="${MISE_INSTALL_PATH:-$root/.cache/tools/mise}"
if [[ -x "$install_path" ]] && [[ "$("$install_path" --version | cut -d' ' -f1)" == "$version" ]]; then
    exit 0
fi
command -v gpg >/dev/null || { echo 'GPG is required to verify Mise; unsigned installation is forbidden.' >&2; exit 1; }
command -v python3 >/dev/null || { echo 'Python 3 is required to bound Mise installation.' >&2; exit 1; }
command -v curl >/dev/null || { echo 'curl is required to download the signed Mise installer.' >&2; exit 1; }
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
chmod 700 "$work"
fetch() {
    curl -fsSL --connect-timeout 10 --max-time 60 --retry 3 --retry-delay 2 --retry-all-errors "$@"
}
key=24853EC9F655CE80B48E6C3A8B81C9D17413A06D
fetch -o "$work/key.asc" "https://keys.openpgp.org/vks/v1/by-fingerprint/$key"
gpg --homedir "$work" --batch --no-tty --import "$work/key.asc"
gpg --homedir "$work" --batch --no-tty --with-colons --list-keys "$key" | grep -q "^fpr:::::::::$key:"
fetch -o "$work/install.sig" "https://github.com/jdx/mise/releases/download/v$version/install.sh.sig"
gpg --homedir "$work" --batch --no-tty --status-fd 3 --output "$work/install.sh" --decrypt "$work/install.sig" 3>"$work/status"
awk -v key="$key" '$2 == "VALIDSIG" && ($3 == key || $12 == key) { valid = 1 } END { exit !valid }' "$work/status" \
    || { echo 'Mise installer signer does not match the pinned release key.' >&2; exit 1; }
mkdir -p "$(dirname "$install_path")"
MISE_VERSION="v$version" MISE_INSTALL_PATH="$install_path" MISE_INSTALL_HELP=0 python3 - "$work/install.sh" <<'PYTHON'
import os
import signal
import subprocess
import sys

process = subprocess.Popen(["sh", sys.argv[1]], start_new_session=True)
try:
    code = process.wait(timeout=300)
except subprocess.TimeoutExpired:
    os.killpg(process.pid, signal.SIGKILL)
    process.wait()
    raise SystemExit("Mise binary installation exceeded 300 seconds")
raise SystemExit(code)
PYTHON
[[ "$("$install_path" --version | cut -d' ' -f1)" == "$version" ]] || { echo 'Installed Mise version mismatch' >&2; exit 1; }
echo "Verified Mise $version installed at $install_path"
