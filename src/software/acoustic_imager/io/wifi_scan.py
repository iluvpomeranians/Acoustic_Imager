"""
WiFi network scanning for Raspberry Pi / Linux (nmcli).
"""

from __future__ import annotations

import subprocess
from typing import List, Dict


def scan_wifi_networks() -> List[Dict[str, str]]:
    """
    Scan for nearby WiFi networks using nmcli.
    Returns list of {"ssid", "signal", "security"} dicts.
    """
    result: List[Dict[str, str]] = []
    try:
        # Rescan (async; list may use cached results)
        subprocess.run(
            ["nmcli", "device", "wifi", "rescan"],
            capture_output=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    try:
        out = subprocess.run(
            [
                "nmcli",
                "-t",
                "-f", "SSID,SIGNAL,SECURITY",
                "device", "wifi", "list",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0 or not out.stdout:
            return result

        seen = set()
        for line in out.stdout.strip().splitlines():
            parts = line.split(":")
            if len(parts) < 3:
                continue
            # SSID may contain colons; last two parts are SIGNAL and SECURITY
            security = (parts[-1] or "").strip()
            signal = (parts[-2] or "").strip()
            ssid = ":".join(parts[:-2]).strip()
            if not ssid or ssid in seen:
                continue
            seen.add(ssid)
            result.append({
                "ssid": ssid,
                "signal": signal,
                "security": security if security and security != "--" else "Open",
            })
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return result


def connect_wifi(ssid: str, password: str) -> tuple[bool, str]:
    """
    Connect to WiFi network. Returns (success, message).
    """
    try:
        if password:
            out = subprocess.run(
                ["nmcli", "device", "wifi", "connect", ssid, "password", password],
                capture_output=True,
                text=True,
                timeout=15,
            )
        else:
            out = subprocess.run(
                ["nmcli", "device", "wifi", "connect", ssid],
                capture_output=True,
                text=True,
                timeout=15,
            )
        if out.returncode == 0:
            return True, "Connected"
        err = (out.stderr or out.stdout or "").strip()
        return False, err[:80] if err else "Connection failed"
    except subprocess.TimeoutExpired:
        return False, "Connection timed out"
    except FileNotFoundError:
        return False, "nmcli not found"
    except Exception as e:
        return False, str(e)[:80]
