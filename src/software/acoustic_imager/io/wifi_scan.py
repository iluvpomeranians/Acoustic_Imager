"""
WiFi network scanning for Raspberry Pi / Linux.
Uses nmcli (NetworkManager) first, falls back to iwlist when nmcli fails or returns empty.
"""

from __future__ import annotations

import re
import subprocess
from typing import List, Dict


def _scan_nmcli() -> List[Dict[str, str]]:
    """Scan using nmcli (NetworkManager)."""
    result: List[Dict[str, str]] = []
    try:
        out = subprocess.run(
            [
                "nmcli",
                "-t",
                "-f", "SSID,SIGNAL,SECURITY",
                "device", "wifi", "list",
                "--rescan", "yes",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode != 0 or not out.stdout:
            return result

        seen = set()
        for line in out.stdout.strip().splitlines():
            parts = line.split(":")
            if len(parts) < 3:
                continue
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


def _parse_iw(text: str) -> List[Dict[str, str]]:
    """Parse 'iw dev wlan0 scan' output into list of {ssid, signal, security}."""
    result: List[Dict[str, str]] = []
    seen: set[str] = set()
    # Split by BSS blocks; format: "BSS aa:bb:cc:dd:ee:ff" or "BSS aa:bb:cc:dd:ee:ff(on wlan0)"
    blocks = re.split(
        r"BSS [0-9a-f]{2}(?::[0-9a-f]{2}){5}(?:\([^)]*\))?\s*\n?",
        text,
        flags=re.IGNORECASE,
    )
    for block in blocks[1:]:
        ssid = ""
        signal = ""
        security = "Open"
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("SSID:"):
                ssid = line[5:].strip()
            elif line.startswith("signal:"):
                m = re.search(r"(-?\d+(?:\.\d+)?)\s*dBm", line)
                if m:
                    dbm = float(m.group(1))
                    pct = min(100, max(0, int(100 + (dbm + 30) * (100 / 60))))
                    signal = str(pct)
            elif "RSN:" in line or "WPA" in line or "WPA2" in line:
                security = "WPA2"
            elif "WEP" in line:
                security = "WEP"
        if ssid and ssid not in seen:
            seen.add(ssid)
            result.append({"ssid": ssid, "signal": signal, "security": security})
    return result


def _scan_iw() -> List[Dict[str, str]]:
    """Scan using 'iw dev wlan0 scan' - often returns more networks than nmcli on Pi."""
    for iface in ["wlan0", "wlan1"]:
        for cmd in [["iw", "dev", iface, "scan"], ["sudo", "iw", "dev", iface, "scan"]]:
            try:
                out = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if out.returncode == 0 and out.stdout:
                    result = _parse_iw(out.stdout)
                    if result:
                        return result
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                continue
    return []


def _scan_iwlist() -> List[Dict[str, str]]:
    """Scan using iwlist (fallback when nmcli unavailable or empty)."""
    result: List[Dict[str, str]] = []
    interfaces = ["wlan0", "wlan1"]
    for iface in interfaces:
        for cmd in [["iwlist", iface, "scan"], ["sudo", "iwlist", iface, "scan"]]:
            try:
                out = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if out.returncode != 0:
                    continue
                result = _parse_iwlist(out.stdout or "")
                if result:
                    return result
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                continue
    return []


def _parse_iwlist(text: str) -> List[Dict[str, str]]:
    """Parse iwlist scan output into list of {ssid, signal, security}."""
    result: List[Dict[str, str]] = []
    cells = re.split(r"Cell \d+ - Address:", text, flags=re.IGNORECASE)
    for cell in cells[1:]:
        ssid = ""
        signal = ""
        security = "Open"
        essid_match = re.search(r'ESSID:"([^"]*)"', cell)
        if essid_match:
            ssid = essid_match.group(1).strip()
        quality_match = re.search(r"Quality=(\d+)/(\d+)", cell)
        if quality_match:
            num, den = int(quality_match.group(1)), int(quality_match.group(2))
            signal = str(int(100 * num / den)) if den else ""
        if "Encryption key:on" in cell or "IE: WPA" in cell or "IE: IEEE 802.11i/WPA2" in cell:
            security = "WPA2"
        elif "Encryption key:on" in cell:
            security = "WEP"
        if ssid:
            result.append({"ssid": ssid, "signal": signal, "security": security})
    return result


def scan_wifi_networks() -> List[Dict[str, str]]:
    """
    Scan for nearby WiFi networks.
    Tries nmcli first, then iw (often returns more networks on Pi), then iwlist.
    On Raspberry Pi, nmcli often returns only the connected network; iw/sudo iw
    typically returns all visible networks.
    Returns list of {"ssid", "signal", "security"} dicts.
    """
    result = _scan_nmcli()
    if not result:
        result = _scan_nmcli_iface()
    # nmcli often returns only the connected network on Pi; try iw for full scan
    if len(result) <= 1:
        iw_result = _scan_iw()
        if len(iw_result) > len(result):
            result = iw_result
    if not result:
        result = _scan_iwlist()
    return result


def _scan_nmcli_iface() -> List[Dict[str, str]]:
    """Try nmcli with explicit wlan0 interface (some Pi setups need this)."""
    try:
        out = subprocess.run(
            [
                "nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY",
                "device", "wifi", "list", "ifname", "wlan0", "--rescan", "yes",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode != 0 or not out.stdout:
            return []
        result = []
        seen = set()
        for line in (out.stdout or "").strip().splitlines():
            parts = line.split(":")
            if len(parts) < 3:
                continue
            security = (parts[-1] or "").strip()
            signal = (parts[-2] or "").strip()
            ssid = ":".join(parts[:-2]).strip()
            if not ssid or ssid in seen:
                continue
            seen.add(ssid)
            result.append({"ssid": ssid, "signal": signal, "security": security if security and security != "--" else "Open"})
        return result
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return []


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


def disconnect_wifi() -> tuple[bool, str]:
    """
    Disconnect from current WiFi. Returns (success, message).
    Tries generic disconnect first, then wlan0 if needed.
    """
    for args in [["nmcli", "device", "disconnect"], ["nmcli", "device", "disconnect", "ifname", "wlan0"]]:
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                return True, "Disconnected"
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass
    return False, "Disconnect failed"
