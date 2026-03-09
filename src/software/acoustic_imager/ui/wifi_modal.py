"""
WiFi modal: discover networks, connect with password (keyboard for secured networks).
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np

from . import ui_cache
from .button import menu_buttons, Button
from .standard_keyboard import (
    draw_standard_alpha_keyboard,
    draw_standard_symbol_keyboard,
    compute_standard_keyboard_dimensions,
)
from ..state import HUD
from ..config import MENU_ACTIVE_BLUE, MENU_ACTIVE_BLUE_LIGHT
from ..io.wifi_scan import scan_wifi_networks, connect_wifi, disconnect_wifi
from ..system_info import get_system_network_info
from .email_modal import _draw_eye_icon

MODAL_W = 520
MODAL_H_LIST = 380
MODAL_H_PASSWORD = 420


def _run_connect(ssid: str, password: str) -> None:
    """Background thread: connect and set status."""
    HUD.wifi_connect_status = "connecting"
    ok, msg = connect_wifi(ssid, password)
    HUD.wifi_connect_status = "ok" if ok else "error"
    HUD.wifi_connect_message = msg


def _run_scan() -> None:
    """Background thread: scan and update networks."""
    HUD.wifi_scanning = True
    try:
        HUD.wifi_networks = scan_wifi_networks()
    finally:
        HUD.wifi_scanning = False


def draw_wifi_modal(frame: np.ndarray) -> None:
    """Draw WiFi modal: list screen or password screen with keyboard."""
    if not HUD.wifi_modal_open:
        return

    fh, fw = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    border_color = (100, 100, 100)

    # Dark overlay (cached black buffer, no frame copy)
    ui_cache.apply_modal_dim(frame, 0.5)

    if HUD.wifi_modal_screen == "list":
        _draw_list_screen(frame, fw, fh, font, text_color, border_color)
    else:
        _draw_password_screen(frame, fw, fh, font, text_color, border_color)


def _draw_list_screen(
    frame: np.ndarray, fw: int, fh: int, font, text_color, border_color
) -> None:
    """Draw network list: title, Scan, current connection (blue), Other Networks."""
    modal_w = MODAL_W
    modal_h = MODAL_H_LIST
    modal_x = (fw - modal_w) // 2
    modal_y = (fh - modal_h) // 2

    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), border_color, 3, cv2.LINE_AA)

    pad = 16
    title_y = modal_y + 32
    cv2.putText(frame, "WiFi", (modal_x + pad, title_y), font, 0.7, text_color, 1, cv2.LINE_AA)

    # Scan button
    scan_w, scan_h = 80, 32
    scan_x = modal_x + modal_w - scan_w - pad
    scan_y = modal_y + 12
    if "wifi_scan" not in menu_buttons:
        menu_buttons["wifi_scan"] = Button(scan_x, scan_y, scan_w, scan_h, "Scan")
    else:
        menu_buttons["wifi_scan"].x, menu_buttons["wifi_scan"].y = scan_x, scan_y
        menu_buttons["wifi_scan"].w, menu_buttons["wifi_scan"].h = scan_w, scan_h
    menu_buttons["wifi_scan"].draw(frame, transparent=True, active_color=MENU_ACTIVE_BLUE)

    # Current connection (top, highlighted in blue)
    connected_ssid, _, _ = get_system_network_info(0)
    row_h = 36
    section_gap = 8
    current_y = modal_y + 52
    row_x = modal_x + pad
    row_w = modal_w - 2 * pad

    # "Current connection" section header
    cv2.putText(frame, "Current connection", (row_x, current_y - 4), font, 0.44, (180, 180, 180), 1, cv2.LINE_AA)
    current_row_y = current_y + 6
    expanded = HUD.wifi_current_expanded
    box_h = (row_h - 4) + (72 if expanded else 0)  # extra height when expanded
    if "wifi_current" not in menu_buttons:
        menu_buttons["wifi_current"] = Button(row_x, current_row_y, row_w, box_h, "")
    else:
        menu_buttons["wifi_current"].x, menu_buttons["wifi_current"].y = row_x, current_row_y
        menu_buttons["wifi_current"].w, menu_buttons["wifi_current"].h = row_w, box_h
    # Blue highlight for current connection box (expanded when clicked)
    cv2.rectangle(frame, (row_x, current_row_y), (row_x + row_w, current_row_y + box_h), MENU_ACTIVE_BLUE, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (row_x, current_row_y), (row_x + row_w, current_row_y + box_h), MENU_ACTIVE_BLUE_LIGHT, 1, cv2.LINE_AA)
    current_label = connected_ssid[:24] if connected_ssid else "Not connected"
    cv2.putText(frame, current_label, (row_x + 8, current_row_y + (row_h - 4) // 2 + 6), font, 0.48, text_color, 1, cv2.LINE_AA)
    if expanded:
        info_y = current_row_y + row_h + 4
        ssid_full, ip, hostname = get_system_network_info(0)
        line_h = 18
        info_color = (220, 220, 220)
        if ssid_full:
            cv2.putText(frame, f"SSID: {ssid_full[:28]}", (row_x + 8, info_y), font, 0.40, info_color, 1, cv2.LINE_AA)
        if ip:
            cv2.putText(frame, f"IP: {ip}", (row_x + 8, info_y + line_h), font, 0.40, info_color, 1, cv2.LINE_AA)
        if hostname:
            cv2.putText(frame, f"Device: {hostname[:24]}", (row_x + 8, info_y + 2 * line_h), font, 0.40, info_color, 1, cv2.LINE_AA)
        # Signal/security from scan if available
        networks = HUD.wifi_networks or []
        match = next((n for n in networks if (n.get("ssid") or "").strip() == ssid_full), None)
        if match:
            sig = match.get("signal", "")
            sec = match.get("security", "Open")
            cv2.putText(frame, f"Signal: {sig}%  Security: {sec}", (row_x + 8, info_y + 3 * line_h), font, 0.40, info_color, 1, cv2.LINE_AA)
        # Disconnect button: only when expanded, at bottom right of box
        if connected_ssid:
            disconnect_btn_w, disconnect_btn_h = 90, 28
            disc_x = row_x + row_w - disconnect_btn_w - 8
            disc_y = current_row_y + box_h - disconnect_btn_h - 6
            if "wifi_disconnect" not in menu_buttons:
                menu_buttons["wifi_disconnect"] = Button(disc_x, disc_y, disconnect_btn_w, disconnect_btn_h, "Disconnect")
            else:
                menu_buttons["wifi_disconnect"].x, menu_buttons["wifi_disconnect"].y = disc_x, disc_y
                menu_buttons["wifi_disconnect"].w, menu_buttons["wifi_disconnect"].h = disconnect_btn_w, disconnect_btn_h
            cv2.rectangle(frame, (disc_x, disc_y), (disc_x + disconnect_btn_w, disc_y + disconnect_btn_h), (60, 60, 60), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (disc_x, disc_y), (disc_x + disconnect_btn_w, disc_y + disconnect_btn_h), (255, 255, 255), 1, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize("Disconnect", font, 0.4, 1)
            cv2.putText(frame, "Disconnect", (disc_x + (disconnect_btn_w - tw) // 2, disc_y + (disconnect_btn_h + th) // 2), font, 0.4, text_color, 1, cv2.LINE_AA)
    if not (expanded and connected_ssid) and "wifi_disconnect" in menu_buttons:
        menu_buttons["wifi_disconnect"].w = 0
        menu_buttons["wifi_disconnect"].h = 0

    # "Other Networks" section (below expanded or collapsed current box)
    other_y0 = current_row_y + box_h + section_gap + 16
    other_label = "Other Networks"
    if HUD.wifi_scanning:
        other_label += "  Scanning..."
    cv2.putText(frame, other_label, (row_x, other_y0 - 4), font, 0.44, (180, 180, 180), 1, cv2.LINE_AA)
    list_y0 = other_y0 + 6
    max_rows = max(0, (modal_h - list_y0 + modal_y - pad) // row_h)
    networks = HUD.wifi_networks or []
    other_networks = [n for n in networks if (n.get("ssid") or "").strip() != connected_ssid]
    for i, net in enumerate(other_networks[:max_rows]):
        ssid = net.get("ssid", "")
        signal = net.get("signal", "")
        security = net.get("security", "Open")
        row_y = list_y0 + i * row_h
        key = f"wifi_net_{i}"
        if key not in menu_buttons:
            menu_buttons[key] = Button(row_x, row_y, row_w, row_h - 4, "")
        else:
            menu_buttons[key].x, menu_buttons[key].y = row_x, row_y
            menu_buttons[key].w, menu_buttons[key].h = row_w, row_h - 4
        cv2.rectangle(frame, (row_x, row_y), (row_x + row_w, row_y + row_h - 4), (50, 50, 50), -1)
        cv2.rectangle(frame, (row_x, row_y), (row_x + row_w, row_y + row_h - 4), (80, 80, 80), 1, cv2.LINE_AA)
        label = f"{ssid[:24]}  {signal}%  {security}" if signal else f"{ssid[:28]}  {security}"
        cv2.putText(frame, label, (row_x + 8, row_y + (row_h - 4) // 2 + 6), font, 0.48, text_color, 1, cv2.LINE_AA)

    if "wifi_modal_panel" not in menu_buttons:
        menu_buttons["wifi_modal_panel"] = Button(modal_x, modal_y, modal_w, modal_h, "")
    else:
        b = menu_buttons["wifi_modal_panel"]
        b.x, b.y, b.w, b.h = modal_x, modal_y, modal_w, modal_h


def _draw_password_screen(
    frame: np.ndarray, fw: int, fh: int, font, text_color, border_color
) -> None:
    """Draw password form + standard keyboard."""
    modal_w = MODAL_W
    modal_h = MODAL_H_PASSWORD
    modal_x = (fw - modal_w) // 2
    modal_y = (fh - modal_h) // 2

    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), border_color, 3, cv2.LINE_AA)

    pad = 16
    ssid = HUD.wifi_connect_ssid or ""
    title_y = modal_y + 32
    cv2.putText(frame, f"Connect to: {ssid[:28]}", (modal_x + pad, title_y), font, 0.56, text_color, 1, cv2.LINE_AA)

    # Password field with show/hide toggle
    field_y = modal_y + 56
    field_h = 40
    eye_btn_w = 36
    eye_btn_gap = 6
    field_w = modal_w - 2 * pad - eye_btn_w - eye_btn_gap
    field_x = modal_x + pad
    cv2.putText(frame, "Password:", (field_x, field_y - 8), font, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (field_x, field_y), (field_x + field_w, field_y + field_h), (28, 28, 34), -1)
    cv2.rectangle(frame, (field_x, field_y), (field_x + field_w, field_y + field_h), (120, 120, 120), 1, cv2.LINE_AA)
    pw_visible = HUD.wifi_password_visible
    pw_display = HUD.wifi_password if pw_visible else ("*" * len(HUD.wifi_password))
    cv2.putText(frame, pw_display or " ", (field_x + 8, field_y + field_h // 2 + 6), font, 0.52, text_color, 1, cv2.LINE_AA)
    # Show password (eye) button
    eye_x = field_x + field_w + eye_btn_gap
    eye_y = field_y + (field_h - eye_btn_w) // 2
    eye_btn_h = min(eye_btn_w, field_h)
    cv2.rectangle(frame, (eye_x, eye_y), (eye_x + eye_btn_w, eye_y + eye_btn_h), (50, 50, 55), -1)
    cv2.rectangle(frame, (eye_x, eye_y), (eye_x + eye_btn_w, eye_y + eye_btn_h), (160, 160, 160), 1, cv2.LINE_AA)
    eye_cx, eye_cy = eye_x + eye_btn_w // 2, eye_y + eye_btn_h // 2
    _draw_eye_icon(frame, eye_cx, eye_cy, min(eye_btn_w, eye_btn_h) // 3, pw_visible, (220, 220, 220))
    if "wifi_password_field" not in menu_buttons:
        menu_buttons["wifi_password_field"] = Button(field_x, field_y, field_w, field_h, "")
    else:
        menu_buttons["wifi_password_field"].x, menu_buttons["wifi_password_field"].y = field_x, field_y
        menu_buttons["wifi_password_field"].w, menu_buttons["wifi_password_field"].h = field_w, field_h
    if "wifi_password_toggle" not in menu_buttons:
        menu_buttons["wifi_password_toggle"] = Button(eye_x, eye_y, eye_btn_w, eye_btn_h, "")
    else:
        menu_buttons["wifi_password_toggle"].x, menu_buttons["wifi_password_toggle"].y = eye_x, eye_y
        menu_buttons["wifi_password_toggle"].w, menu_buttons["wifi_password_toggle"].h = eye_btn_w, eye_btn_h

    # Connect button
    conn_w, conn_h = 100, 36
    conn_x = modal_x + modal_w - conn_w - pad
    conn_y = field_y + field_h + 12
    if "wifi_connect_btn" not in menu_buttons:
        menu_buttons["wifi_connect_btn"] = Button(conn_x, conn_y, conn_w, conn_h, "Connect")
    else:
        menu_buttons["wifi_connect_btn"].x, menu_buttons["wifi_connect_btn"].y = conn_x, conn_y
        menu_buttons["wifi_connect_btn"].w, menu_buttons["wifi_connect_btn"].h = conn_w, conn_h
    menu_buttons["wifi_connect_btn"].draw(frame, transparent=True, active_color=MENU_ACTIVE_BLUE)

    # Status message
    status = HUD.wifi_connect_status
    msg = HUD.wifi_connect_message
    if status == "connecting":
        cv2.putText(frame, "Connecting...", (modal_x + pad, conn_y + conn_h + 24), font, 0.44, (200, 200, 0), 1, cv2.LINE_AA)
    elif status == "ok":
        cv2.putText(frame, "Connected!", (modal_x + pad, conn_y + conn_h + 24), font, 0.44, (0, 200, 100), 1, cv2.LINE_AA)
    elif status == "error" and msg:
        cv2.putText(frame, msg[:40], (modal_x + pad, conn_y + conn_h + 24), font, 0.40, (0, 0, 220), 1, cv2.LINE_AA)

    # Keyboard: standard layout (matches app style)
    keyboard_y0 = modal_y + modal_h - 200
    keyboard_h = 200
    content_w = modal_w - 2 * pad
    key_gap = 6
    n_rows = 5
    dims = compute_standard_keyboard_dimensions(content_w, keyboard_h, key_gap=key_gap, n_rows=n_rows)
    key_w, key_h, sp_w = dims["key_w"], dims["key_h"], dims["sp_w"]
    key_x_base = modal_x + pad
    key_y = keyboard_y0 + key_gap
    key_font = 0.56
    key_font_special = 0.50
    is_symbol = HUD.wifi_keyboard_mode == "symbol"
    shift_highlight = HUD.wifi_shift_next

    if is_symbol:
        draw_standard_symbol_keyboard(
            frame, key_x_base, key_y, content_w, key_w, key_h, sp_w, key_gap,
            "wifi_key_", font, key_font, key_font_special,
        )
    else:
        draw_standard_alpha_keyboard(
            frame, key_x_base, key_y, content_w, key_w, key_h, sp_w, key_gap,
            "wifi_key_", font, key_font, key_font_special,
            shift_highlight,
            shift_highlight_color=MENU_ACTIVE_BLUE,
        )

    if "wifi_modal_panel" not in menu_buttons:
        menu_buttons["wifi_modal_panel"] = Button(modal_x, modal_y, modal_w, modal_h, "")
    else:
        b = menu_buttons["wifi_modal_panel"]
        b.x, b.y, b.w, b.h = modal_x, modal_y, modal_w, modal_h

    if "wifi_keyboard_region" not in menu_buttons:
        menu_buttons["wifi_keyboard_region"] = Button(modal_x, keyboard_y0, modal_w, keyboard_h, "")
    else:
        menu_buttons["wifi_keyboard_region"].x = modal_x
        menu_buttons["wifi_keyboard_region"].y = keyboard_y0
        menu_buttons["wifi_keyboard_region"].w = modal_w
        menu_buttons["wifi_keyboard_region"].h = keyboard_h


def handle_wifi_modal_click(x: int, y: int) -> bool:
    """Handle click. Returns True if handled."""
    if not HUD.wifi_modal_open:
        return False

    if HUD.wifi_modal_screen == "list":
        return _handle_list_click(x, y)
    return _handle_password_click(x, y)


def _handle_list_click(x: int, y: int) -> bool:
    if "wifi_scan" in menu_buttons and menu_buttons["wifi_scan"].contains(x, y):
        if not HUD.wifi_scanning:
            thread = threading.Thread(target=_run_scan, daemon=True)
            thread.start()
        return True
    if "wifi_disconnect" in menu_buttons and menu_buttons["wifi_disconnect"].w > 0 and menu_buttons["wifi_disconnect"].contains(x, y):
        ok, _ = disconnect_wifi()
        if ok and not HUD.wifi_scanning:
            thread = threading.Thread(target=_run_scan, daemon=True)
            thread.start()
        return True
    if "wifi_current" in menu_buttons and menu_buttons["wifi_current"].contains(x, y):
        HUD.wifi_current_expanded = not HUD.wifi_current_expanded
        return True
    connected_ssid, _, _ = get_system_network_info(0)
    other_networks = [n for n in (HUD.wifi_networks or []) if (n.get("ssid") or "").strip() != connected_ssid]
    for i, net in enumerate(other_networks):
        key = f"wifi_net_{i}"
        if key in menu_buttons and menu_buttons[key].contains(x, y):
            ssid = net.get("ssid", "")
            security = net.get("security", "Open")
            if security == "Open":
                HUD.wifi_connect_ssid = ssid
                HUD.wifi_password = ""
                HUD.wifi_connect_status = ""
                HUD.wifi_connect_message = ""
                thread = threading.Thread(target=_run_connect, args=(ssid, ""), daemon=True)
                thread.start()
                return True
            HUD.wifi_connect_ssid = ssid
            HUD.wifi_password = ""
            HUD.wifi_modal_screen = "password"
            HUD.wifi_keyboard_mode = "alpha"
            HUD.wifi_shift_next = False
            HUD.wifi_password_visible = False
            HUD.wifi_connect_status = ""
            HUD.wifi_connect_message = ""
            return True
    if "wifi_modal_panel" in menu_buttons and menu_buttons["wifi_modal_panel"].contains(x, y):
        return True
    HUD.wifi_modal_open = False
    HUD.wifi_current_expanded = False
    return True


def _handle_password_click(x: int, y: int) -> bool:
    # Show/hide password toggle
    if "wifi_password_toggle" in menu_buttons and menu_buttons["wifi_password_toggle"].contains(x, y):
        HUD.wifi_password_visible = not HUD.wifi_password_visible
        return True
    # Back, Clear, Done (check before keyboard region so they always work)
    for key_id, action in [
        ("wifi_key_backspace", "backspace"),
        ("wifi_key_clear", "clear"),
        ("wifi_key_done", "done"),
    ]:
        if key_id in menu_buttons and menu_buttons[key_id].contains(x, y):
            if action == "backspace":
                HUD.wifi_password = HUD.wifi_password[:-1]
            elif action == "clear":
                HUD.wifi_password = ""
            else:
                HUD.wifi_modal_screen = "list"
                HUD.wifi_connect_ssid = ""
                HUD.wifi_password = ""
                HUD.wifi_password_visible = False
            return True
    # Connect button
    if "wifi_connect_btn" in menu_buttons and menu_buttons["wifi_connect_btn"].contains(x, y):
        if HUD.wifi_connect_status != "connecting":
            HUD.wifi_connect_status = ""
            HUD.wifi_connect_message = ""
            thread = threading.Thread(
                target=_run_connect,
                args=(HUD.wifi_connect_ssid, HUD.wifi_password),
                daemon=True,
            )
            thread.start()
        return True

    # Keyboard
    in_keyboard = "wifi_keyboard_region" in menu_buttons and menu_buttons["wifi_keyboard_region"].contains(x, y)
    if in_keyboard:
        _handle_wifi_keyboard_click(x, y)
        return True

    if "wifi_modal_panel" in menu_buttons and menu_buttons["wifi_modal_panel"].contains(x, y):
        return True
    # Click outside: go back to list
    HUD.wifi_modal_screen = "list"
    HUD.wifi_connect_ssid = ""
    HUD.wifi_password = ""
    HUD.wifi_password_visible = False
    return True


def _handle_wifi_keyboard_click(x: int, y: int) -> None:
    """Process keyboard key presses for wifi password."""
    from .keyboard import ROWS_QWERTY
    from .standard_keyboard import STANDARD_ALPHA_SHORTCUT_ROW, STANDARD_SYMBOL_ROW1, STANDARD_SYMBOL_ROW2, STANDARD_SYMBOL_ROW3, STANDARD_SYMBOL_ROW4
    from .keyboard import SPECIAL_KEYS_COMPACT

    is_alpha = HUD.wifi_keyboard_mode == "alpha"
    shift = HUD.wifi_shift_next
    pw = HUD.wifi_password

    # Mode switch
    if "wifi_key_switch_sym" in menu_buttons and menu_buttons["wifi_key_switch_sym"].contains(x, y):
        HUD.wifi_keyboard_mode = "symbol"
        return
    if "wifi_key_switch_alpha" in menu_buttons and menu_buttons["wifi_key_switch_alpha"].contains(x, y):
        HUD.wifi_keyboard_mode = "alpha"
        return
    if "wifi_key_shift" in menu_buttons and menu_buttons["wifi_key_shift"].contains(x, y):
        HUD.wifi_shift_next = not HUD.wifi_shift_next
        return

    # Special keys (Back, Clear, Done) - check first
    for key_id in ("wifi_key_backspace", "wifi_key_clear", "wifi_key_done"):
        if key_id in menu_buttons:
            b = menu_buttons[key_id]
            if b.w > 0 and b.h > 0 and b.contains(x, y):
                if key_id == "wifi_key_backspace":
                    HUD.wifi_password = pw[:-1]
                elif key_id == "wifi_key_clear":
                    HUD.wifi_password = ""
                else:
                    HUD.wifi_modal_screen = "list"
                    HUD.wifi_connect_ssid = ""
                    HUD.wifi_password = ""
                return

    # Alpha shortcut row
    for label, val, _ in STANDARD_ALPHA_SHORTCUT_ROW:
        k = f"wifi_key_{val}"
        if k in menu_buttons and menu_buttons[k].contains(x, y):
            if val == "space":
                HUD.wifi_password = pw + " "
            elif val == "dot":
                HUD.wifi_password = pw + "."
            elif val == "at":
                HUD.wifi_password = pw + "@"
            elif val == "dash":
                HUD.wifi_password = pw + "-"
            elif val == "underscore":
                HUD.wifi_password = pw + "_"
            elif val == "snippet_com":
                HUD.wifi_password = pw + ".com"
            return

    # Letter keys
    if is_alpha:
        for row in ROWS_QWERTY:
            for c in row:
                k = f"wifi_key_{c}"
                if k in menu_buttons and menu_buttons[k].contains(x, y):
                    char = c.upper() if shift else c.lower()
                    HUD.wifi_password = pw + char
                    if shift:
                        HUD.wifi_shift_next = False
                    return
    else:
        for row in [STANDARD_SYMBOL_ROW1, STANDARD_SYMBOL_ROW2, STANDARD_SYMBOL_ROW3, STANDARD_SYMBOL_ROW4]:
            for c in row:
                from .standard_keyboard import _symbol_key_suffix
                suf = _symbol_key_suffix(c)
                k = f"wifi_key_{suf}"
                if k in menu_buttons and menu_buttons[k].contains(x, y):
                    HUD.wifi_password = pw + c
                    return
