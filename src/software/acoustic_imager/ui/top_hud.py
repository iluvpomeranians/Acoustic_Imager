"""
Top HUD: network, FPS, and time pills at the top of the main view.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional  # noqa: F401 Optional used in _draw_pill
import time
import cv2
import numpy as np

from ..config import (
    HUD_MENU_OPACITY,
    MENU_ACTIVE_BLUE,
    MENU_ACTIVE_BLUE_LIGHT,
    UI_PILL_H,
    UI_PILL_W,
    DB_BAR_WIDTH,
    FREQ_BAR_WIDTH,
)
from ..state import button_state
from .menu import _blue_gradient_overlay
from .battery_icon import draw_battery_icon

@dataclass
class HudRects:
    net:  Tuple[int,int,int,int]
    fps:  Tuple[int,int,int,int]
    time: Tuple[int,int,int,int]


def _draw_pill(frame: np.ndarray, x: int, y: int, w: int, h: int,
               bg=(0,0,0), alpha: Optional[float] = None, border=(255,255,255), is_active: bool = False) -> None:
    H, W = frame.shape[:2]
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x + w); y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return

    if alpha is None:
        alpha = HUD_MENU_OPACITY
    if is_active:
        alpha = HUD_MENU_OPACITY
        border = (255, 255, 255)

    roi = frame[y0:y1, x0:x1]
    pill_h, pill_w = roi.shape[:2]
    if is_active:
        overlay = _blue_gradient_overlay(pill_h, pill_w, MENU_ACTIVE_BLUE, MENU_ACTIVE_BLUE_LIGHT)
    else:
        overlay = np.empty_like(roi)
        overlay[:] = bg
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)

    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), border, 1, cv2.LINE_AA)

def _in_rect(x: int, y: int, r: Tuple[int,int,int,int]) -> bool:
    rx, ry, rw, rh = r
    return (rx <= x < rx+rw) and (ry <= y < ry+rh)

def draw_hud(
    frame: np.ndarray,
    *,
    details_level: str,
    open_panel: str,
    fps_ema: float,
    elapsed_s: float,
    frame_count: int,
    source_label: str,
    source_stats,
    fps_mode: str,
    frame_bytes: int,
    offset_y: float = 0.0,
    battery_percent: Optional[int] = None,
) -> HudRects:
    """
    Draw compact HUD top-left. offset_y moves the HUD vertically (0=visible, negative=retracted up).
    Returns hit rects for mouse click handling (with offset applied).
    """
    if details_level == "OFF":
        return HudRects(net=(0,0,0,0), fps=(0,0,0,0), time=(0,0,0,0))

    h, w, _ = frame.shape

    # Layout constants; y with visibility offset (retract up when offset_y < 0); same pill size as bottom HUD
    pad = 110
    y = 1 + int(offset_y)
    pill_h = UI_PILL_H
    icon_w = 34
    gap = 8

    # Compute throughput from fps_ema
    bytes_per_s = frame_bytes * fps_ema
    mbps_bits = (bytes_per_s * 8) / 1e6

    time_txt = time.strftime("%I:%M:%S%P")
    fps_txt  = f"{fps_ema:4.1f}"
    net_txt  = f"{mbps_bits:4.1f}"

    # "Icons" (no external assets): simple glyphs inside circles
    # If you later want PNG icons, we can swap these with image blits.
    def icon_circle(cx, cy, label: str):
        cv2.circle(frame, (cx, cy), 12, (255,255,255), -1, cv2.LINE_AA)  # filled white
        cv2.putText(frame, label, (cx-8, cy+6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)  # black letter

    def draw_clock_icon(cx, cy):
        cv2.circle(frame, (cx, cy), 12, (255,255,255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 12, (0,0,0), 1, cv2.LINE_AA)
        # hands
        cv2.line(frame, (cx, cy), (cx, cy-6), (0,0,0), 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx+5, cy+2), (0,0,0), 2, cv2.LINE_AA)

    def draw_fps_icon(cx, cy):
        cv2.circle(frame, (cx, cy), 12, (255,255,255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 12, (0,0,0), 1, cv2.LINE_AA)
        # gauge arc + needle
        cv2.ellipse(frame, (cx, cy+2), (7,7), 0, 200, 340, (0,0,0), 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy+2), (cx+6, cy-2), (0,0,0), 2, cv2.LINE_AA)

    def draw_net_icon(cx, cy):
        cv2.circle(frame, (cx, cy), 12, (255,255,255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 12, (0,0,0), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (cx-7, cy+4), (cx-5, cy+8), (0,0,0), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (cx-2, cy+2), (cx,   cy+8), (0,0,0), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (cx+3, cy-1), (cx+5, cy+8), (0,0,0), -1, cv2.LINE_AA)

    # Basic measurement (rough, but stable)
    def tw(s: str, scale=0.55, thick=1):
        (ww, hh), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        return ww

    net_w  = UI_PILL_W
    fps_w  = UI_PILL_W
    time_w = UI_PILL_W + 42  # Wider to fit clock + time + battery icon

    # Center HUD within the camera feed segment (between dB bar and freq bar)
    camera_feed_left = DB_BAR_WIDTH
    camera_feed_width = w - DB_BAR_WIDTH - FREQ_BAR_WIDTH
    total_width = net_w + gap + fps_w + gap + time_w
    start_x = camera_feed_left + (camera_feed_width - total_width) // 2

    x_net  = start_x
    x_fps  = x_net + net_w + gap
    x_time = x_fps + fps_w + gap

    # Draw pills (highlight green when active/open)
    _draw_pill(frame, x_net,  y, net_w,  pill_h, is_active=(open_panel == "net"))
    _draw_pill(frame, x_fps,  y, fps_w,  pill_h, is_active=(open_panel == "fps"))
    _draw_pill(frame, x_time, y, time_w, pill_h, is_active=(open_panel == "time"))

    cy = y + pill_h // 2
    text_y = y + pill_h // 2 + 6  # baseline for text
    icon_radius = 12
    icon_w = icon_radius * 2
    gap_icon_text = 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55

    def draw_pill_icon_text(pill_x: int, pill_w: int, label: str, draw_icon_fn) -> None:
        (label_w, _), _ = cv2.getTextSize(label, font, scale, 1)
        content_w = icon_w + gap_icon_text + label_w
        start_x = pill_x + (pill_w - content_w) // 2
        icon_cx = start_x + icon_radius
        draw_icon_fn(icon_cx, cy)
        cv2.putText(frame, label, (start_x + icon_w + gap_icon_text, text_y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

    # --- NETWORK ---
    draw_pill_icon_text(x_net, net_w, f"{net_txt} Mb/s", draw_net_icon)

    # --- FPS ---
    draw_pill_icon_text(x_fps, fps_w, f"{fps_txt} FPS", draw_fps_icon)

    # --- TIME --- (time on left half, battery on right half; each centered in its half)
    left_half_w = time_w // 2
    right_half_w = time_w - left_half_w
    # Time (clock + text) centered in left half
    (time_label_w, _), _ = cv2.getTextSize(time_txt, font, scale, 1)
    time_content_w = icon_w + gap_icon_text + time_label_w
    time_start_x = x_time + (left_half_w - time_content_w) // 2
    time_icon_cx = time_start_x + icon_radius
    draw_clock_icon(time_icon_cx, cy)
    cv2.putText(frame, time_txt, (time_start_x + icon_w + gap_icon_text, text_y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

    # Battery centered in right half (main view only; gallery draws its own battery)
    if not button_state.gallery_open:
        bat_w = 28 + 4  # BATTERY_BODY_W + BATTERY_TIP_W
        bat_h = 14
        bat_x = x_time + left_half_w + (right_half_w - bat_w) // 2
        bat_y = y + (pill_h - bat_h) // 2
        draw_battery_icon(frame, x=bat_x, y=bat_y, percent=battery_percent)

    rects = HudRects(
        net=(x_net,  y, net_w,  pill_h),
        fps=(x_fps,  y, fps_w,  pill_h),
        time=(x_time, y, time_w, pill_h),
    )

    # Expanded panel (only when icon clicked)
    panel_y = y + pill_h + 6

    def panel(lines, anchor_x):
        line_h = 18
        panel_w = max(220, max(tw(s, 0.5, 1) for s in lines) + 20)
        panel_h = 10 + line_h * len(lines)
        _draw_pill(frame, anchor_x, panel_y, panel_w, panel_h)  # same opacity as menu/bottom HUD (HUD_MENU_OPACITY)
        yy = panel_y + 22
        for s in lines:
            cv2.putText(frame, s, (anchor_x + 10, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)
            yy += line_h

    # Build extra info
    if open_panel == "time":
        bat_pct = battery_percent if battery_percent is not None else 100
        panel([
            f"Uptime: {elapsed_s:0.1f}s",
            f"Frame: {frame_count}",
            f"Source: {source_label}",
            f"Battery: {bat_pct}%",
        ], rects.time[0])

    elif open_panel == "fps":
        panel([
            f"Mode: {fps_mode}",
            f"dt: {1000.0/max(1e-6, (1.0/max(1e-6, fps_ema))):0.1f} ms (approx)",
            "Tip: MAX mode removes throttle",
        ], rects.fps[0])

    elif open_panel == "net":
        mbps_bytes = bytes_per_s / 1e6
        lines = [
            f"{mbps_bytes:0.2f} MB/s  ({mbps_bits:0.1f} Mb/s)",
        ]
        if source_label.startswith("SPI"):
            mhz = (getattr(source_stats, "sclk_hz_rep", 0.0) / 1e6) if getattr(source_stats, "sclk_hz_rep", 0) else 0.0
            lines += [
                f"SPI: {mhz:0.0f} MHz",
                f"ok: {getattr(source_stats,'frames_ok',0)}  badParse: {getattr(source_stats,'bad_parse',0)}  badCRC: {getattr(source_stats,'bad_crc',0)}",
            ]
            last_err = getattr(source_stats, "last_err", "")
            if last_err:
                lines.append(f"lastErr: {last_err[:40]}")
        panel(lines, rects.net[0])

    return rects

def handle_hud_click(mx: int, my: int, rects: HudRects, open_panel: str) -> str:
    """
    Toggle which panel is open based on click.
    """
    if _in_rect(mx, my, rects.net):
        return "" if open_panel == "net" else "net"
    if _in_rect(mx, my, rects.fps):
        return "" if open_panel == "fps" else "fps"
    if _in_rect(mx, my, rects.time):
        return "" if open_panel == "time" else "time"
    return open_panel
