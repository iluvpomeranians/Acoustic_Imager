from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import time
import cv2
import numpy as np

@dataclass
class HudRects:
    time: Tuple[int,int,int,int]
    fps:  Tuple[int,int,int,int]
    net:  Tuple[int,int,int,int]

def _draw_pill(frame: np.ndarray, x: int, y: int, w: int, h: int,
               bg=(0,0,0), alpha: float = 0.35, border=(255,255,255)) -> None:
    H, W = frame.shape[:2]
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x + w); y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return

    roi = frame[y0:y1, x0:x1]
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
) -> HudRects:
    """
    Draw compact HUD top-left. Returns hit rects for mouse click handling.
    """
    if details_level == "OFF":
        return HudRects((0,0,0,0),(0,0,0,0),(0,0,0,0))

    h, w, _ = frame.shape

    # Layout constants
    pad = 110
    y = 1
    pill_h = 34
    icon_w = 34
    gap = 8

    # Compute throughput from fps_ema
    bytes_per_s = frame_bytes * fps_ema
    mbps_bits = (bytes_per_s * 8) / 1e6

    time_txt = time.strftime("%I:%M:%S%P")
    fps_txt  = f"{fps_ema:4.1f}"
    net_txt  = f"{mbps_bits:4.1f}"

    # “Icons” (no external assets): simple glyphs inside circles
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

    time_w = 170
    fps_w  = 165
    net_w  = 165

    x_time = pad
    x_fps  = x_time + time_w + gap
    x_net  = x_fps  + fps_w  + gap

    # Draw pills
    _draw_pill(frame, x_time, y, time_w, pill_h)
    _draw_pill(frame, x_fps,  y, fps_w,  pill_h)
    _draw_pill(frame, x_net,  y, net_w,  pill_h)

    cy = y + pill_h // 2
    text_y = y + 23  # consistent baseline

    # --- TIME ---
    draw_clock_icon(x_time + 18, cy)
    cv2.putText(
        frame,
        time_txt,
        (x_time + 40, text_y),   # text positioned after icon
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),         # WHITE text
        1,
        cv2.LINE_AA
    )

    # --- FPS ---
    draw_fps_icon(x_fps + 18, cy)
    cv2.putText(
        frame,
        f"{fps_txt} FPS",
        (x_fps + 40, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # --- NETWORK ---
    draw_net_icon(x_net + 18, cy)
    cv2.putText(
        frame,
        f"{net_txt} Mb/s",
        (x_net + 40, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    rects = HudRects(
        time=(x_time, y, time_w, pill_h),
        fps=(x_fps,  y, fps_w,  pill_h),
        net=(x_net,  y, net_w,  pill_h),
    )

    # Expanded panel (only when icon clicked)
    panel_y = y + pill_h + 6

    def panel(lines, anchor_x):
        line_h = 18
        panel_w = max(220, max(tw(s, 0.5, 1) for s in lines) + 20)
        panel_h = 10 + line_h * len(lines)
        _draw_pill(frame, anchor_x, panel_y, panel_w, panel_h, alpha=0.45)
        yy = panel_y + 22
        for s in lines:
            cv2.putText(frame, s, (anchor_x + 10, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)
            yy += line_h

    # Build extra info
    if open_panel == "time":
        panel([
            f"Uptime: {elapsed_s:0.1f}s",
            f"Frame: {frame_count}",
            f"Source: {source_label}",
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
    if _in_rect(mx, my, rects.time):
        return "" if open_panel == "time" else "time"
    if _in_rect(mx, my, rects.fps):
        return "" if open_panel == "fps" else "fps"
    if _in_rect(mx, my, rects.net):
        return "" if open_panel == "net" else "net"
    return open_panel