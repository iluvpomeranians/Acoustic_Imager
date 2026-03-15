# Delay / Latency Improvements Reference

Reference for the acoustic imager pipeline timing and where to optimize next. Check this before starting latency work.

## Pipeline stages (typical ms, one run)

| Stage         | ~ms   | Notes                          |
|---------------|-------|--------------------------------|
| heat_music    | 16–19 | MUSIC / DSP; highest cost      |
| blend         | 13–15 | Heatmap blend (already optimized) |
| bars          | 12–13 | DB + freq bar drawing          |
| heat_draw     | 11–12 | Heatmap drawing                |
| waitKey       | 11–13 | UI event loop (often fixed)    |
| heat_scale    | 11–20 | Scaling (high variance)        |
| heat          | 10–13 | Heatmap pipeline (pre-draw)   |
| ui            | 8–9   | Rest of UI drawing            |
| bg            | 2.7–3 | Background                     |
| imshow        | <1    | Display                        |
| read, heat_stability | <0.1 | Negligible              |

**Total** ~106–114 ms per frame.

## Priority order for optimization

1. **heat_music** — Largest; any win here helps most (resolution, algorithm, caching).
2. **bars** — Second-biggest draw cost; look for redundant work or cheaper rendering.
3. **heat_scale** — High and variable; good candidate for caching or simplification.
4. **heat_draw** — Next if still draw-bound.
5. **heat** — Upstream of heat_draw; consider after heat_scale/heat_draw.

`waitKey` is often a fixed 10–15 ms and may be hard to reduce.

## Already done

- **Blend**: Fused colormap+weight LUT (no applyColorMap + first multiply); optional half-res blend (`config.BLEND_HALF_RES`). LUT writes use contiguous single-channel buffers to satisfy OpenCV `cv2.LUT` dst layout.
- **heat_music**: (1) `ANGLES_2D_RESOLUTION` reduced to 35 (from 51) for coarser 2D grid; (2) `SPI_MUSIC_EVERY_N_FRAMES = 2` to run 2D MUSIC every other frame and reuse cached angles/spec; (3) two-stage 2D MUSIC when `SPI_MUSIC_2D_COARSE_RESOLUTION > 0` (e.g. 21): coarse grid then small fine patch around peak via `music_spectrum_2d_refined` in `dsp/beamforming.py`, with `SPI_MUSIC_2D_REFINE_HALF_WIDTH` (e.g. 2 → 5×5 patch). Set `SPI_MUSIC_2D_COARSE_RESOLUTION = 0` to disable two-stage and use full grid only.
