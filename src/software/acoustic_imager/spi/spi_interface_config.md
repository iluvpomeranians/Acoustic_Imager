# Acoustic Imager config reference

See [config.py](config.py) for the live configuration. This doc summarizes the two SPI interface modes and their pinouts.

## SPI interface modes

Set `SPI_INTERFACE` in config.py to `0` or `1` to choose the Pi SPI bus and chip-select. **J2** is the STM32-side header (lines 1–10) per the ribbon cable pinout.

| Item | SPI INTERFACE 2 | SPI INTERFACE 1 |
|------|---------------|---------------|
| **config.py** |  `SPI_INTERFACE = 0` | `SPI_INTERFACE = 1` |
| **Device** | `/dev/spidev0.0` | `/dev/spidev1.2` |
| **SPI_BUS / SPI_DEV** | 0, 0 | 1, 2 |

---

| Purpose   | STM-32 PINOUT (J2) | RASPI PINOUTS (SPI0) |RASPI PINOUTS (SPI1) |
|------|--------------|---------------|---------------|
| Gain control (AUTO_GAIN_CNTL) | 1 | 22 | 22 |
| Frame-ready (MCU_STATUS) | 2 | 26 | 26 |
| RPI status | 3 | 27 | 27 |
| GND | 4 | 20 | 25 |
| SCLK | 5 | 23 | 40 |
| Chip select (NSS) | 6 | 24 | 36 |
| MISO | 7 | 21 | 35 |
| MOSI | 8 | 19 | 38 |
| GND | 9 | 25 | 39 |
| NRST | 10 | 28 | 28 |

Notes:

- **SPI0** uses the Pi's main 40-pin header (pins 19, 21, 23, 24).
- **SPI1** is the auxiliary SPI; enable with `dtoverlay=spi1-3cs` in `/boot/config.txt`. CE2 gives `/dev/spidev1.2`.
- Frame-ready (MCU_STATUS) is an input to the Pi (STM32 drives it). Gain control (AUTO_GAIN_CNTL) is an output from the Pi (drives STM32).

