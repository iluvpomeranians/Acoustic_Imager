# STM32G473VGTX Firmware Build & Configuration Guide
This document describes how to build, flash, and maintain the firmware for the STM32G473-based Acoustic Imager project.

---

# 1. Prerequisites

## ARM GCC Toolchain (Required)
Install the ARM embedded GCC toolchain for cross-compilation:

### Ubuntu/Debian
```
sudo apt update
sudo apt install gcc-arm-none-eabi gdb-arm-none-eabi
```

### macOS (Homebrew)
```
brew install --cask gcc-arm-embedded
```

### Windows
Download and install from:
https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm

---

## ST-Link Tools (Flashing)
### Ubuntu/Debian
```
sudo apt install stlink-tools
```

### macOS
```
brew install stlink
```

### Windows
Download from:
https://github.com/stlink-org/stlink/releases

---

## OpenOCD (Debugging)
### Ubuntu/Debian
```
sudo apt install openocd
```

### macOS
```
brew install openocd
```

---

# 2. Building the Firmware

The STM32 firmware lives under:

```
embedded/STM32_G4/
```

## Using CMake (Recommended)

```
cd embedded/STM32_G4/
mkdir build
cd build

# Configure
cmake ..

# Build
make -j8

# Flash (requires ST-Link)
make flash
```

## Using Makefile (If you prefer manual builds)

```
make
make flash       # flash to the STM32G473
make clean
```

---

# 3. STM32CubeMX Integration

The firmware configuration is defined by:

```
embedded/STM32_G4/G4.ioc
```

Open it using STM32CubeMX.

## 1. Generate CubeMX Project Files
Inside CubeMX:

1. Open `G4.ioc`
2. Ensure MCU = **STM32G473VGTX**
3. Configure peripherals (ADC, DMA, GPIO, SPI/I2S as needed)
3. Project Manager → Toolchain / IDE = **Makefile**
4. Generate Code
   *(This will regenerate `Core/` and `Drivers/` inside STM32_G4/)*

---

## 2. Updating CMakeLists.txt After Code Generation
After regenerating code, verify that:

### HAL sources are included:
- `Drivers/STM32G4xx_HAL_Driver/Src/*.c`
- `Drivers/CMSIS/*`
- `Core/Src/*.c`

### Startup file:
```
startup_stm32g473xx.s
```

### Linker script:
```
STM32G473VGTX_FLASH.ld
```

### Your project-specific sources:
```
acquisition/
preprocessing/
fft/
main.cpp
```

Update include paths if CubeMX adds or modifies folders.

---

# 4. Directory Structure (Final Expected Layout)

```
embedded/
└── STM32_G4/
    ├── Core/
    │   ├── Inc/
    │   ├── Src/
    │   └── startup_stm32g473xx.s
    ├── Drivers/
    │   ├── CMSIS/
    │   └── STM32G4xx_HAL_Driver/
    ├── acquisition/
    ├── preprocessing/
    ├── fft/
    ├── CMakeLists.txt
    ├── G4.ioc
    ├── main.cpp
    ├── STM32G473VGTX_FLASH.ld
    ├── README.md
    └── test.cpp
```

---

# 5. Debugging

## Using OpenOCD + GDB

Start OpenOCD:

```
make debug
```

In another terminal:

```
arm-none-eabi-gdb build/firmware.elf
(gdb) target remote localhost:3333
(gdb) monitor reset halt
(gdb) load
(gdb) continue
```

---

# 6. Troubleshooting

### Toolchain Not Found
Verify installation:
```
arm-none-eabi-gcc --version
```

### Linker Script Errors
Ensure:
- `STM32G473VGTX_FLASH.ld` exists
- `CMakeLists.txt` points to it

### HAL Errors
Regenerate firmware from CubeMX (`G4.ioc`)

### Flash Fails
Check:
- ST-Link connected
- Board powered correctly
- Try:
```
make erase
make flash
```

### Memory Usage
STM32G473 has:
- **128 KB Flash**
- **40 KB SRAM**

Check usage:
```
make size
```

---

# 7. Notes
- The old STM32H7 configuration has been removed.
- Only the **G4.ioc** file is authoritative for firmware generation.
- The project uses CMake builds by default; Makefile-based builds are optional.
