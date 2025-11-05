# STM32H723VGT6 Build Configuration Guide

## Prerequisites

### 1. ARM GCC Toolchain
Install the ARM GCC toolchain for cross-compilation:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install gcc-arm-none-eabi gdb-arm-none-eabi
```

**macOS (using Homebrew):**
```bash
brew install --cask gcc-arm-embedded
```

**Windows:**
Download from: https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm

### 2. ST-Link Tools (for flashing)
**Ubuntu/Debian:**
```bash
sudo apt install stlink-tools
```

**macOS:**
```bash
brew install stlink
```

**Windows:**
Download from: https://github.com/stlink-org/stlink/releases

### 3. OpenOCD (for debugging)
**Ubuntu/Debian:**
```bash
sudo apt install openocd
```

**macOS:**
```bash
brew install openocd
```

## Build Instructions

### Using CMake (Recommended)
```bash
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build
make

# Flash to device
make flash
```

### Using Makefile
```bash
# Build project
make

# Flash to device
make flash

# Clean build files
make clean
```

## CubeMX Integration

### 1. Generate Project Files
1. Open STM32CubeMX
2. Create new project for STM32H723VGT6
3. Configure I2S2, DMA, GPIO pins as specified in `configuringSTM.txt`
4. Generate code

### 2. Update Build Files
After generating CubeMX files, update the build configuration:

**For CMakeLists.txt:**
- Uncomment HAL source files in `HAL_SOURCES`
- Uncomment CubeMX source files in `CUBEMX_SOURCES`
- Add startup file to `STARTUP_SOURCE`
- Update include paths if needed

**For Makefile:**
- Uncomment HAL source files in `SOURCES`
- Uncomment CubeMX source files in `SOURCES`
- Update include paths in `INCLUDES`

### 3. Directory Structure After CubeMX
```
STM32/
├── CMakeLists.txt
├── Makefile
├── STM32H723VGTx_FLASH.ld
├── main.cpp
├── acquisition/
│   ├── data_acquisition.cpp
│   └── data_acquisition.hpp
├── preprocessing/
│   └── preprocessing.hpp
├── fft/
│   └── fft_computation.hpp
├── Core/
│   ├── Inc/
│   │   ├── main.h
│   │   ├── stm32h7xx_hal_conf.h
│   │   └── stm32h7xx_it.h
│   └── Src/
│       ├── main.c
│       ├── stm32h7xx_it.c
│       ├── stm32h7xx_hal_msp.c
│       ├── system_stm32h7xx.c
│       └── startup_stm32h723vgtx.s
└── Drivers/
    ├── STM32H7xx_HAL_Driver/
    └── CMSIS/
```

## Debugging

### Using OpenOCD + GDB
1. Start OpenOCD:
   ```bash
   make debug
   ```

2. In another terminal, start GDB:
   ```bash
   arm-none-eabi-gdb build/STM32_MEMS_Array.elf
   (gdb) target remote localhost:3333
   (gdb) monitor reset halt
   (gdb) load
   (gdb) continue
   ```

### Using STM32CubeIDE
1. Import project into STM32CubeIDE
2. Configure debug settings
3. Use built-in debugger

## Troubleshooting

### Common Issues

1. **Toolchain not found:**
   - Ensure ARM GCC is in your PATH
   - Check installation with: `arm-none-eabi-gcc --version`

2. **Linker script not found:**
   - Generate linker script from CubeMX
   - Update path in CMakeLists.txt or Makefile

3. **HAL library not found:**
   - Generate HAL files from CubeMX
   - Update include paths and source files

4. **Flash programming fails:**
   - Check ST-Link connection
   - Ensure device is in programming mode
   - Try: `make erase` then `make flash`

### Memory Usage
The STM32H723VGT6 has:
- 1MB Flash memory
- 1MB RAM
- Use `make size` to check memory usage
