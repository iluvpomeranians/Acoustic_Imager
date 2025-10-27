cmake_minimum_required(VERSION 3.16)

# Project name and version
project(STM32_MEMS_Array VERSION 1.0.0 LANGUAGES C CXX ASM)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# =============================================================================
# Toolchain Configuration
# =============================================================================

# Set the system processor
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR ARM)

# Specify the cross compiler
set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)
set(CMAKE_AR arm-none-eabi-ar)
set(CMAKE_OBJCOPY arm-none-eabi-objcopy)
set(CMAKE_OBJDUMP arm-none-eabi-objdump)
set(CMAKE_SIZE arm-none-eabi-size)

# =============================================================================
# STM32 Configuration
# =============================================================================

# STM32H723VGT6 specific settings
set(MCU_FAMILY STM32H7xx)
set(MCU_MODEL STM32H723xx)
set(MCU_FPU fpv5-d16)
set(MCU_FLOAT_ABI hard)

# CPU flags
set(CPU_FLAGS "-mcpu=cortex-m7 -mthumb -mfpu=${MCU_FPU} -mfloat-abi=${MCU_FLOAT_ABI}")

# =============================================================================
# Compiler Flags
# =============================================================================

# Common flags
set(COMMON_FLAGS
    ${CPU_FLAGS}
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-missing-field-initializers
    -fdata-sections
    -ffunction-sections
    -fstack-usage
    -fno-common
    -fmessage-length=0
)

# Debug flags
set(DEBUG_FLAGS
    -g3
    -gdwarf-2
    -O0
    -DDEBUG
)

# Release flags
set(RELEASE_FLAGS
    -Os
    -DNDEBUG
    -flto
)

# Linker flags
set(LINKER_FLAGS
    ${CPU_FLAGS}
    -specs=nano.specs
    -T${CMAKE_SOURCE_DIR}/STM32H723VGTx_FLASH.ld
    -Wl,--gc-sections
    -Wl,--print-memory-usage
    -Wl,--start-group
    -lc
    -lm
    -Wl,--end-group
)

# Apply flags
set(CMAKE_C_FLAGS "${COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS "${COMMON_FLAGS}")
set(CMAKE_ASM_FLAGS "${COMMON_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${LINKER_FLAGS}")

# Debug/Release specific flags
set(CMAKE_C_FLAGS_DEBUG "${DEBUG_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${DEBUG_FLAGS}")
set(CMAKE_ASM_FLAGS_DEBUG "${DEBUG_FLAGS}")

set(CMAKE_C_FLAGS_RELEASE "${RELEASE_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${RELEASE_FLAGS}")
set(CMAKE_ASM_FLAGS_RELEASE "${RELEASE_FLAGS}")

# =============================================================================
# Definitions
# =============================================================================

add_definitions(
    -D${MCU_FAMILY}
    -D${MCU_MODEL}
    -DUSE_HAL_DRIVER
    -DSTM32H723xx
    -DHSE_VALUE=25000000
    -DHSI_VALUE=64000000
    -DLSE_VALUE=32768
    -DLSI_VALUE=32000
    -DVDD_VALUE=3300
    -DTICK_INT_PRIORITY=0
    -DUSE_RTOS=0
    -DPREFETCH_ENABLE=1
    -DINSTRUCTION_CACHE_ENABLE=1
    -DDATA_CACHE_ENABLE=1
)

# =============================================================================
# Include Directories
# =============================================================================

include_directories(
    # Project headers
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/acquisition
    ${CMAKE_SOURCE_DIR}/preprocessing
    ${CMAKE_SOURCE_DIR}/fft
    
    # STM32 HAL includes (adjust paths as needed)
    ${CMAKE_SOURCE_DIR}/Drivers/STM32H7xx_HAL_Driver/Inc
    ${CMAKE_SOURCE_DIR}/Drivers/STM32H7xx_HAL_Driver/Inc/Legacy
    ${CMAKE_SOURCE_DIR}/Drivers/CMSIS/Device/ST/STM32H7xx/Include
    ${CMAKE_SOURCE_DIR}/Drivers/CMSIS/Include
    
    # Generated files from CubeMX (adjust paths as needed)
    ${CMAKE_SOURCE_DIR}/Core/Inc
    ${CMAKE_SOURCE_DIR}/Core/Src
)

# =============================================================================
# Source Files
# =============================================================================

# Project source files
set(PROJECT_SOURCES
    main.cpp
    acquisition/data_acquisition.cpp
)

# Project header files (for IDE support)
set(PROJECT_HEADERS
    acquisition/data_acquisition.hpp
    preprocessing/preprocessing.hpp
    fft/fft_computation.hpp
)

# STM32 HAL source files (adjust paths as needed)
set(HAL_SOURCES
    # Add HAL source files here when CubeMX generates them
    # Examples:
    # Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal.c
    # Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_cortex.c
    # Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_i2s.c
    # Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_i2s_ex.c
    # Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_dma.c
    # Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_dma_ex.c
    # Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_gpio.c
    # Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_rcc.c
    # Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_rcc_ex.c
    # Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_pwr.c
    # Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_pwr_ex.c
    # Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_flash.c
    # Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_flash_ex.c
    # Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_tim.c
    # Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_tim_ex.c
)

# CubeMX generated source files (adjust paths as needed)
set(CUBEMX_SOURCES
    # Add CubeMX generated source files here
    # Examples:
    # Core/Src/main.c
    # Core/Src/stm32h7xx_it.c
    # Core/Src/stm32h7xx_hal_msp.c
    # Core/Src/system_stm32h7xx.c
)

# Startup file
set(STARTUP_SOURCE
    # Add startup file here when CubeMX generates it
    # Example:
    # Core/Src/startup_stm32h723vgtx.s
)

# =============================================================================
# Create Executable
# =============================================================================

# Create the executable
add_executable(${PROJECT_NAME}.elf
    ${PROJECT_SOURCES}
    ${PROJECT_HEADERS}
    ${HAL_SOURCES}
    ${CUBEMX_SOURCES}
    ${STARTUP_SOURCE}
)

# =============================================================================
# Post-build Commands
# =============================================================================

# Generate hex file
add_custom_command(TARGET ${PROJECT_NAME}.elf POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O ihex $<TARGET_FILE:${PROJECT_NAME}.elf> ${PROJECT_NAME}.hex
    COMMENT "Generating HEX file"
)

# Generate binary file
add_custom_command(TARGET ${PROJECT_NAME}.elf POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:${PROJECT_NAME}.elf> ${PROJECT_NAME}.bin
    COMMENT "Generating BIN file"
)

# Display size information
add_custom_command(TARGET ${PROJECT_NAME}.elf POST_BUILD
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:${PROJECT_NAME}.elf>
    COMMENT "Memory usage:"
)

# =============================================================================
# Custom Targets
# =============================================================================

# Flash target (adjust programmer as needed)
add_custom_target(flash
    COMMAND st-flash write ${PROJECT_NAME}.bin 0x08000000
    DEPENDS ${PROJECT_NAME}.elf
    COMMENT "Flashing to STM32"
)

# Erase target
add_custom_target(erase
    COMMAND st-flash erase
    COMMENT "Erasing STM32 flash"
)

# Debug target (adjust debugger as needed)
add_custom_target(debug
    COMMAND openocd -f interface/stlink.cfg -f target/stm32h7x.cfg
    COMMENT "Starting OpenOCD debug server"
)

# =============================================================================
# Print Configuration
# =============================================================================

message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "MCU: ${MCU_MODEL}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C Flags: ${CMAKE_C_FLAGS}")
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Linker Flags: ${CMAKE_EXE_LINKER_FLAGS}")
