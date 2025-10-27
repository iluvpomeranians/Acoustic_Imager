#ifndef DATA_ACQUISITION_H
#define DATA_ACQUISITION_H

#include <stdint.h>
#include <stdbool.h>
#include "stm32h7xx_hal.h"

// Configuration constants for 4x4 MEMS microphone array
#define MIC_ARRAY_SIZE         16          // 4x4 = 16 microphones
#define SAMPLES_PER_CHANNEL    1024        // samples per channel per frame
#define TOTAL_SAMPLES_PER_FRAME (MIC_ARRAY_SIZE * SAMPLES_PER_CHANNEL)
#define SAMPLE_RATE_HZ         16000       // 16kHz sampling rate
#define BITS_PER_SAMPLE        16          // 16-bit samples

// Buffer management
#define DMA_BUFFER_SIZE        (TOTAL_SAMPLES_PER_FRAME * 2)  // Double buffering

// Status codes
typedef enum {
    ACQUISITION_OK = 0,
    ACQUISITION_ERROR,
    ACQUISITION_BUSY,
    ACQUISITION_TIMEOUT
} AcquisitionStatus_t;

// Function declarations
AcquisitionStatus_t Audio_InitAcquisition(void);
AcquisitionStatus_t Audio_StartAcquisition(void);
AcquisitionStatus_t Audio_StopAcquisition(void);
bool Audio_GetFrame(int16_t *destBuffer);
bool Audio_IsDataReady(void);
void Audio_ClearDataReady(void);
uint32_t Audio_GetSampleRate(void);
uint16_t Audio_GetChannelCount(void);

// Error handling
void Audio_ErrorHandler(void);

#endif
