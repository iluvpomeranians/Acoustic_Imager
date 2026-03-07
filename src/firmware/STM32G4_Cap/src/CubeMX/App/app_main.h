#ifndef APP_MAIN_H
#define APP_MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * INCLUDES
 * ========================================================================= */
#include <stdint.h>
#include <stdbool.h>

/* =========================================================================
 * DEFINES
 * ========================================================================= */
#define APP_VERSION "1.0.0"

#define SAMPLE_RATE 1000
#define N_SAMPLES 2048
#define FRAME_SIZE 512
#define N_ADCS 4
#define N_CH_PER_ADC 4
#define N_MICS 16

#define VALIDATION_MODE 0

// TODO: Consider discarding Nyquist bin (bin 512)
#define N_BINS ((FRAME_SIZE >> 1) + 1) // For real FFT: N_BINS = (N/2)+1 = 513 
#define FFT_FLOATS_PER_MIC (2u * N_BINS)
#define FFT_PAYLOAD_BYTES (FFT_FLOATS_PER_MIC * sizeof(float))

#define APP_USB_CMD_MAX_LEN 64u

// Update the divider ratio once the final battery-sense resistor values are confirmed.
#define BATT_ADC_VREF_MV 2900u
#define BATT_DIVIDER_NUMERATOR 1u
#define BATT_DIVIDER_DENOMINATOR 1u

/* =========================================================================
 * TYPE DEFINITIONS
 * ========================================================================= */
typedef enum {
    APP_STATE_IDLE,
    APP_STATE_RUNNING,
    APP_STATE_ERROR
} app_state_t;

/* =========================================================================
 * FUNCTION PROTOTYPES
 * ========================================================================= */
void app_init(void);
void app_start(void);
void app_loop(void);
void app_usb_cdc_rx(const uint8_t *buf, uint32_t len);
// void app_handle_error(uint32_t error_code);

// Unit test loops
void test_spi_stream_loop(void);
void test_dsp_pipeline_loop(void);

#ifdef __cplusplus
}
#endif

#endif /* APP_MAIN_H */