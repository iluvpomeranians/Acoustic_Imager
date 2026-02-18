/**
 * @file template.c
 * @brief Brief description of module functionality
 * @details Detailed description of what this module does and its responsibilities
 */

/* =========================================================================
 * INCLUDES
 * ========================================================================= */
#include <stdint.h>
#include <math.h>

#include "arm_math.h"

#include "app_main.h"
#include "dsp_pipeline.h"

/* =========================================================================
 * DEFINES & CONSTANTS
 * ========================================================================= */

/* =========================================================================
 * STATIC VARIABLES
 * ========================================================================= */
static const float adc_scalar = 3.3f / 4095.0f;

static float fft_input_buf[FRAME_SIZE];
static float fft_temp_buf[2048];

/* =========================================================================
 * FORWARD DECLARATIONS
 * ========================================================================= */

/* static void module_init(void); */
/* static void module_process(void); */

/* =========================================================================
 * PUBLIC FUNCTIONS
 * ========================================================================= */

// Process raw ADC data to float voltage values for a specific channel 
// (de-interleaves)
void process_adc_to_float(uint16_t *adc_raw, float *output, uint8_t ch, 
                          uint32_t length) {
  for (uint32_t i = 0; i < length; i++) {
      output[i] = (float)adc_raw[i * N_CH_PER_ADC + ch] * adc_scalar;
  }
}

float calculate_dc_offset(float *data, uint32_t length)
{
  float sum = 0.0f;
  for (uint32_t i = 0; i < length; i++) {
      sum += data[i];
  }
  return sum / (float)length;
}

void remove_dc_bias(float *data, uint32_t length, float dc_offset)
{
    for (uint32_t i = 0; i < length; i++) {
        data[i] -= dc_offset;
    }
}

void apply_fft(arm_rfft_fast_instance_f32 *fft_instance, 
               float *input, 
               float *magnitude_output, 
               uint32_t fft_size) {
    arm_rfft_fast_f32(fft_instance, input, fft_temp_buf, 0);
    
    // TODO: We likely don't need to calculate magnitude, I think the 
    // beamforming algorithm can work directly with complex FFT output.
    uint32_t bin_count = fft_size / 2;
    for (uint32_t k = 0; k < bin_count; k++) {
        float real = fft_temp_buf[2*k];
        float imag = fft_temp_buf[2*k + 1];
        magnitude_output[k] = sqrtf(real*real + imag*imag);
    }
}

void normalize_magnitude(float *mag, uint32_t length)
{
    float max = 0.0f;
    for (uint32_t i = 0; i < length; i++) {
        if (mag[i] > max) max = mag[i];
    }
    
    if (max > 0.0f) {
        for (uint32_t i = 0; i < length; i++) {
            mag[i] /= max;
        }
    }
}

void process_adc_pipeline(arm_rfft_fast_instance_f32 *fft_instance,
                          uint16_t *adc_raw, 
                          uint32_t adc_id, 
                          float *fft_output)
{
  // When a half-buffer is ready, 4 channels of interleaved data are available.
  // Rather than de-interleaving all 4 channels into separate buffers, we can 
  // process each channel is a scratch buffer.
  for (uint8_t ch = 0; ch < N_CH_PER_ADC; ch++) {
  
    process_adc_to_float(adc_raw, fft_input_buf, ch, FRAME_SIZE);
    float dc_offset = calculate_dc_offset(fft_input_buf, FRAME_SIZE);
    remove_dc_bias(fft_input_buf, FRAME_SIZE, dc_offset);
    apply_fft(fft_instance, fft_input_buf, fft_output, FRAME_SIZE);

    // normalize_magnitude(fft_output, 512);
  }
}

/**
 * @brief Public function description
 * @param[in] param1 Description of input parameter
 * @return Description of return value
 */

/* ============================================================================
 * STATIC FUNCTIONS
 * ============================================================================ */

/**
 * @brief Static helper function description
 * @return void
 */
