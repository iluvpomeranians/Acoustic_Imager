#ifndef DSP_PIPELINE_H
#define DSP_PIPELINE_H

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * INCLUDES
 * ========================================================================= */

#include <stdint.h>

#include "arm_math.h"
/* =========================================================================
 * DEFINES
 * ========================================================================= */

/* =========================================================================
 * TYPE DEFINITIONS
 * ========================================================================= */

/* =========================================================================
 * FUNCTION PROTOTYPES
 * ========================================================================= */

void process_adc_channel_pipeline(arm_rfft_fast_instance_f32 *fft_instance,
                                  uint16_t *adc_raw,
                                  uint8_t channel_index,
                                  float *fft_output);

void process_adc_pipeline(arm_rfft_fast_instance_f32 *fft_instance,
                          uint16_t *adc_raw, 
                          uint32_t adc_id, 
                          float *fft_output);

#ifdef __cplusplus
}
#endif

#endif /* DSP_PIPELINE_H */