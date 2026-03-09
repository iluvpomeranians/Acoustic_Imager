/**
 * @file dsp_pipeline_test.c
 * @brief Unit tests for DSP pipeline functions
 * @details This file contains unit tests for the DSP pipeline functions defined in dsp_pipeline.c,
 */

/* =========================================================================
 * INCLUDES
 * ========================================================================= */
#include <stdint.h>
#include <math.h>

#include "arm_math.h"
#include "usb/usb_debug.h"
#include "app_main.h"       
#include "dsp_pipeline.h"
/* =========================================================================
 * DEFINES & CONSTANTS
 * ========================================================================= */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TEST_FS_HZ        (48000.0f)
#define TEST_TONE_HZ      (3000.0f)

// Print settings
#define PRINT_MAX_BINS    40   // print bins 0..39
#define PRINT_PEAK_SCAN   512 // scan first N/2 bins for peak

/* =========================================================================
 * STATIC VARIABLES
 * ========================================================================= */
static float s_time[FRAME_SIZE];
static float s_rfft_packed[FRAME_SIZE];         // CMSIS packed output length = N floats
static float s_mag[(FRAME_SIZE / 2) + 1];       // one-sided magnitude length = N/2 + 1
/* =========================================================================
 * FORWARD DECLARATIONS
 * ========================================================================= */

 static void rfft_packed_to_mag(const float *packed, float *mag, uint32_t N);
/* static void module_init(void); */
/* static void module_process(void); */

/* =========================================================================
 * PUBLIC FUNCTIONS
 * ========================================================================= */

void dsp_unit_test_sine_fft(void)
{
  usb_printf("\r\n--- DSP UNIT TEST: SINE -> RFFT -> MAG ---\r\n");

  // Init FFT
  arm_rfft_fast_instance_f32 fft;
  arm_status st = arm_rfft_fast_init_f32(&fft, FRAME_SIZE);
  if (st != ARM_MATH_SUCCESS) {
      usb_printf("arm_rfft_fast_init_f32 FAILED (status=%d)\r\n", (int)st);
      return;
  }

  // Generate sine: x[n] = sin(2*pi*f*n/fs)
  const float w = 2.0f * (float)M_PI * (TEST_TONE_HZ / TEST_FS_HZ);
  for (uint32_t n = 0; n < FRAME_SIZE; n++) {
      s_time[n] = arm_sin_f32(w * (float)n);
  }

  // Run RFFT (forward)
  arm_rfft_fast_f32(&fft, s_time, s_rfft_packed, 0);

  // Magnitude (one-sided)
  rfft_packed_to_mag(s_rfft_packed, s_mag, FRAME_SIZE);

  // Find peak bin in 0..N/2
  const uint32_t half = FRAME_SIZE >> 1;
  uint32_t k_peak = 0;
  float peak = 0.0f;
  for (uint32_t k = 0; k <= half; k++) {
      if (s_mag[k] > peak) {
          peak = s_mag[k];
          k_peak = k;
      }
  }

  // Expected bin (closest)
  const float bin_res = TEST_FS_HZ / (float)FRAME_SIZE;
  const float k_expected = TEST_TONE_HZ / bin_res;

  usb_printf("fs=%.1f Hz, N=%lu, bin_res=%.3f Hz/bin\r\n",
              (double)TEST_FS_HZ, (unsigned long)FRAME_SIZE, (double)bin_res);
  usb_printf("tone=%.1f Hz, expected_bin=%.2f\r\n",
              (double)TEST_TONE_HZ, (double)k_expected);
  usb_printf("peak_bin=%lu, peak_mag=%.6f\r\n",
              (unsigned long)k_peak, (double)peak);

  // Print first few bins
  usb_printf("\r\nBin\tFreq(Hz)\tMag\r\n");
  uint32_t max_bins = PRINT_MAX_BINS;
  if (max_bins > (half + 1U)) max_bins = half + 1U;

  for (uint32_t k = 0; k < max_bins; k++) {
    float f = (float)k * bin_res;
    usb_printf("%lu\t%.1f\t\t%.6f\r\n",
                (unsigned long)k, (double)f, (double)s_mag[k]);
  }

  // Print a small neighborhood around the peak
  usb_printf("\r\nPeak neighborhood:\r\n");
  uint32_t k0 = (k_peak > 5U) ? (k_peak - 5U) : 0U;
  uint32_t k1 = (k_peak + 5U <= half) ? (k_peak + 5U) : half;

  for (uint32_t k = k0; k <= k1; k++) {
    float f = (float)k * bin_res;
    usb_printf("%lu\t%.1f\t\t%.6f%s\r\n",
                (unsigned long)k, (double)f, (double)s_mag[k],
                (k == k_peak) ? "  <-- PEAK" : "");
  }

  usb_printf("--- END DSP UNIT TEST ---\r\n");
}

/**
 * @brief Public function description
 * @param[in] param1 Description of input parameter
 * @return Description of return value
 */

/* ============================================================================
 * STATIC FUNCTIONS
 * ============================================================================ */

 static void rfft_packed_to_mag(const float *packed, float *mag, uint32_t N)
{
    const uint32_t half = N >> 1; // N/2

    mag[0]    = fabsf(packed[0]); // DC = Re(X[0])
    mag[half] = fabsf(packed[1]); // Nyquist = Re(X[N/2]) for even N

    for (uint32_t k = 1; k < half; k++) {
        float re = packed[2U * k];
        float im = packed[2U * k + 1U];
        mag[k] = sqrtf(re * re + im * im);
    }
}
/**
 * @brief Static helper function description
 * @return void
 */
