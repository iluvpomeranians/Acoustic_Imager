/**
 * @file app_main.c
 * @brief Main application entry point for STM32G4 Acoustic Imager project
 * @details This file contains the main application logic and initialization
 */

/* =========================================================================
 * INCLUDES
 * ========================================================================= */
#include "main.h"
#include "adc.h"
#include "dma.h"
#include "spi.h"
#include "tim.h"
#include "usart.h"
#include "usb_device.h"
#include "gpio.h"

#include "arm_math.h"

#include "app_main.h"

#include "protocol/spi_protocol.h"
#include "transport/spi_stream.h"
#include "transport/spi_stream_test.h"

#include "usb/usb_debug.h"
#include "dsp/dsp_pipeline.h"

/* =========================================================================
 * DEFINES & CONSTANTS
 * ========================================================================= */
#define ADC_DMA_BUF_SIZE (2 * N_CH_PER_ADC * FRAME_SIZE) // 4096 samples per half-buffer

/* =========================================================================
 * STATIC VARIABLES
 * ========================================================================= */
// Each ADC gets dedicated DMA circular buffer (2 * 4 * 1024 = 8192 samples total)
// Half-buffer = 4096 samples (at 48kHz = 85.2ms)
static uint16_t adc1_buf[ADC_DMA_BUF_SIZE];
static uint16_t adc2_buf[ADC_DMA_BUF_SIZE];
static uint16_t adc3_buf[ADC_DMA_BUF_SIZE];
static uint16_t adc4_buf[ADC_DMA_BUF_SIZE];

// FFT Output: Frequency domain results (1024-point FFT → 512 bins per ADC)
static float fft_out_adc1[2 * N_BINS];
static float fft_out_adc2[2 * N_BINS];
static float fft_out_adc3[2 * N_BINS];
static float fft_out_adc4[2 * N_BINS];

static arm_rfft_fast_instance_f32 fft_instance;

static uint8_t spi_tx_buffer[SPI_PACKET_SIZE];
static spi_stream_t spi_stream_ctx;

static uint32_t _print_counter = 0;

// Ping-pong processing flags
// Bit assignment: [4-7]=Full flags (ADC1-4), [0-3]=Half flags (ADC1-4)
volatile uint32_t adc_ready_mask = 0;
volatile uint8_t fft_in_progress = 0;  // Prevent overlapping FFT calculations

// Track which half of buffer is ready for FFT
// ready_half: 0=first half (0-1023), 1=second half (1024-2047)
volatile uint8_t ready_half[4] = {0};  // One per ADC

// Interrupt counters (for debug / watch)
volatile uint32_t irq_events = 0;            // total ADC IRQ events
volatile uint32_t irq_count_adc[4] = {0};    // per-ADC IRQ counters

/* =========================================================================
 * FORWARD DECLARATIONS
 * ========================================================================= */

/* static void module_init(void); */
/* static void module_process(void); */

/* =========================================================================
 * PUBLIC FUNCTIONS
 * ========================================================================= */

uint32_t value = 0;
float voltage;

// TODO: Deprecated
typedef struct {
    uint8_t header;
    uint8_t adc_id;
    uint32_t frame_counter;
    float fft_data[512]; // TODO: Correct MACRO use
    uint16_t checksum;
} __attribute__((packed)) SPI_Packet_t;

void usb_cdc_smoke_test() {
  const char *test_str = "Hello from USB CDC!\r\n";
  usb_printf("%s", test_str);
}

void app_init(void) {
  // Initialize FFT instance (precompute twiddle factors, etc.)
  arm_rfft_fast_init_f32(&fft_instance, FRAME_SIZE);
  spi_stream_init(&spi_stream_ctx);
}

void app_start(void) {

  // Any additional startup logic can go here
  HAL_Delay(1000);  // Wait for peripherals to stabilize

  // Set GAIN_CNTL to HIGH
  HAL_GPIO_WritePin(GPIOD, GPIO_PIN_9, GPIO_PIN_SET);

  // Calibrate all ADCs before use
  HAL_ADCEx_Calibration_Start(&hadc1, (uint32_t)ADC_SINGLE_ENDED);
  HAL_ADCEx_Calibration_Start(&hadc2, (uint32_t)ADC_SINGLE_ENDED);
  HAL_ADCEx_Calibration_Start(&hadc3, (uint32_t)ADC_SINGLE_ENDED);
  HAL_ADCEx_Calibration_Start(&hadc4, (uint32_t)ADC_SINGLE_ENDED);


  // Start Timer6 (triggers all ADCs synchronously)
  HAL_TIM_Base_Start(&htim6);

  // Start all 4 ADCs with DMA (2048 samples = 1024×2 ping-pong)
  HAL_ADC_Start_DMA(&hadc1, (uint32_t*)adc1_buf, (uint32_t)(ADC_DMA_BUF_SIZE));
  HAL_ADC_Start_DMA(&hadc2, (uint32_t*)adc2_buf, (uint32_t)(ADC_DMA_BUF_SIZE));
  HAL_ADC_Start_DMA(&hadc3, (uint32_t*)adc3_buf, (uint32_t)(ADC_DMA_BUF_SIZE));
  HAL_ADC_Start_DMA(&hadc4, (uint32_t*)adc4_buf, (uint32_t)(ADC_DMA_BUF_SIZE));

  usb_dbg_init(16, 200);  // avg over 16 prints, print every 200 ms

}

/**
 * @brief Main application loop
 * @return void
 */
void app_loop(void) {
  // Main application loop - check for ADC data ready and process
  
  // Clear the global mask immediately to avoid missing new events
  __disable_irq();
  uint32_t local_mask = adc_ready_mask;
  adc_ready_mask = 0;
  __enable_irq();
  // Process the ready half-buffer for this ADC


  // TODO: Instead of processing to an output buffer, we could
  // set the SPI payload buffer as the output of the ADC processing
  // TODO: The implementation is process_adc_pipeline() is currently
  // inccorect; we either need a 4-channel output buffer or we need to
  // send each channel's FFT output sequentially over SPI in this
  // function.
  // TODO: parameters N_CH_PER_ADC, FRAME_SIZE, 48000, N_BINS are
  // currently dummy values

  // FFT Processing Pipeline
  // When any ADC half-buffer completes, process that acoustic window
  if (local_mask && !fft_in_progress) {
    fft_in_progress = 1;

    uint8_t have_offset = 0;
    uint32_t half_offset_for_print = 0;
    // Check each ADC for ready half-buffers
    for (uint8_t adc = 0; adc < 4; adc++) {
      uint8_t half_flag = (local_mask >> adc) & 1;       // Half-complete
      uint8_t full_flag = (local_mask >> (adc + 4)) & 1; // Full-complete
      
      if (half_flag || full_flag) {
        uint16_t *buf_ptr = NULL;
        float *fft_out_ptr = NULL;
        uint32_t half_offset = 0;
        
        if (half_flag) {
          half_offset = 0;
          ready_half[adc] = 0;
        } else {
          half_offset = ADC_DMA_BUF_SIZE / 2; 
          ready_half[adc] = 1;
        }
      
        if (!have_offset) {
            have_offset = 1;
            half_offset_for_print = half_offset;
        }

        if (adc == 0) { buf_ptr = adc1_buf; fft_out_ptr = fft_out_adc1; }
        else if (adc == 1) { buf_ptr = adc2_buf; fft_out_ptr = fft_out_adc2; }
        else if (adc == 2) { buf_ptr = adc3_buf; fft_out_ptr = fft_out_adc3; }
        else { buf_ptr = adc4_buf; fft_out_ptr = fft_out_adc4; }
        
        if (buf_ptr && fft_out_ptr) {
          // Process the ready half-buffer for this ADC
          uint16_t *active_half = buf_ptr + half_offset;

          // TODO: The implementation is process_adc_pipeline() is currently
          // incorrect; we either need a 4-channel output buffer or we need to
          // send each channel's FFT output sequentially over SPI in this
          // function.
          process_adc_pipeline(&fft_instance, active_half, adc, fft_out_ptr);

          // TODO: parameters N_CH_PER_ADC, FRAME_SIZE, 48000, N_BINS are
          // currently dummy values
          spi_stream_build_fft_packet(&spi_stream_ctx, 
                                      spi_tx_buffer, 
                                      SPI_PACKET_SIZE,
                                      adc,
                                      fft_out_ptr,
                                      N_MICS,
                                      FRAME_SIZE,
                                      SAMPLE_RATE,
                                      N_BINS);
          // package_adc_for_spi(adc, fft_out_ptr, spi_tx_buffer, N_CH_PER_ADC,
          //                     FRAME_SIZE, 48000, N_BINS);

          spi_stream_tx_blocking(spi_tx_buffer, SPI_PACKET_SIZE);
          // transmit_spi_packet(spi_tx_buffer, SPI_PACKET_SIZE);
        }

        if (have_offset) {
        usb_dbg_push_adc_window_4adc(&adc1_buf[half_offset_for_print],
                                     &adc2_buf[half_offset_for_print],
                                     &adc3_buf[half_offset_for_print],
                                     &adc4_buf[half_offset_for_print],
                                     FRAME_SIZE);
        }
      }
    }
    fft_in_progress = 0;
  }
  
  if (++_print_counter >= 10000) {
    _print_counter = 0;
    spi_stream_unit_test_build_packet();
    // usb_cdc_smoke_test();
  }
  HAL_Delay(1);  
}


  /* Periodic debug print of IRQ counters over UART (every ~1000 iterations) */
  
  
 



void test_spi_stream_loop(void) {
  HAL_Delay(1000);
  spi_stream_unit_test_build_packet();
  spi_stream_unit_test_nulls();
  spi_stream_unit_test_small_cap();
  spi_stream_unit_test_frame_counter();
  
}

/* ============================================================================
 * STATIC FUNCTIONS
 * ============================================================================ */

/**
 * @brief Static helper function description
 * @return void
 */
  