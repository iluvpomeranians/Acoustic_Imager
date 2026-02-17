/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "adc.h"
#include "dma.h"
#include "spi.h"
#include "tim.h"
#include "usart.h"
#include "usb_device.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "arm_math.h"
#include "protocol/spi_protocol.h"
#include "usb/usb_debug.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define N_SAMPLES 2048
#define FRAME_SIZE 1024
#define N_CH_PER_ADC 4

#define VALIDATION_MODE 0

// TODO: Consider discarding Nyquist bin (bin 512)
#define N_BINS ((FRAME_SIZE >> 1) + 1) // For real FFT: N_BINS = (N/2)+1 = 513 
#define FFT_PAYLOAD_BYTES (N_BINS * sizeof(float)) // 513 bins × 4 bytes/float = 2052 bytes



/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
// Each ADC gets dedicated DMA circular buffer (2048 samples total)
// Half-buffer = 1024 samples (at 48kHz = 21.3ms)
static uint16_t adc1_buf[2 * FRAME_SIZE];
static uint16_t adc2_buf[2 * FRAME_SIZE];
static uint16_t adc3_buf[2 * FRAME_SIZE];
static uint16_t adc4_buf[2 * FRAME_SIZE];

// FFT Output: Frequency domain results (1024-point FFT → 512 bins per ADC)
static float fft_out_adc1[N_BINS];
static float fft_out_adc2[N_BINS];
static float fft_out_adc3[N_BINS];
static float fft_out_adc4[N_BINS];

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

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */


// HAL_SYSCFG_VREFBUF_HighImpedanceConfig(SYSCFG_VREFBUF_HIGH_IMPEDANCE_DISABLE);
// HAL_SYSCFG_EnableVREFBUF();
// IMPORTANT FOR ADC VALUES, DO NOT CHANGE
//C:\Users\Fabz\Desktop\aroundfuckin\Capstone_490_Software\src\firmware\STM32G4_Cap\src\CubeMX\Core\Src\stm32g4xx_hal_msp.c


uint32_t value = 0;
float voltage;
const float adc_scalar = 3.3f / 4095.0f;

static float fft_input_buf[FRAME_SIZE];
static float fft_temp_buf[2048];
static arm_rfft_fast_instance_f32 fft_instance;

static uint32_t spi_frame_counter = 0;
static uint8_t spi_tx_buffer[SPI_PACKET_SIZE];

typedef struct {
    uint8_t header;
    uint8_t adc_id;
    uint32_t frame_counter;
    float fft_data[512]; // TODO: Correct MACRO use
    uint16_t checksum;
} __attribute__((packed)) SPI_Packet_t;

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

void apply_fft(float *input, float *magnitude_output, uint32_t fft_size)
{
    arm_rfft_fast_f32(&fft_instance, input, fft_temp_buf, 0);
    
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

void process_adc_pipeline(uint16_t *adc_raw, uint32_t adc_id, float *fft_output)
{
  // When a half-buffer is ready, 4 channels of interleaved data are available.
  // Rather than de-interleaving all 4 channels into separate buffers, we can 
  // process each channel is a scratch buffer.
  for (uint8_t ch = 0; ch < N_CH_PER_ADC; ch++) {
  
    process_adc_to_float(adc_raw, fft_input_buf, ch, FRAME_SIZE);
    float dc_offset = calculate_dc_offset(fft_input_buf, FRAME_SIZE);
    remove_dc_bias(fft_input_buf, FRAME_SIZE, dc_offset);
    apply_fft(fft_input_buf, fft_output, FRAME_SIZE);

    // normalize_magnitude(fft_output, 512);
  }
}

uint16_t calculate_checksum(uint8_t *data, uint32_t length)
{
  uint16_t sum = 0;
  for (uint32_t i = 0; i < length; i++) {
      sum += data[i];
  }
  return sum;
}

void package_adc_for_spi(uint8_t adc_id,
                         const float *fft_output,
                         uint8_t *packet_buffer,
                         uint16_t mic_count,
                         uint16_t fft_size,
                         uint32_t sample_rate,
                         uint16_t bin_count) {

  SPI_FrameHeader_t *hdr = (SPI_FrameHeader_t *)(void *)packet_buffer;
  
  hdr->magic         = SPI_MAGIC;
  hdr->version       = (uint16_t)SPI_VERSION;
  hdr->header_len    = (uint16_t)sizeof(SPI_FrameHeader_t);
  hdr->frame_counter = spi_frame_counter++;

  hdr->mic_count     = mic_count;     // e.g. 16 (or 4 if per-ADC packet)
  hdr->fft_size      = fft_size;      // e.g. 1024
  hdr->sample_rate   = sample_rate;   // e.g. 48000
  hdr->bin_count     = bin_count;     // e.g. 513 for 1024-point RFFT (or 512 if you choose)
  hdr->reserved      = (uint16_t)((uint16_t)adc_id); // Option A: stash adc_id here (low 8 bits)
  // Alternatively: use reserved as flags and don't store adc_id here.

  // pkt->header = SPI_PACKET_HEADER;
  // pkt->adc_id = adc_id;
  // pkt->frame_counter = spi_frame_counter++;
  // TODO: is memcpy faster than a loop here?

  hdr->payload_len   = (uint32_t)FFT_PAYLOAD_BYTES;

  // 2) Copy payload immediately after header
  uint8_t *payload_ptr = packet_buffer + sizeof(SPI_FrameHeader_t);
  memcpy(payload_ptr, fft_output, FFT_PAYLOAD_BYTES);

  // 3) Optional checksum placed after payload (2 bytes)
  // Layout: [header][payload][checksum]
  uint8_t *checksum_ptr = payload_ptr + FFT_PAYLOAD_BYTES;
  uint32_t checksum_len = (uint32_t)(sizeof(SPI_FrameHeader_t) + FFT_PAYLOAD_BYTES);

  uint16_t checksum = calculate_checksum(packet_buffer, checksum_len);
  memcpy(checksum_ptr, &checksum, sizeof(checksum));

  // for (uint32_t i = 0; i < 512; i++) {
  //     pkt->fft_data[i] = fft_output[i];
  // }
  
  // uint32_t checksum_len = SPI_PACKET_SIZE - sizeof(uint16_t);
  // pkt->checksum = calculate_checksum(packet_buffer, checksum_len);
}

void transmit_spi_packet(uint8_t *packet_buffer, uint32_t packet_size)
{
    HAL_SPI_Transmit(&hspi4, packet_buffer, packet_size, HAL_MAX_DELAY);
}

void usb_cdc_smoke_test() {
    const char *test_str = "Hello from USB CDC!\r\n";
    usb_printf("%s", test_str);
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
  for(int i = 0; i < N_SAMPLES; i++)
  {
    adc1_buf[i] = 1234;
  }

  int flag = 10000;
  int temp_buffer[N_SAMPLES];
  if(flag >= 0)
  {

    memcpy(adc1_buf, temp_buffer, sizeof(temp_buffer));
    flag --;
  }
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */
  // TODO: TEST the UART and USB init
  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_ADC1_Init();
  MX_ADC2_Init();
  MX_ADC3_Init();
  MX_ADC4_Init();
  MX_SPI4_Init();
  MX_TIM6_Init();
  MX_USART2_UART_Init();
  MX_USB_Device_Init();
  /* USER CODE BEGIN 2 */
  
  // Calibrate all ADCs before use
  HAL_ADCEx_Calibration_Start(&hadc1, (uint32_t)ADC_SINGLE_ENDED);
  HAL_ADCEx_Calibration_Start(&hadc2, (uint32_t)ADC_SINGLE_ENDED);
  HAL_ADCEx_Calibration_Start(&hadc3,(uint32_t) ADC_SINGLE_ENDED);
  HAL_ADCEx_Calibration_Start(&hadc4, (uint32_t)ADC_SINGLE_ENDED);

  arm_rfft_fast_init_f32(&fft_instance, FRAME_SIZE);

  // Test USB CDC
  HAL_Delay(1000);
  usb_cdc_smoke_test();
  
  // Start Timer6 (triggers all ADCs synchronously)
  HAL_TIM_Base_Start(&htim6);

  // Start all 4 ADCs with DMA (2048 samples = 1024×2 ping-pong)
  HAL_ADC_Start_DMA(&hadc1, (uint32_t*)adc1_buf, (uint32_t)(2 * FRAME_SIZE));
  HAL_ADC_Start_DMA(&hadc2, (uint32_t*)adc2_buf, (uint32_t)(2 * FRAME_SIZE));
  HAL_ADC_Start_DMA(&hadc3, (uint32_t*)adc3_buf, (uint32_t)(2 * FRAME_SIZE));
  HAL_ADC_Start_DMA(&hadc4, (uint32_t*)adc4_buf, (uint32_t)(2 * FRAME_SIZE));

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

    // Clear the global mask immediately to avoid missing new events
    __disable_irq();
    uint32_t local_mask = adc_ready_mask;
    adc_ready_mask = 0;
    __enable_irq();

    // FFT Processing Pipeline
    // When any ADC half-buffer completes, process that acoustic window
    if (local_mask && !fft_in_progress) {
      fft_in_progress = 1;
      
      // Check each ADC for ready half-buffers
      for (int adc = 0; adc < 4; adc++) {
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
            half_offset = 1024;
            ready_half[adc] = 1;
          }
          
          if (adc == 0) {
            buf_ptr = adc1_buf;
            fft_out_ptr = fft_out_adc1;
          } else if (adc == 1) {
            buf_ptr = adc2_buf;
            fft_out_ptr = fft_out_adc2;
          } else if (adc == 2) {
            buf_ptr = adc3_buf;
            fft_out_ptr = fft_out_adc3;
          } else if (adc == 3) {
            buf_ptr = adc4_buf;
            fft_out_ptr = fft_out_adc4;
          }
          
          if (buf_ptr && fft_out_ptr) {
            // TODO: Instead of processing to an output buffer, we could
            // set the SPI payload buffer as the output of the ADC processing
            process_adc_pipeline(&buf_ptr[half_offset], adc, fft_out_ptr);

            // TODO: parameters N_CH_PER_ADC, FRAME_SIZE, 48000, N_BINS are
            // currently dummy values
            package_adc_for_spi(adc, fft_out_ptr, spi_tx_buffer, N_CH_PER_ADC,
                                FRAME_SIZE, 48000, N_BINS);
            transmit_spi_packet(spi_tx_buffer, SPI_PACKET_SIZE);
          }
        }
      }
      
      // Clear ready flags after processing
      fft_in_progress = 0;
    }

    /* Periodic debug print of IRQ counters over UART (every ~1000 iterations) */
   /* static uint32_t _print_counter = 0;
    if (++_print_counter >= 1000) {
      _print_counter = 0;
      char _buf[128];
      int _len = snprintf(_buf, sizeof(_buf), "IRQ total:%lu A1:%lu A2:%lu A3:%lu A4:%lu mask:0x%08lX\r\n",
                          (unsigned long)irq_events,
                          (unsigned long)irq_count_adc[0],
                          (unsigned long)irq_count_adc[1],
                          (unsigned long)irq_count_adc[2],
                          (unsigned long)irq_count_adc[3],
                          (unsigned long)local_mask);
      if (_len > 0) {
        extern UART_HandleTypeDef huart2; /* declared in usart.c / usart.h 
        HAL_UART_Transmit(&huart2, (uint8_t*)_buf, (uint16_t)_len, HAL_MAX_DELAY);
      }
    }*/

    HAL_Delay(1);
  
  }
  
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = RCC_PLLM_DIV1;
  RCC_OscInitStruct.PLL.PLLN = 12;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV4;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
