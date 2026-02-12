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
#include "usb.h"
#include "gpio.h"
#include <stdio.h>
#include "arm_math.h"  // CMSIS-DSP library

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define N_SAMPLES 2048

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
// Each ADC gets dedicated DMA circular buffer (2048 samples total)
// Half-buffer = 1024 samples (at 48kHz = 21.3ms)
static uint16_t adc1_buf[2048];
static uint16_t adc2_buf[2048];
static uint16_t adc3_buf[2048];
static uint16_t adc4_buf[2048];

// FFT Output: Frequency domain results (1024-point FFT → 512 bins per ADC)
static float fft_out_adc1[512];
static float fft_out_adc2[512];
static float fft_out_adc3[512];
static float fft_out_adc4[512];

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
uint32_t value = 0;
float voltage;
const float adc_scalar = 3.3f / 4095.0f; // 12-bit ADC, 3.3V reference

float abi_probe(float a, float b) {
    return a + b;
}

void cmsis_dsp_smoketest(void)
{
    arm_rfft_fast_instance_f32 fft;
    arm_rfft_fast_init_f32(&fft, 128);
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_ADC1_Init();
  MX_ADC2_Init();
  MX_ADC3_Init();
  MX_ADC4_Init();
  MX_SPI4_Init();
  MX_USB_PCD_Init();
  MX_TIM6_Init();
  MX_USART2_UART_Init();
  /* USER CODE BEGIN 2 */
  
  // Calibrate all ADCs before use
  HAL_ADCEx_Calibration_Start(&hadc1, ADC_SINGLE_ENDED);
  HAL_ADCEx_Calibration_Start(&hadc2, ADC_SINGLE_ENDED);
  HAL_ADCEx_Calibration_Start(&hadc3, ADC_SINGLE_ENDED);
  HAL_ADCEx_Calibration_Start(&hadc4, ADC_SINGLE_ENDED);

  // Start Timer6 (triggers all ADCs synchronously)
  HAL_TIM_Base_Start(&htim6);

  // Start all 4 ADCs with DMA (2048 samples = 1024×2 ping-pong)
  HAL_ADC_Start_DMA(&hadc1, (uint32_t*)adc1_buf, 2048);
  HAL_ADC_Start_DMA(&hadc2, (uint32_t*)adc2_buf, 2048);
  HAL_ADC_Start_DMA(&hadc3, (uint32_t*)adc3_buf, 2048);
  HAL_ADC_Start_DMA(&hadc4, (uint32_t*)adc4_buf, 2048);

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

    // FFT Processing Pipeline
    // When any ADC half-buffer completes, process that acoustic window
    if (adc_ready_mask && !fft_in_progress) {
      fft_in_progress = 1;
      
      // Check each ADC for ready half-buffers
      for (int adc = 0; adc < 4; adc++) {
        uint8_t half_flag = (adc_ready_mask >> adc) & 1;       // Half-complete
        uint8_t full_flag = (adc_ready_mask >> (adc + 4)) & 1; // Full-complete
        
        if (half_flag || full_flag) {
          // Determine which half of buffer to process
          uint16_t *buf_ptr = NULL;
          uint32_t half_offset = 0;
          
          if (half_flag) {
            // Half-complete: First half (0-1023) just finished
            half_offset = 0;
            ready_half[adc] = 0;
          } else {
            // Full-complete: Second half (1024-2047) just finished
            half_offset = 1024;
            ready_half[adc] = 1;
          }
          
          // Select buffer based on ADC
          if (adc == 0) buf_ptr = adc1_buf;
          else if (adc == 1) buf_ptr = adc2_buf;
          else if (adc == 2) buf_ptr = adc3_buf;
          else if (adc == 3) buf_ptr = adc4_buf;
          
          if (buf_ptr) {
            // Example pseudocode:
            // float *input = (float*)&buf_ptr[half_offset];
            // Generate_FFT(input, 256, output_fft);
            // Send output_fft via SPI to RasPi
            
            // TODO: Call FFT function here
            // fft_process(adc, buf_ptr, half_offset);
            
          }
        }
      }
      
      // Clear ready flags after processing
      adc_ready_mask = 0;
      fft_in_progress = 0;
    }

    /* Periodic debug print of IRQ counters over UART (every ~1000 iterations) */
    static uint32_t _print_counter = 0;
    if (++_print_counter >= 1000) {
      _print_counter = 0;
      char _buf[128];
      int _len = snprintf(_buf, sizeof(_buf), "IRQ total:%lu A1:%lu A2:%lu A3:%lu A4:%lu mask:0x%08lX\r\n",
                          (unsigned long)irq_events,
                          (unsigned long)irq_count_adc[0],
                          (unsigned long)irq_count_adc[1],
                          (unsigned long)irq_count_adc[2],
                          (unsigned long)irq_count_adc[3],
                          (unsigned long)adc_ready_mask);
      if (_len > 0) {
        extern UART_HandleTypeDef huart2; /* declared in usart.c / usart.h */
        HAL_UART_Transmit(&huart2, (uint8_t*)_buf, (uint16_t)_len, HAL_MAX_DELAY);
      }
    }

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
