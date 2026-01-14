/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body (Renode-Compatible)
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025.
  * All rights reserved.
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
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdint.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
// ADC buffer for single-channel microphone test
#define ADC_BUFFER_SIZE 1024
uint16_t adc1_buffer[ADC_BUFFER_SIZE];  // 1024 samples = ~20ms @ 50kHz

volatile uint8_t adc_half_complete = 0;
volatile uint8_t adc_full_complete = 0;

// Live stats for debugger viewing
volatile uint16_t live_avg = 0;
volatile uint16_t live_min = 0;
volatile uint16_t live_max = 0;
volatile uint16_t live_pp = 0;

// Live voltage values (0.0 to 3.3V)
volatile float live_avg_volts = 0.0f;
volatile float live_min_volts = 0.0f;
volatile float live_max_volts = 0.0f;
volatile float live_pp_volts = 0.0f;
// Diagnostics: DMA half/full interrupt counters
volatile uint32_t dma_ht_count = 0;
volatile uint32_t dma_tc_count = 0;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

// DMA half-complete callback - first half of buffer is ready
void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef* hadc) {
  dma_ht_count++;
  adc_half_complete = 1;
}

// DMA complete callback - second half of buffer is ready
void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef* hadc) {
  dma_tc_count++;
  adc_full_complete = 1;
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
  // Enable debug during sleep/stop modes to prevent ST-LINK disconnect
  __HAL_DBGMCU_FREEZE_IWDG();
  __HAL_DBGMCU_FREEZE_WWDG();
  DBGMCU->CR |= DBGMCU_CR_DBG_SLEEP | DBGMCU_CR_DBG_STOP | DBGMCU_CR_DBG_STANDBY;
  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_ADC1_Init();
  MX_TIM2_Init();
  MX_SPI1_Init();
  /* USER CODE BEGIN 2 */
  
  // Calibrate ADC1 for accurate conversions
  if (HAL_ADCEx_Calibration_Start(&hadc1, ADC_SINGLE_ENDED) != HAL_OK) {
    Error_Handler();  // Calibration failed
  }

  // Start ADC conversion with DMA using full circular buffer
  if (HAL_ADC_Start_DMA(&hadc1, (uint32_t*)adc1_buffer, ADC_BUFFER_SIZE) != HAL_OK) {
    Error_Handler();  // DMA start failed
  }

  // Start Timer2 to trigger ADC conversions at ~50kHz
  if (HAL_TIM_Base_Start(&htim2) != HAL_OK) {
    Error_Handler();  // Timer start failed
  }

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    
    // Process first half of buffer (samples 0-511)
    if (adc_half_complete) {
        adc_half_complete = 0;
        
        // Calculate stats for first half
        uint32_t sum = 0;
        uint16_t min = 0xFFFF, max = 0;
        for (int i = 0; i < ADC_BUFFER_SIZE/2; i++) {
            sum += adc1_buffer[i];
            if (adc1_buffer[i] < min) min = adc1_buffer[i];

        }
        uint16_t avg = sum / (ADC_BUFFER_SIZE/2);
        
        // Update live variables for debugger
        live_avg = avg;
        live_min = min;
        live_max = max;
        live_pp = max - min;
        
        // Convert to voltage (3.3V / 4096 counts)
        live_avg_volts = (avg / 4095.0f) * 3.3f;
        live_min_volts = (min / 4095.0f) * 3.3f;
        live_max_volts = (max / 4095.0f) * 3.3f;
        live_pp_volts = ((max - min) / 4095.0f) * 3.3f;
    }
    
    // Process second half of buffer (samples 512-1023)
    if (adc_full_complete) {
        adc_full_complete = 0;
        
        // Calculate stats for second half
        uint32_t sum = 0;
        uint16_t min = 0xFFFF, max = 0;
        for (int i = ADC_BUFFER_SIZE/2; i < ADC_BUFFER_SIZE; i++) {
            sum += adc1_buffer[i];
            if (adc1_buffer[i] < min) min = adc1_buffer[i];
            if (adc1_buffer[i] > max) max = adc1_buffer[i];
        }
        uint16_t avg = sum / (ADC_BUFFER_SIZE/2);
        
        // Update live variables for full half
        live_avg = avg;
        live_min = min;
        live_max = max;
        live_pp = max - min;
        
        // Convert to voltage (3.3V / 4096 counts)
        live_avg_volts = (avg / 4095.0f) * 3.3f;
        live_min_volts = (min / 4095.0f) * 3.3f;
        live_max_volts = (max / 4095.0f) * 3.3f;
        live_pp_volts = ((max - min) / 4095.0f) * 3.3f;
    }
    
    // Small delay to allow debugger to read memory (DMA continues independently)
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
  RCC_OscInitStruct.PLL.PLLN = 8;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV4;
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

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK)
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
