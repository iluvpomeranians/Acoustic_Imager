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
#include "gpio.h"
#include "stdio.h"

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* Renode uses STM32F4-style UART registers */
typedef struct
{
    volatile uint32_t SR;     // 0x00
    volatile uint32_t DR;     // 0x04
    volatile uint32_t BRR;    // 0x08
    volatile uint32_t CR1;    // 0x0C
    volatile uint32_t CR2;    // 0x10
    volatile uint32_t CR3;    // 0x14
    volatile uint32_t GTPR;   // 0x18
} RENODE_UART_TypeDef;

#define RENODE_USART2   ((RENODE_UART_TypeDef *)0x40004400)

/* USER CODE END PTD */

/* Private function prototypes -----------------------------------------------*/
void UART2_Init_Renode(void);

/* USER CODE BEGIN PV */
/* USER CODE END PV */

/* Minimal _write() redirect for printf */
int _write(int file, char *ptr, int len)
{
    for(int i = 0; i < len; i++)
    {
        while(!(RENODE_USART2->SR & (1 << 7)));   // TXE
        RENODE_USART2->DR = ptr[i];
    }
    return len;
}

/* Renode-Friendly USART2 Init */
void UART2_Init_Renode(void)
{
    /* Enable USART2 clock (F4-style RCC) */
    *(volatile uint32_t *)(0x40023840) |= (1 << 17);  // RCC_APB1ENR_USART2EN

    /* Enable GPIOA clock */
    *(volatile uint32_t *)(0x40023830) |= (1 << 0);   // RCC_AHB1ENR_GPIOAEN

    /* PA2 = AF7 */
    GPIOA->MODER &= ~(3 << (2 * 2));
    GPIOA->MODER |=  (2 << (2 * 2));
    GPIOA->AFR[0] &= ~(0xF << (2 * 4));
    GPIOA->AFR[0] |=  (7   << (2 * 4));

    /* Baud rate for 16MHz HSI */
    RENODE_USART2->BRR = 0x8B;   // 115200 baud

    /* Enable UART */
    RENODE_USART2->CR1 |= (1 << 3);  // TE
    RENODE_USART2->CR1 |= (1 << 13); // UE
}

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
    /* USER CODE BEGIN 1 */
    SysTick->CTRL = 0;   // Disable SysTick for Renode
    /* USER CODE END 1 */

    /* Initialize Renode-safe HAL subsystems */
    MX_GPIO_Init();
    MX_DMA_Init();
    MX_SPI4_Init();

    UART2_Init_Renode();

    /* ================================================ */
    /*                SYSTEM BOOT BANNER                */
    /* ================================================ */
    printf("\n=====================================================\n");
    printf("                  STM32G4 SYSTEM BOOT                \n");
    printf("=====================================================\n");
    printf("Build Date : %s %s\n", __DATE__, __TIME__);
    printf("Compiler   : %s\n", __VERSION__);
    printf("Target MCU : STM32G473xx  (Cortex-M4F)\n");
    printf("UART Driver: Renode F4-Compatible\n");
    printf("-----------------------------------------------------\n");

    printf("Enabled HAL Modules:\n");
#ifdef HAL_GPIO_MODULE_ENABLED
    printf(" - GPIO\n");
#endif
#ifdef HAL_DMA_MODULE_ENABLED
    printf(" - DMA\n");
#endif
#ifdef HAL_SPI_MODULE_ENABLED
    printf(" - SPI\n");
#endif
    printf("-----------------------------------------------------\n");

    printf("RCC Snapshot:\n");
    printf(" AHB2ENR  = 0x%08lX\n", RCC->AHB2ENR);
    printf(" APB1ENR1 = 0x%08lX\n", RCC->APB1ENR1);
    printf(" APB2ENR  = 0x%08lX\n", RCC->APB2ENR);
    printf("-----------------------------------------------------\n");

    printf("GPIOA MODER = 0x%08lX\n", GPIOA->MODER);
    printf("=====================================================\n");

    /* ================================================ */
    /*                SPI4 TEST DIAGNOSTICS             */
    /* ================================================ */

    printf("SPI4 Diagnostics:\n");
    printf(" SPI4->CR1   = 0x%08lX\n", SPI4->CR1);
    printf(" SPI4->CR2   = 0x%08lX\n", SPI4->CR2);
    printf(" SPI4->SR    = 0x%08lX\n", SPI4->SR);
    printf(" SPI4->DR    = 0x%08lX\n", SPI4->DR);


    printf(" NOTE: SPI transactions cannot be simulated in Renode.\n");
    printf("       Only register state can be inspected.\n");
    printf("=====================================================\n\n");

    /* Infinite loop */
    while (1)
    {
        printf("Heartbeat OK: STM32G4 running...\n");

        /* ~1 second Renode-safe delay */
        for (volatile uint32_t i = 0; i < 1200000; i++);
    }
}

/* USER CODE END 3 */

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1);

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK)
  {
    Error_Handler();
  }
}

void Error_Handler(void)
{
  __disable_irq();
  while (1)
  {
  }
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
}
#endif /* USE_FULL_ASSERT */
