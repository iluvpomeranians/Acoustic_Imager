#include "main.h"
#include "spi.h" 
#include "acquisition/data_acquisition.hpp"
#include "preprocessing/audio_preprocessing.hpp"
#include "fft/fft_processor.hpp"
#include <cstring>

// External CubeMX functions
extern "C" {
    void SystemClock_Config(void);
    void MX_GPIO_Init(void);
    void MX_DMA_Init(void);
    void MX_SPI3_Init(void);
    void Error_Handler(void);
}

// Processing pipeline state
typedef enum {
    PIPELINE_IDLE,
    PIPELINE_ACQUIRING,
    PIPELINE_PREPROCESSING, 
    PIPELINE_FFT_COMPUTING,
    PIPELINE_READY_TO_SEND,
    PIPELINE_ERROR
} PipelineState_t;

static PipelineState_t pipelineState = PIPELINE_IDLE;
static unsigned int processedFrames = 0;

int main(void)
{
    // HAL and hardware initialization
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_DMA_Init();
    MX_SPI3_Init();
    
    // Initialize processing modules
    if (Audio_InitAcquisition() != ACQUISITION_OK) {
        Error_Handler();
    }
    
    if (Preprocessing_Init() != PREPROCESS_OK) {
        Error_Handler();
    }
    
    if (FFT_Init() != FFT_OK) {
        Error_Handler();
    }
    
    // Start audio acquisition
    if (Audio_StartAcquisition() != ACQUISITION_OK) {
        Error_Handler();
    }
    
    pipelineState = PIPELINE_ACQUIRING;
    
    // Main processing loop
    while (1) {
        switch (pipelineState) {
            case PIPELINE_ACQUIRING:
                if (Audio_IsDataReady()) {
                    pipelineState = PIPELINE_PREPROCESSING;
                }
                break;
                
            case PIPELINE_PREPROCESSING:
                if (Audio_IsDataReady()) {
                    int16_t* rawFrame = Audio_GetFramePointer();
                    
                    if (rawFrame != NULL) {
                        // Apply preprocessing (filtering, windowing, normalization)
                        PreprocessStatus_t status = Preprocessing_ProcessFrame(rawFrame, nullptr);
                        
                        Audio_ReleaseFrame();
                        
                        if (status == PREPROCESS_OK) {
                            pipelineState = PIPELINE_FFT_COMPUTING;
                        } else {
                            pipelineState = PIPELINE_ERROR;
                        }
                    }
                }
                break;
                
            case PIPELINE_FFT_COMPUTING:
                if (Preprocessing_IsFrameReady()) {
                    float* processedFrame = Preprocessing_GetProcessedFrame();
                    
                    if (processedFrame != NULL) {
                        // Compute FFT for all channels
                        static MultiChannelFFTResult_t fftResults;
                        FFTStatus_t fftStatus = FFT_ComputeAllChannels(processedFrame, &fftResults);
                        
                        Preprocessing_ReleaseFrame();
                        
                        if (fftStatus == FFT_OK) {
                            processedFrames++;
                            pipelineState = PIPELINE_READY_TO_SEND;
                        } else {
                            pipelineState = PIPELINE_ERROR;
                        }
                    }
                }
                break;
                
            case PIPELINE_READY_TO_SEND:
                if (FFT_IsResultReady()) {
                    MultiChannelFFTResult_t* results = FFT_GetResult();
                    
                    // TODO: Send FFT results to Raspberry Pi via SPI
                    // For now, just log and continue
                    if (results != NULL && results->isValid) {
                        // Results are ready for transmission
                        // Add SPI communication code here
                    }
                    
                    FFT_ReleaseResult();
                    pipelineState = PIPELINE_ACQUIRING;
                }
                break;
                
            case PIPELINE_ERROR:
                // Handle pipeline errors
                Audio_ErrorHandler();
                
                // Try to restart the pipeline
                HAL_Delay(100);
                if (Audio_StartAcquisition() == ACQUISITION_OK) {
                    pipelineState = PIPELINE_ACQUIRING;
                }
                break;
                
            default:
                pipelineState = PIPELINE_ACQUIRING;
                break;
        }
        
        // Handle any acquisition errors
        if (Audio_GetStatus() != ACQUISITION_OK) {
            pipelineState = PIPELINE_ERROR;
        }
        
        // Optional: Add a small delay to prevent busy-waiting
        // HAL_Delay(1);
    }
}