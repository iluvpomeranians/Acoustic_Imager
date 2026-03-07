#ifndef SPI_STREAM_H
#define SPI_STREAM_H

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * INCLUDES
 * ========================================================================= */
#include <stddef.h>
#include <stdint.h>
/* =========================================================================
 * DEFINES
 * ========================================================================= */

/* =========================================================================
 * TYPE DEFINITIONS
 * ========================================================================= */
typedef struct {
  uint32_t frame_counter;
  uint32_t batch_counter;
} spi_stream_t;

/* =========================================================================
 * FUNCTION PROTOTYPES
 * ========================================================================= */
/**
 * @brief Initialize SPI stream context.
 */
void spi_stream_init(spi_stream_t *s);

uint32_t spi_stream_next_batch(spi_stream_t *s);

size_t spi_stream_build_mic_packet(
    spi_stream_t *s,
    uint8_t *dst,
    size_t dst_cap,
    uint32_t batch_id,
    uint16_t mic_index,
    const float *fft_bins,
    uint16_t mic_count,
    uint16_t fft_size,
    uint32_t sample_rate,
    uint16_t bin_count,
    uint16_t flags,
    uint16_t battery_millivolts);

/**
 * @brief Blocking transmit over SPI (hspi4).
 */
void spi_stream_tx_blocking(const uint8_t *buf, size_t len);

void package_adc_for_spi(uint8_t adc_id,
                         const float *fft_output,
                         uint8_t *packet_buffer,
                         uint16_t mic_count,
                         uint16_t fft_size,
                         uint32_t sample_rate,
                         uint16_t bin_count);

void transmit_spi_packet(uint8_t *packet_buffer, uint32_t packet_size);

#ifdef __cplusplus
}
#endif

#endif /* SPI_STREAM_H */