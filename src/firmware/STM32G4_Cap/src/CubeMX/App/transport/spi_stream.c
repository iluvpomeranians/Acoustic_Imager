/**
 * @file template.c
 * @brief Brief description of module functionality
 * @details Detailed description of what this module does and its responsibilities
 */

/* =========================================================================
 * INCLUDES
 * ========================================================================= */
#include "spi.h"
#include "spi_stream.h"
#include "../protocol/spi_protocol.h"
#include "app_main.h"

#include <string.h>

/* =========================================================================
 * DEFINES & CONSTANTS
 * ========================================================================= */

/* =========================================================================
 * STATIC VARIABLES
 * ========================================================================= */
static uint32_t spi_frame_counter = 0;

/* =========================================================================
 * FORWARD DECLARATIONS
 * ========================================================================= */

/* static void module_init(void); */
/* static void module_process(void); */
static uint16_t checksum16_sum_bytes(const uint8_t *data, size_t length);

/* =========================================================================
 * PUBLIC FUNCTIONS
 * ========================================================================= */
void spi_stream_init(spi_stream_t *s)
{
    if (!s) return;
    s->frame_counter = 0;
}

size_t spi_stream_build_fft_packet(
  spi_stream_t *s,
  uint8_t *dst,
  size_t dst_cap,
  uint8_t adc_id,
  const float *fft_bins,
  uint16_t mic_count,
  uint16_t fft_size,
  uint32_t sample_rate,
  uint16_t bin_count) {
  if (!s || !dst || !fft_bins) return 0;

  const size_t header_len  = sizeof(SPI_FrameHeader_t);
  const size_t payload_len = 2 * (size_t)bin_count * sizeof(float);
  const size_t checksum_len = sizeof(uint16_t);

  const size_t total_len = header_len + payload_len + checksum_len;
  if (dst_cap < total_len) return 0;

  SPI_FrameHeader_t hdr;

  hdr.magic         = SPI_MAGIC;
  hdr.version       = (uint16_t)SPI_VERSION;
  hdr.header_len    = (uint16_t)header_len;
  hdr.frame_counter = s->frame_counter++;

  hdr.mic_count     = mic_count;
  hdr.fft_size      = fft_size;
  hdr.sample_rate   = sample_rate;
  hdr.bin_count     = bin_count;

  // Use reserved to store adc_id (your current behavior)
  hdr.reserved      = (uint16_t)adc_id;

  hdr.payload_len   = (uint32_t)payload_len;

  memcpy(dst, &hdr, sizeof(hdr));
  
  uint8_t *payload_ptr = dst + header_len;
  memcpy(payload_ptr, fft_bins, payload_len);

  // checksum over [header + payload], stored after payload
  const uint16_t cs = checksum16_sum_bytes(dst, header_len + payload_len);
  memcpy(payload_ptr + payload_len, &cs, sizeof(cs));

  return total_len;
}

void spi_stream_tx_blocking(const uint8_t *buf, size_t len)
{
    if (!buf || len == 0) return;

    // HAL takes uint16_t length; enforce that here
    if (len > 0xFFFFu) return;

    (void)HAL_SPI_Transmit(&hspi4, (uint8_t *)buf, (uint16_t)len, HAL_MAX_DELAY);
}

/**
 * @brief Public function description
 * @param[in] param1 Description of input parameter
 * @return Description of return value
 */

/* ============================================================================
 * STATIC FUNCTIONS
 * ============================================================================ */
static uint16_t checksum16_sum_bytes(const uint8_t *data, size_t length)
{
    uint16_t sum = 0;
    for (size_t i = 0; i < length; i++) {
        sum = (uint16_t)(sum + data[i]);
    }
    return sum;
}
/**
 * @brief Static helper function description
 * @return void
 */

 /* ============================================================================
 * OLD SPI IMPLEMENTATION (to be deleted after refactor works and new functions
 * are integrated)
 * ============================================================================ */
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
