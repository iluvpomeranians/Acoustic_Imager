/**
 * @file spi_stream_test.c
 * @brief Unit tests for SPI stream packet builder
 * @details 
 */

/* =========================================================================
 * INCLUDES
 * ========================================================================= */
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "spi_stream_test.h"
#include "spi_stream.h"
#include "usb/usb_debug.h"
#include "protocol/spi_protocol.h"
#include "app_main.h"

/* =========================================================================
 * DEFINES & CONSTANTS
 * ========================================================================= */

/* =========================================================================
 * STATIC VARIABLES
 * ========================================================================= */

/* =========================================================================
 * FORWARD DECLARATIONS
 * ========================================================================= */

/* static void module_init(void); */
/* static void module_process(void); */

/* =========================================================================
 * PUBLIC FUNCTIONS
 * ========================================================================= */
/**
 * @brief Dumps a byte buffer in hex format over USB CDC for debugging.
 * @param[in] p Pointer to the byte buffer to dump
 * @param[in] n Number of bytes in the buffer
 * @return void
 */
static void dump_hex(const uint8_t *p, size_t n)
{
  for (size_t i = 0; i < n; i++) {
      usb_printf("%02X%s", p[i], ((i + 1) % 16 == 0) ? "\r\n" : " ");
  }
  if (n % 16 != 0) usb_printf("\r\n");
}

/**
 * @brief Safe float read from possibly unaligned payload
 * @param[in] p Pointer to 4 bytes representing a float in IEEE-754 format
 * @return The float value read from the byte array
 */ 
static float read_f32_le(const uint8_t *p)
{
  float f;
  memcpy(&f, p, sizeof(f));
  return f;
}

// Recompute checksum exactly like your function
static uint16_t checksum16_sum_bytes_ref(const uint8_t *data, size_t length)
{
  uint16_t sum = 0;
  for (size_t i = 0; i < length; i++) {
      sum = (uint16_t)(sum + data[i]);
  }
  return sum;
}

void spi_stream_unit_test_build_packet(void)
{
  usb_printf("\r\n=== spi_stream_build_fft_packet UNIT TEST ===\r\n");

  // Test parameters
  const uint8_t  adc_id      = 3;
  const uint16_t mic_count   = 4;
  const uint16_t fft_size    = 1024;
  const uint32_t sample_rate = 100000;
  const uint16_t bin_count   = 8; // small for easy printing

  // Build a known complex payload: [Re0, Im0, Re1, Im1, ...]
  // Your builder uses payload_len = 2 * bin_count * sizeof(float)
  float fft_bins[2 * bin_count];
  for (uint16_t k = 0; k < bin_count; k++) {
      fft_bins[2*k + 0] = (float)k + 0.25f;   // Re
      fft_bins[2*k + 1] = -(float)k - 0.75f;  // Im
  }

  spi_stream_t stream;
  spi_stream_init(&stream);

  // Ensure dst is aligned to 4 bytes to avoid unaligned header stores
  union {
      uint32_t align;
      uint8_t  bytes[256];
  } pkt;

  memset(pkt.bytes, 0xEE, sizeof(pkt.bytes)); // poison to detect what gets written

  const size_t total = spi_stream_build_fft_packet(
      &stream,
      pkt.bytes,
      sizeof(pkt.bytes),
      adc_id,
      fft_bins,
      mic_count,
      fft_size,
      sample_rate,
      bin_count);

  usb_printf("build returned total_len=%u\r\n", (unsigned)total);

  if (total == 0) {
      usb_printf("FAIL: build returned 0\r\n");
      return;
  }

  // 1) Check header values (read via memcpy to avoid alignment assumptions)
  SPI_FrameHeader_t hdr;
  memcpy(&hdr, pkt.bytes, sizeof(hdr));

  usb_printf("Header:\r\n");
  usb_printf("  magic       = 0x%08lX (expect 0x%08lX)\r\n",
              (unsigned long)hdr.magic, (unsigned long)SPI_MAGIC);
  usb_printf("  version      = %u (expect %u)\r\n",
              (unsigned)hdr.version, (unsigned)SPI_VERSION);
  usb_printf("  header_len   = %u (expect %u)\r\n",
              (unsigned)hdr.header_len, (unsigned)sizeof(SPI_FrameHeader_t));
  usb_printf("  frame_counter= %lu (expect 0)\r\n",
              (unsigned long)hdr.frame_counter);
  usb_printf("  mic_count    = %u (expect %u)\r\n",
              (unsigned)hdr.mic_count, (unsigned)mic_count);
  usb_printf("  fft_size     = %u (expect %u)\r\n",
              (unsigned)hdr.fft_size, (unsigned)fft_size);
  usb_printf("  sample_rate  = %lu (expect %lu)\r\n",
              (unsigned long)hdr.sample_rate, (unsigned long)sample_rate);
  usb_printf("  bin_count    = %u (expect %u)\r\n",
              (unsigned)hdr.bin_count, (unsigned)bin_count);
  usb_printf("  reserved(adc)= %u (expect %u)\r\n",
              (unsigned)hdr.reserved, (unsigned)adc_id);
  usb_printf("  payload_len  = %lu (expect %lu)\r\n",
              (unsigned long)hdr.payload_len,
              (unsigned long)(2u * (unsigned long)bin_count * (unsigned long)sizeof(float)));

  // 2) Verify payload floats match what we put in
  const size_t header_len  = sizeof(SPI_FrameHeader_t);
  const size_t payload_len = 2u * (size_t)bin_count * sizeof(float);
  const uint8_t *payload   = pkt.bytes + header_len;

usb_printf("Payload check (first %u complex bins):\r\n", (unsigned)bin_count);

int payload_ok = 1;
for (uint16_t k = 0; k < bin_count; k++) {

  // Read back from packet payload (byte-safe)
  float re = read_f32_le(payload + (size_t)(2u*k + 0u) * sizeof(float));
  float im = read_f32_le(payload + (size_t)(2u*k + 1u) * sizeof(float));

  // Expected from source array
  float re_exp = fft_bins[2u*k + 0u];
  float im_exp = fft_bins[2u*k + 1u];

  // Convert floats to raw IEEE-754 bit patterns for printing
  uint32_t re_u, im_u, re_exp_u, im_exp_u;
  memcpy(&re_u,     &re,     sizeof(re_u));
  memcpy(&im_u,     &im,     sizeof(im_u));
  memcpy(&re_exp_u, &re_exp, sizeof(re_exp_u));
  memcpy(&im_exp_u, &im_exp, sizeof(im_exp_u));

  // Print as hex
  usb_printf("  k=%u: re=0x%08lX (exp 0x%08lX), im=0x%08lX (exp 0x%08lX)%s\r\n",
              (unsigned)k,
              (unsigned long)re_u,     (unsigned long)re_exp_u,
              (unsigned long)im_u,     (unsigned long)im_exp_u,
              (re_u == re_exp_u && im_u == im_exp_u) ? "" : "  <-- MISMATCH");

  // Bitwise compare (exact)
  if (re_u != re_exp_u || im_u != im_exp_u) {
      payload_ok = 0;
  }
}

usb_printf("Payload: %s\r\n", payload_ok ? "PASS" : "FAIL");

  // 3) Verify checksum stored matches recomputed
  const uint8_t *cs_ptr = payload + payload_len;
  uint16_t cs_stored;
  memcpy(&cs_stored, cs_ptr, sizeof(cs_stored));

  const uint16_t cs_calc = checksum16_sum_bytes_ref(pkt.bytes, header_len + payload_len);

  usb_printf("Checksum:\r\n");
  usb_printf("  stored = 0x%04X\r\n", cs_stored);
  usb_printf("  calc   = 0x%04X\r\n", cs_calc);
  usb_printf("Checksum: %s\r\n", (cs_stored == cs_calc) ? "PASS" : "FAIL");

  // 4) Sanity: total length expected
  const size_t expected_total = header_len + payload_len + sizeof(uint16_t);
  usb_printf("Total length: %u (expected %u) => %s\r\n",
              (unsigned)total, (unsigned)expected_total,
              (total == expected_total) ? "PASS" : "FAIL");

  // 5) Hex dump of header + a bit of payload
  usb_printf("Hex dump (first 64 bytes):\r\n");
  dump_hex(pkt.bytes, (total < 64) ? total : 64);

  usb_printf("=== END UNIT TEST ===\r\n");
}

void spi_stream_unit_test_nulls(void)
{
  spi_stream_t s;
  spi_stream_init(&s);

  uint8_t dst[64];
  float bins[16];

  usb_printf("\r\n=== NULL TESTS ===\r\n");

  usb_printf("null stream => %u\r\n",
      (unsigned)spi_stream_build_fft_packet(NULL, dst, sizeof(dst), 0, bins, 1, 1024, 100000, 8));

  usb_printf("null dst => %u\r\n",
      (unsigned)spi_stream_build_fft_packet(&s, NULL, sizeof(dst), 0, bins, 1, 1024, 100000, 8));

  usb_printf("null bins => %u\r\n",
      (unsigned)spi_stream_build_fft_packet(&s, dst, sizeof(dst), 0, NULL, 1, 1024, 100000, 8));
}

void spi_stream_unit_test_small_cap(void)
{
  spi_stream_t s;
  spi_stream_init(&s);

  uint8_t dst[32];     // intentionally too small
  float bins[16];      // 2*8 floats

  usb_printf("\r\n=== CAPACITY TEST ===\r\n");
  size_t n = spi_stream_build_fft_packet(&s, dst, sizeof(dst), 0, bins, 1, 1024, 100000, 8);
  usb_printf("dst_cap=%u => returned %u (expect 0)\r\n",
              (unsigned)sizeof(dst), (unsigned)n);
}

void spi_stream_unit_test_frame_counter(void)
{
  spi_stream_t s;
  spi_stream_init(&s);

  union { uint32_t a; uint8_t b[256]; } pkt;
  float bins[16];

  usb_printf("\r\n=== FRAME COUNTER TEST ===\r\n");
  for (int i = 0; i < 3; i++) {
      size_t n = spi_stream_build_fft_packet(&s, pkt.b, sizeof(pkt.b), 0, bins, 1, 1024, 100000, 8);
      SPI_FrameHeader_t hdr;
      memcpy(&hdr, pkt.b, sizeof(hdr));
      usb_printf("call %d: len=%u, frame_counter=%lu\r\n",
                  i, (unsigned)n, (unsigned long)hdr.frame_counter);
  }
}
/* ============================================================================
 * STATIC FUNCTIONS
 * ============================================================================ */

/**
 * @brief Static helper function description
 * @return void
 */


