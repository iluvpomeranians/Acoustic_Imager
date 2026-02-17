#ifndef __SPI_PROTOCOL_H__
#define __SPI_PROTOCOL_H__

#ifdef __cplusplus
extern "C" {
#endif

/* Includes -----------------------------------------------------------------*/
#include <stdint.h>

/* Defines ------------------------------------------------------------------*/
#define SPI_MAGIC 0xAABBCCDDu
#define SPI_VERSION 1u

#define SPI_BUFFER_SIZE 4
#define SPI_PACKET_HEADER 0xAA
#define SPI_PACKET_SIZE (1 + 1 + 4 + 512*4 + 2)

#define SPI_FRAMEHEADER_SIZE_BYTES  (28u)

/* typedefs -----------------------------------------------------------------*/
typedef struct __attribute__((packed)) {
    uint32_t magic;        // Frame magic (e.g., 0xAABBCCDD)
    uint16_t version;      // Protocol version
    uint16_t header_len;   // sizeof(SPI_FrameHeader_t)
    uint32_t frame_counter; // Sequence number
    uint16_t mic_count;    // Number of microphones
    uint16_t fft_size;     // FFT size (e.g., 1024)
    uint32_t sample_rate;  // Sampling rate (Hz)
    uint16_t bin_count;    // Number of FFT bins in payload
    uint16_t reserved;     // Flags / future use
    uint32_t payload_len;  // Bytes following header (FFT payload size)
} SPI_FrameHeader_t;

/* Function prototypes ------------------------------------------------------*/

#ifdef __cplusplus
static_assert(sizeof(SPI_FrameHeader_t) == SPI_FRAMEHEADER_SIZE_BYTES,
              "SPI_FrameHeader_t size mismatch");
#else
_Static_assert(sizeof(SPI_FrameHeader_t) == SPI_FRAMEHEADER_SIZE_BYTES,
               "SPI_FrameHeader_t size mismatch");
#endif
#ifdef __cplusplus
}
#endif

#endif /* __SPI_PROTOCOL_H__ */