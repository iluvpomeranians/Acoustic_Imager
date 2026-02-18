#include "usb_debug.h"
#include "usbd_cdc_if.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

int usb_printf(const char *fmt, ...)
{
    char buffer[256]; // Adjust size as needed
    va_list args;
    va_start(args, fmt);
    int len = vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    if (len < 0) {
        // Encoding error
        return len;
    } else if (len >= (int)sizeof(buffer)) {
        // Output was truncated
        len = sizeof(buffer) - 1; // Adjust to actual buffer size
    }
    // Best-effort send; drop if USB busy
    if (CDC_Transmit_FS((uint8_t*)buffer, (uint16_t)len) == USBD_OK) {
        return len;
    }

    return 0;
}