#ifndef USB_DEBUG_H
#define USB_DEBUG_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes -----------------------------------------------------------------*/

/* Defines ------------------------------------------------------------------*/

/* typedefs -----------------------------------------------------------------*/

/* Function prototypes ------------------------------------------------------*/
int usb_printf(const char *fmt, ...);

#ifdef __cplusplus
}
#endif

#endif /* USB_DEBUG_H */