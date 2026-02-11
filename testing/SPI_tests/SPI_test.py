import spidev
import time

# Open SPI bus 0, device 0 (CE0)
spi = spidev.SpiDev()
spi.open(0, 0)

# Configure SPI
spi.max_speed_hz = 1000000   # 1 MHz (start safe)
spi.mode = 0                 # SPI mode 0 (CPOL=0, CPHA=0)
spi.bits_per_word = 8

print("SPI loopback test started...")

try:
    while True:
        tx_data = [0x12, 0x34, 0xAB, 0xCD]

        rx_data = spi.xfer2(tx_data)

        print("TX:", [hex(x) for x in tx_data])
        print("RX:", [hex(x) for x in rx_data])
        print("------------------------")

        time.sleep(1)

except KeyboardInterrupt:
    print("Exiting...")
    spi.close()
