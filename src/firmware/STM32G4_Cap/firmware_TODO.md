
# Firmware TODO

- [ ] Check one ADC with 3 diff resistor combinations (increasing resistance) and check -> CHECKED, MUST FIX GAIN, EITHER 0 OR 3.3V
- [ ] Test counter in the main loop for the interrupts
- [x] Evaluate using CMSIS for FFTs and fixed point representation
        -> Added CMSIS DSP to build. Probably not worth to switch to fixed point.
- [ ] Do the check the CLKCNT thing to see how many CCs it takes for FFT
- [ ] Address that ADCs are configured properly for channel scanning in CubeMX
- [x] Clean up branches
- [x] Fix adc_ready
- [x] Add buffer de-interleaving
- [x] Bench debug that all the ADCs are firing at the correct rate
 

## Processing steps:
    1. DC removal (subtract constant)
        a. Make work for both gains
    2. Normalization?
    3. FFT calculations
    4. Packaging
    5. Transmit SPI

## Unit Tests
- [ ] Test the USART2 and USB
- [ ] Test that ADC buffers are filling after adc_ready_mask change