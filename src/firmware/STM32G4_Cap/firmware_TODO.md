
# Firmware TODO

- [ ] Check one ADC with 3 diff resistor combinations (increasing resistance) and check -> CHECKED, MUST FIX GAIN, EITHER 0 OR 3.3V
- [ ] Test counter in the main loop for the interrupts
- [ ] Do the check the CLKCNT thing to see how many CCs it takes for FFT
- [ ] Address that ADCs are configured properly for channel scanning in CubeMX
- [ ] Bench debug that all the ADCs are firing at the correct rate
- [ ] Discuss labeling FFT frames (with counters?)
- [x] Evaluate using CMSIS for FFTs and fixed point representation
        -> Added CMSIS DSP to build. Probably not worth to switch to fixed point.
- [x] Add buffer de-interleaving
- [x] Fix adc_ready
- [x] Clean up branches

## Processing steps:
    1. DC removal (subtract constant)
        a. Make work for both gains
    2. Normalization?
    3. FFT calculations
    4. Packaging
    5. Transmit SPI