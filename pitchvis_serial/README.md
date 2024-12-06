# PitchVis Wood Box Serial Port Controller

This tool will do a PitchVis analysis and output control strings on a serial port. The MCU on the PitchVis woodbox can then consume these and light up the corresponding LEDs.

The output consists of strings of the format:
`0xFF <num_triples (16 bit)> <r1> <g1> <b1> <r2> <g2> <b2> ...`
