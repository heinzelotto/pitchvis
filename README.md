# pitchvis
live analysis of musical pitches in an audio signal and a nice visualization

Watch a demo video here:

[![Pitchvis Demo Video](https://img.youtube.com/vi/zVvoLa5aYY0/0.jpg)](https://www.youtube.com/watch?v=zVvoLa5aYY0)

# instructions

To run the viewer, run `cargo r --features bevy/dynamic_linking --bin pitchvis` from within `pitchvis_viewer/`.

For the webgl version:
```bash
npm install
# local test build
npm run serve
# build into dist/
npm run build
```

To output to a serial port (e. g. for transfer to a microcontroller that actuates a LED strip), run `cargo r --features bevy/dynamic_linking --bin pitchvis_serial -- </path/to/serial/fd> <baudrate>` from within `pitchvis_serial/`. The serial output format is `0xFF <num_triples / 256> <num_triples % 256> <r1> <g1> <b1> <r2> <g2> <b2> ...`. Led values are within [0x00; 0xFE] and 0xFF is the marker byte beginning each sequence.

## prerequisites

This uses `cpal` for audio input, which should give results on most platforms. You might need to install some system dependencies like `libasound2-dev` and/or `libudev-dev`.


