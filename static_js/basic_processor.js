const FRAME_SIZE = 512;

/**
 * A simple AudioWorkletProcessor node.
 *
 * @class BasicProcessor
 * @extends AudioWorkletProcessor
 */
class BasicProcessor extends AudioWorkletProcessor {
    /**
     * Constructor to initialize, input and output FreeQueue instances
     * and atomicState to synchronise Worker with AudioWorklet
     * @param {Object} options AudioWorkletProcessor options
     *    to initialize inputQueue, outputQueue and atomicState
     */
    constructor(options) {
      super();

      console.log(options);
  
      //this.cb = WASM_AUDIO_CB; /**options.processorOptions.cb;*/
      //this.cb();
    }
  
    process(inputs, outputs) {
      const input = inputs[0];
      const output = outputs[0];

	//console.log(inputs);

      this.port.postMessage(input[0]);
      
      return true;
    }
  }

  registerProcessor("basic_processor", BasicProcessor);
