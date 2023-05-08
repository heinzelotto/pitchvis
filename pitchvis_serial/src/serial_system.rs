use bevy::prelude::*;
use pitchvis_analysis::util::*;
use serialport::SerialPort;

// serial port resource that can be shared between threads
#[derive(Resource)]
pub struct SerialPortResource(pub std::sync::Arc<std::sync::Mutex<Box<dyn SerialPort>>>);

impl SerialPortResource {
    pub fn new(path: &str, baud_rate: u32) -> Self {
        Self(std::sync::Arc::new(std::sync::Mutex::new(
            serialport::new(path, baud_rate)
                .timeout(std::time::Duration::from_secs(10)) // TODO: ???
                .open()
                .expect("Failed to open port"),
        )))
    }
}

pub fn update_serial_to_system(
    buckets_per_octave: usize,
) -> impl FnMut(Res<crate::analysis_system::AnalysisStateResource>, ResMut<SerialPortResource>) + Copy
{
    move |analysis_state: Res<crate::analysis_system::AnalysisStateResource>,
          serial_port: ResMut<SerialPortResource>| {
        update_serial(buckets_per_octave, analysis_state, serial_port);
    }
}

pub fn update_serial(
    buckets_per_octave: usize,
    analysis_state: Res<crate::analysis_system::AnalysisStateResource>,
    serial_port: ResMut<SerialPortResource>,
) {
    let analysis_state = &analysis_state.0;

    let k_max = arg_max(&analysis_state.x_cqt_peakfiltered);
    let max_size = analysis_state.x_cqt_peakfiltered[k_max];

    // special value to indicate begin of data
    let mut output: Vec<u8> = vec![0xFF];
    // 16 bit number of RGB triples to follow
    let num_triples: u16 = analysis_state.x_cqt_peakfiltered.len().try_into().unwrap();
    output.push((num_triples / 256) as u8);
    output.push((num_triples % 256) as u8);
    output.extend(
        analysis_state
            .x_cqt_peakfiltered
            .iter()
            .enumerate()
            .flat_map(|(idx, size)| {
                let (mut r, mut g, mut b) = pitchvis_analysis::color_mapping::calculate_color(
                    buckets_per_octave,
                    ((idx + (buckets_per_octave - 3 * (buckets_per_octave / 12))) as f32)
                        % buckets_per_octave as f32,
                );

                let color_coefficient = 1.0 - (1.0 - size / max_size).powf(2.0);
                r *= color_coefficient;
                g *= color_coefficient;
                b *= color_coefficient;

                [(r * 254.0) as u8, (g * 254.0) as u8, (b * 254.0) as u8]
            }),
    );
    println!("output: {:02x?}", &output);

    serial_port
        .0
        .lock()
        .unwrap()
        .write_all(output.as_slice())
        .expect("Write failed!");
    serial_port
        .0
        .lock()
        .unwrap()
        .flush()
        .expect("Flush failed!");
}
