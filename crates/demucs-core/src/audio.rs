//! WAV reading and writing.
//!
//! `inference.py` reads through `librosa.load` / `soundfile` and writes float32
//! WAVs. Only PCM 16/24/32 and IEEE float 32/64 WAV are handled here; other
//! containers (FLAC, MP3) are a separate concern from numerical parity and are
//! deliberately out of scope for now.

use std::io::{Read, Write};
use std::path::Path;

use crate::error::{Error, IoContext, Result};

/// Audio held channel-major: channel `c` occupies `data[c * samples ..][..samples]`.
#[derive(Debug, Clone)]
pub struct Audio {
    pub sample_rate: u32,
    pub samples: usize,
    pub data: Vec<f32>,
}

impl Audio {
    pub fn channels(&self) -> usize {
        if self.samples == 0 {
            self.data.len()
        } else {
            self.data.len() / self.samples
        }
    }

    pub fn channel(&self, index: usize) -> &[f32] {
        let start = index * self.samples;
        &self.data[start..start + self.samples]
    }

    /// Matches the mono-to-stereo expansion in `inference.py`: a single channel
    /// is duplicated when the config asks for two.
    pub fn ensure_channels(&self, wanted: usize) -> Result<Audio> {
        let have = self.channels();
        if have == wanted {
            return Ok(self.clone());
        }
        if have == 1 && wanted == 2 {
            let mono = self.channel(0).to_vec();
            let mut data = Vec::with_capacity(mono.len() * 2);
            data.extend_from_slice(&mono);
            data.extend_from_slice(&mono);
            return Ok(Audio {
                sample_rate: self.sample_rate,
                samples: self.samples,
                data,
            });
        }
        Err(Error::Audio(format!(
            "cannot convert {have} channels to {wanted}"
        )))
    }
}

/// Reads WAV or FLAC, converting samples to `f32` in the original scale.
///
/// These are the two containers the reference's `sf.write` can produce and the
/// two that `soundfile` reads without an external decoder, so a FLAC written by
/// either implementation round-trips through the other.
pub fn read_audio(path: impl AsRef<Path>) -> Result<Audio> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_path(path)?;
    if bytes.len() >= 4 && &bytes[0..4] == b"fLaC" {
        return read_flac(path);
    }
    parse_wav(&bytes).map_err(|e| Error::Audio(format!("{}: {e}", path.display())))
}

/// Reads a FLAC stream through `claxon`, scaling by `2**(bits - 1)` exactly as
/// `soundfile` does when it hands float samples to `librosa`.
pub fn read_flac(path: impl AsRef<Path>) -> Result<Audio> {
    let path = path.as_ref();
    let mut reader = claxon::FlacReader::open(path)
        .map_err(|e| Error::Audio(format!("{}: {e}", path.display())))?;
    let info = reader.streaminfo();
    let channels = info.channels as usize;
    if channels == 0 {
        return Err(Error::Audio("zero channels".into()));
    }
    let samples = info.samples.unwrap_or(0) as usize;
    let scale = 1.0f32 / (1i64 << (info.bits_per_sample - 1)) as f32;
    let mut interleaved = Vec::with_capacity(samples * channels);
    for sample in reader.samples() {
        let value = sample.map_err(|e| Error::Audio(format!("{}: {e}", path.display())))?;
        interleaved.push(value as f32 * scale);
    }
    if interleaved.len() % channels != 0 {
        return Err(Error::Audio(format!(
            "{}: {} samples do not divide into {channels} channels",
            path.display(),
            interleaved.len()
        )));
    }
    let frames = interleaved.len() / channels;
    let mut data = vec![0.0f32; interleaved.len()];
    for frame in 0..frames {
        for channel in 0..channels {
            data[channel * frames + frame] = interleaved[frame * channels + channel];
        }
    }
    Ok(Audio {
        sample_rate: info.sample_rate,
        samples: frames,
        data,
    })
}

fn parse_wav(bytes: &[u8]) -> Result<Audio> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(Error::Audio(
            "not a RIFF/WAVE file (only uncompressed WAV is supported)".into(),
        ));
    }

    let mut pos = 12usize;
    let mut format: Option<(u16, u16, u32, u16)> = None; // (format, channels, rate, bits)
    let mut data_range: Option<(usize, usize)> = None;

    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body_start = pos + 8;
        let body_end = (body_start + size).min(bytes.len());
        if id == b"fmt " {
            if size < 16 {
                return Err(Error::Audio("truncated fmt chunk".into()));
            }
            let format_tag = u16::from_le_bytes(bytes[body_start..body_start + 2].try_into().unwrap());
            let channels = u16::from_le_bytes(bytes[body_start + 2..body_start + 4].try_into().unwrap());
            let rate = u32::from_le_bytes(bytes[body_start + 4..body_start + 8].try_into().unwrap());
            let bits = u16::from_le_bytes(bytes[body_start + 14..body_start + 16].try_into().unwrap());
            format = Some((format_tag, channels, rate, bits));
        } else if id == b"data" {
            data_range = Some((body_start, body_end));
        }
        // Chunks are word-aligned.
        pos = body_end + (size & 1);
    }

    let (format_tag, channels, sample_rate, bits) =
        format.ok_or_else(|| Error::Audio("missing fmt chunk".into()))?;
    let (data_start, data_end) =
        data_range.ok_or_else(|| Error::Audio("missing data chunk".into()))?;
    let channels = channels as usize;
    if channels == 0 {
        return Err(Error::Audio("zero channels".into()));
    }
    let payload = &bytes[data_start..data_end];

    let data: Vec<f32> = match (format_tag, bits) {
        // WAVE_FORMAT_IEEE_FLOAT
        (3, 32) => payload
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        (3, 64) => payload
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        // WAVE_FORMAT_PCM
        (1, 16) => payload
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes(c.try_into().unwrap()) as f32 / 32768.0)
            .collect(),
        (1, 24) => payload
            .chunks_exact(3)
            .map(|c| {
                let raw = (c[0] as i32) | ((c[1] as i32) << 8) | ((c[2] as i32) << 16);
                let signed = (raw << 8) >> 8;
                signed as f32 / 8_388_608.0
            })
            .collect(),
        (1, 32) => payload
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32 / 2_147_483_648.0)
            .collect(),
        // WAVE_FORMAT_EXTENSIBLE carries the real tag in the sub-format GUID.
        (0xFFFE, 32) => payload
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        (tag, bits) => {
            return Err(Error::Audio(format!(
                "unsupported WAV format {tag} with {bits} bits per sample"
            )))
        }
    };

    let samples = data.len() / channels;
    if data.len() % channels != 0 {
        return Err(Error::Audio(format!(
            "data chunk of {} samples is not a whole number of {channels}-channel frames",
            data.len()
        )));
    }

    // WAV stores interleaved; the pipeline wants channel-major.
    let mut deinterleaved = vec![0.0f32; data.len()];
    for frame in 0..samples {
        for channel in 0..channels {
            deinterleaved[channel * samples + frame] = data[frame * channels + channel];
        }
    }

    Ok(Audio {
        sample_rate,
        samples,
        data: deinterleaved,
    })
}

/// `sf.write`'s `subtype=` values that the reference's `--pcm_type` can select.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subtype {
    Pcm16,
    Pcm24,
    Float,
}

impl Subtype {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "PCM_16" => Some(Subtype::Pcm16),
            "PCM_24" => Some(Subtype::Pcm24),
            "FLOAT" => Some(Subtype::Float),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Subtype::Pcm16 => "PCM_16",
            Subtype::Pcm24 => "PCM_24",
            Subtype::Float => "FLOAT",
        }
    }

    /// `sf.available_subtypes(codec)`, restricted to the three subtypes above.
    pub fn supported(codec: Codec) -> &'static [Subtype] {
        match codec {
            Codec::Wav => &[Subtype::Pcm16, Subtype::Pcm24, Subtype::Float],
            Codec::Flac => &[Subtype::Pcm16, Subtype::Pcm24],
        }
    }

    /// `sf.default_subtype(codec)`.
    pub fn default_for(codec: Codec) -> Self {
        match codec {
            Codec::Wav => Subtype::Pcm16,
            Codec::Flac => Subtype::Pcm16,
        }
    }

    pub fn is_supported(self, codec: Codec) -> bool {
        Subtype::supported(codec).contains(&self)
    }
}

/// Output container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    Wav,
    Flac,
}

impl Codec {
    pub fn extension(self) -> &'static str {
        match self {
            Codec::Wav => "wav",
            Codec::Flac => "flac",
        }
    }

    /// `validate_sndfile_subtype`: a subtype the codec cannot store falls back to
    /// the codec's default with a warning, exactly as the reference prints one.
    pub fn validated_subtype(self, wanted: Subtype) -> Subtype {
        if wanted.is_supported(self) {
            return wanted;
        }
        let default = Subtype::default_for(self);
        println!(
            "WARNING: codec {} doesn't support subtype {}, defaulting to {}",
            self.extension(),
            wanted.name(),
            default.name()
        );
        default
    }
}

/// Writes an audio file through the same codec/subtype pair `sf.write` would.
pub fn write_audio(
    path: impl AsRef<Path>,
    codec: Codec,
    subtype: Subtype,
    sample_rate: u32,
    channels: usize,
    data: &[f32],
) -> Result<()> {
    match codec {
        Codec::Wav => write_wav(path, subtype, sample_rate, channels, data),
        Codec::Flac => write_flac(path, subtype, sample_rate, channels, data),
    }
}

/// Writes a WAV in the requested subtype, matching `sf.write(..., subtype=...)`.
///
/// `data` is channel-major, the same convention [`read_audio`] returns and the
/// one the separation pipeline works in. WAV stores frames interleaved, so the
/// channel order is permuted on the way out.
pub fn write_wav(
    path: impl AsRef<Path>,
    subtype: Subtype,
    sample_rate: u32,
    channels: usize,
    data: &[f32],
) -> Result<()> {
    let path = path.as_ref();
    if channels == 0 {
        return Err(Error::Audio("cannot write zero channels".into()));
    }
    if data.len() % channels != 0 {
        return Err(Error::Audio(format!(
            "{} samples do not divide into {channels} channels",
            data.len()
        )));
    }
    let frames = data.len() / channels;
    let bytes_per_sample = match subtype {
        Subtype::Pcm16 => 2u32,
        Subtype::Pcm24 => 3,
        Subtype::Float => 4,
    };
    let format_tag: u16 = match subtype {
        Subtype::Float => 3, // IEEE float
        _ => 1,              // PCM
    };
    let block_align = (channels as u32) * bytes_per_sample;
    let data_bytes = (data.len() as u32) * bytes_per_sample;

    let mut out = Vec::with_capacity(data_bytes as usize + 64);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&format_tag.to_le_bytes());
    out.extend_from_slice(&(channels as u16).to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * block_align).to_le_bytes());
    out.extend_from_slice(&(block_align as u16).to_le_bytes());
    out.extend_from_slice(&(bytes_per_sample as u16 * 8).to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for frame in 0..frames {
        for channel in 0..channels {
            let value = data[channel * frames + frame];
            match subtype {
                Subtype::Float => out.extend_from_slice(&value.to_le_bytes()),
                Subtype::Pcm16 => {
                    out.extend_from_slice(&quantize(value, 16).to_le_bytes()[..2])
                }
                Subtype::Pcm24 => {
                    out.extend_from_slice(&quantize(value, 24).to_le_bytes()[..3])
                }
            }
        }
    }

    let mut file = std::fs::File::create(path).with_path(path)?;
    file.write_all(&out).with_path(path)?;
    Ok(())
}

/// IEEE float32 WAV, the format the pipeline has always written.
pub fn write_wav_f32(
    path: impl AsRef<Path>,
    sample_rate: u32,
    channels: usize,
    data: &[f32],
) -> Result<()> {
    write_wav(path, Subtype::Float, sample_rate, channels, data)
}

/// Writes a FLAC stream, matching `sf.write(..., format='FLAC')`.
pub fn write_flac(
    path: impl AsRef<Path>,
    subtype: Subtype,
    sample_rate: u32,
    channels: usize,
    data: &[f32],
) -> Result<()> {
    use flacenc::bitsink::ByteSink;
    use flacenc::component::BitRepr;
    use flacenc::config::Encoder;
    use flacenc::error::Verify;
    use flacenc::source::MemSource;

    let path = path.as_ref();
    if channels == 0 {
        return Err(Error::Audio("cannot write zero channels".into()));
    }
    let bits = match subtype {
        Subtype::Pcm16 => 16usize,
        Subtype::Pcm24 => 24,
        Subtype::Float => {
            return Err(Error::Audio(
                "FLAC cannot store the FLOAT subtype".to_string(),
            ))
        }
    };
    if data.len() % channels != 0 {
        return Err(Error::Audio(format!(
            "{} samples do not divide into {channels} channels",
            data.len()
        )));
    }
    let frames = data.len() / channels;
    let mut interleaved = Vec::with_capacity(data.len());
    for frame in 0..frames {
        for channel in 0..channels {
            interleaved.push(quantize(data[channel * frames + frame], bits));
        }
    }

    let config = Encoder::default()
        .into_verified()
        .map_err(|(_, e)| Error::Audio(format!("flac encoder configuration: {e:?}")))?;
    let source = MemSource::from_samples(&interleaved, channels, bits, sample_rate as usize);
    let stream = flacenc::encode_with_fixed_block_size(&config, source, 4096)
        .map_err(|e| Error::Audio(format!("flac encoding: {e}")))?;
    let mut sink = ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|e| Error::Audio(format!("flac stream: {e}")))?;
    std::fs::write(path, sink.as_slice()).with_path(path)?;
    Ok(())
}

/// `lrintf(value * 2**(bits-1))` with libsndfile's clip to the signed range.
fn quantize(value: f32, bits: usize) -> i32 {
    let scale = (1i64 << (bits - 1)) as f32;
    let minimum = -(1i64 << (bits - 1)) as f32;
    let maximum = ((1i64 << (bits - 1)) - 1) as f32;
    (value * scale).round().clamp(minimum, maximum) as i32
}

/// Copies a reader into memory, used by tests and by format probes.
pub fn read_all(mut reader: impl Read) -> std::io::Result<Vec<u8>> {
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer)?;
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_wav(format_tag: u16, channels: u16, rate: u32, bits: u16, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + payload.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&format_tag.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        let block_align = channels * bits / 8;
        out.extend_from_slice(&(rate * block_align as u32).to_le_bytes());
        out.extend_from_slice(&block_align.to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn reads_mono_float_wav_channel_major() {
        let payload: Vec<u8> = [0.5f32, -0.25, 0.125]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bytes = build_wav(3, 1, 44100, 32, &payload);
        let audio = parse_wav(&bytes).unwrap();
        assert_eq!(audio.channels(), 1);
        assert_eq!(audio.samples, 3);
        assert_eq!(audio.channel(0), &[0.5, -0.25, 0.125]);
    }

    #[test]
    fn deinterleaves_stereo() {
        // frames: (L=1.0, R=2.0), (L=3.0, R=4.0)
        let payload: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bytes = build_wav(3, 2, 44100, 32, &payload);
        let audio = parse_wav(&bytes).unwrap();
        assert_eq!(audio.channels(), 2);
        assert_eq!(audio.samples, 2);
        assert_eq!(audio.channel(0), &[1.0, 3.0]);
        assert_eq!(audio.channel(1), &[2.0, 4.0]);
    }

    #[test]
    fn reads_pcm16_scaled_like_soundfile() {
        let payload: Vec<u8> = [32767i16, -32768, 0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bytes = build_wav(1, 1, 44100, 16, &payload);
        let audio = parse_wav(&bytes).unwrap();
        assert!((audio.channel(0)[0] - 0.999_969_5).abs() < 1e-6);
        assert_eq!(audio.channel(0)[1], -1.0);
        assert_eq!(audio.channel(0)[2], 0.0);
    }

    #[test]
    fn reads_pcm24_with_sign_extension() {
        let payload: Vec<u8> = vec![0x00, 0x00, 0x80, 0xff, 0xff, 0x7f];
        let bytes = build_wav(1, 1, 44100, 24, &payload);
        let audio = parse_wav(&bytes).unwrap();
        assert_eq!(audio.channel(0)[0], -1.0);
        assert!((audio.channel(0)[1] - 0.999_999_88).abs() < 1e-6);
    }

    #[test]
    fn rejects_non_riff_data() {
        assert!(parse_wav(b"not a wav file at all").is_err());
    }

    #[test]
    fn rejects_unsupported_bit_depths() {
        let bytes = build_wav(1, 1, 44100, 8, &[1, 2, 3]);
        assert!(parse_wav(&bytes).is_err());
    }

    #[test]
    fn mono_is_duplicated_for_stereo_configs() {
        let audio = Audio {
            sample_rate: 44100,
            samples: 2,
            data: vec![0.5, -0.5],
        };
        let stereo = audio.ensure_channels(2).unwrap();
        assert_eq!(stereo.channels(), 2);
        assert_eq!(stereo.channel(0), &[0.5, -0.5]);
        assert_eq!(stereo.channel(1), &[0.5, -0.5]);
    }

    #[test]
    fn reads_a_soundfile_produced_float_wav_exactly() {
        // A real `soundfile` float32 WAV carries `fact` and `PEAK` chunks between
        // `fmt ` and `data`, so the data offset is not 44. Ground truth values
        // come from `soundfile.read` on the same file.
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/clip20.wav");
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let audio = read_audio(&path).unwrap();
        assert_eq!(audio.channels(), 1);
        assert_eq!(audio.samples, 882_000);
        assert_eq!(audio.sample_rate, 44_100);

        let at = |i: usize| audio.channel(0)[i];
        for (index, expected) in [
            (100_000usize, -0.045_654_3f32),
            (100_001, -0.020_416_26),
            (100_002, 0.002_288_82),
            (500_000, -0.079_589_84),
            (500_004, -0.196_411_13),
        ] {
            assert!(
                (at(index) - expected).abs() < 1e-7,
                "sample {index}: read {} expected {expected}",
                at(index)
            );
        }

        let bitsum: u64 = audio
            .channel(0)
            .iter()
            .map(|v| v.to_bits() as u64)
            .sum();
        assert_eq!(bitsum, 1_845_707_634_207_744, "whole-file bit checksum differs");
    }

    #[test]
    fn round_trips_through_the_writer() {
        let dir = std::env::temp_dir().join("demucs-audio-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("roundtrip.wav");
        // channel-major with two channels: L = [0.25, -0.75], R = [1.0, -1.0]
        let data = [0.25f32, -0.75, 1.0, -1.0];
        write_wav_f32(&path, 22050, 2, &data).unwrap();
        let audio = read_audio(&path).unwrap();
        assert_eq!(audio.sample_rate, 22050);
        assert_eq!(audio.channels(), 2);
        assert_eq!(audio.channel(0), &[0.25, -0.75]);
        assert_eq!(audio.channel(1), &[1.0, -1.0]);
    }

    #[test]
    fn writer_interleaves_frames_on_disk() {
        // Reading back through the same library would hide a layout mistake, so
        // check the raw bytes: frame 0 must be (L, R) and frame 1 must follow.
        let dir = std::env::temp_dir().join("demucs-audio-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("interleave.wav");
        // channel-major: L = [1, 2], R = [3, 4]
        write_wav_f32(&path, 8000, 2, &[1.0f32, 2.0, 3.0, 4.0]).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let audio = parse_wav(&bytes).unwrap();
        assert_eq!(audio.channel(0), &[1.0, 2.0]);
        assert_eq!(audio.channel(1), &[3.0, 4.0]);

        // Locate the data chunk and verify the interleaving itself.
        let mut pos = 12usize;
        let mut payload = None;
        while pos + 8 <= bytes.len() {
            let id = &bytes[pos..pos + 4];
            let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
            if id == b"data" {
                payload = Some(pos + 8);
            }
            pos += 8 + size + (size & 1);
        }
        let start = payload.expect("data chunk");
        let values: Vec<f32> = bytes[start..start + 16]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![1.0, 3.0, 2.0, 4.0], "frames must interleave L,R");
    }
}
