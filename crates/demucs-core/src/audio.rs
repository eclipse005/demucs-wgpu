//! WAV reading and writing.
//!
//! `inference.py` reads through `librosa.load` / `soundfile` and writes float32
//! WAVs. Only PCM 16/24/32 and IEEE float 32/64 WAV are handled here; other
//! containers (FLAC, MP3) are a separate concern from numerical parity and are
//! deliberately out of scope for now.

use std::io::{Read, Seek, SeekFrom, Write};
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
    /// is duplicated when the config asks for two. Takes `self` by value so a
    /// matching layout moves without cloning a full-track plane.
    pub fn ensure_channels(mut self, wanted: usize) -> Result<Audio> {
        let have = self.channels();
        if have == wanted {
            return Ok(self);
        }
        if have == 1 && wanted == 2 {
            let mono = std::mem::take(&mut self.data);
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

/// Header / streaminfo without decoding samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioInfo {
    pub sample_rate: u32,
    pub channels: usize,
    pub frames: usize,
}

/// Reads container metadata (rate, channels, frame count) without allocating the PCM.
pub fn probe_audio(path: impl AsRef<Path>) -> Result<AudioInfo> {
    let path = path.as_ref();
    let mut file = std::fs::File::open(path).with_path(path)?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).with_path(path)?;
    if magic == *b"fLaC" {
        // claxon needs a path / fresh reader for streaminfo.
        let reader = claxon::FlacReader::open(path)
            .map_err(|e| Error::Audio(format!("{}: {e}", path.display())))?;
        let info = reader.streaminfo();
        let frames = info.samples.unwrap_or(0) as usize;
        return Ok(AudioInfo {
            sample_rate: info.sample_rate,
            channels: info.channels as usize,
            frames,
        });
    }
    file.seek(SeekFrom::Start(0)).with_path(path)?;
    let header = probe_wav(&mut file).map_err(|e| Error::Audio(format!("{}: {e}", path.display())))?;
    Ok(header)
}

/// Reads WAV or FLAC, converting samples to `f32` in the original scale.
///
/// Decode writes straight into a channel-major buffer with an O(window)
/// interleaved scratch — never a full-track interleaved plane plus a second
/// deinterleaved copy.
///
/// These are the two containers the reference's `sf.write` can produce and the
/// two that `soundfile` reads without an external decoder, so a FLAC written by
/// either implementation round-trips through the other.
pub fn read_audio(path: impl AsRef<Path>) -> Result<Audio> {
    let path = path.as_ref();
    let mut file = std::fs::File::open(path).with_path(path)?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).with_path(path)?;
    if magic == *b"fLaC" {
        return read_flac(path);
    }
    file.seek(SeekFrom::Start(0)).with_path(path)?;
    read_wav_reader(&mut file).map_err(|e| Error::Audio(format!("{}: {e}", path.display())))
}

/// Reads a FLAC stream through `claxon`, scaling by `2**(bits - 1)` exactly as
/// `soundfile` does when it hands float samples to `librosa`. Samples land
/// directly in channel-major order (no full interleaved Vec).
pub fn read_flac(path: impl AsRef<Path>) -> Result<Audio> {
    let path = path.as_ref();
    let mut reader = claxon::FlacReader::open(path)
        .map_err(|e| Error::Audio(format!("{}: {e}", path.display())))?;
    let info = reader.streaminfo();
    let channels = info.channels as usize;
    if channels == 0 {
        return Err(Error::Audio("zero channels".into()));
    }
    let known = info.samples.map(|s| s as usize);
    let scale = 1.0f32 / (1i64 << (info.bits_per_sample - 1)) as f32;
    // Prefer a sized channel-major buffer; if streaminfo omitted the count,
    // gather then place (still one plane, not interleaved+deinterleaved).
    if let Some(frames) = known {
        let mut data = vec![0.0f32; frames * channels];
        let mut index = 0usize;
        for sample in reader.samples() {
            let value = sample.map_err(|e| Error::Audio(format!("{}: {e}", path.display())))?;
            if index >= frames * channels {
                return Err(Error::Audio(format!(
                    "{}: more samples than streaminfo frames",
                    path.display()
                )));
            }
            let frame = index / channels;
            let channel = index % channels;
            data[channel * frames + frame] = value as f32 * scale;
            index += 1;
        }
        if index != frames * channels {
            return Err(Error::Audio(format!(
                "{}: got {index} samples, expected {}",
                path.display(),
                frames * channels
            )));
        }
        return Ok(Audio {
            sample_rate: info.sample_rate,
            samples: frames,
            data,
        });
    }
    let mut interleaved = Vec::new();
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

/// Bytes of PCM per interleaved frame for a (tag, bits) pair.
pub(crate) fn wav_bytes_per_frame(channels: usize, bits: u16) -> Result<usize> {
    let bps = (bits as usize) / 8;
    if bps == 0 || bits as usize % 8 != 0 {
        return Err(Error::Audio(format!("unsupported bits per sample {bits}")));
    }
    Ok(channels * bps)
}

fn probe_wav(reader: &mut (impl Read + Seek)) -> Result<AudioInfo> {
    let (format_tag, channels, sample_rate, bits, data_len, _data_pos) = scan_wav_header(reader)?;
    let _ = format_tag;
    let bpf = wav_bytes_per_frame(channels, bits)?;
    if data_len % bpf != 0 {
        return Err(Error::Audio(format!(
            "data chunk of {data_len} bytes is not a whole number of {channels}-channel frames"
        )));
    }
    Ok(AudioInfo {
        sample_rate,
        channels,
        frames: data_len / bpf,
    })
}

/// Walk RIFF chunks; on return the reader is positioned at the start of PCM.
pub(crate) fn scan_wav_header(
    reader: &mut (impl Read + Seek),
) -> Result<(u16, usize, u32, u16, usize, u64)> {
    let mut header = [0u8; 12];
    reader.read_exact(&mut header)?;
    if &header[0..4] != b"RIFF" || &header[8..12] != b"WAVE" {
        return Err(Error::Audio(
            "not a RIFF/WAVE file (only uncompressed WAV is supported)".into(),
        ));
    }
    let mut format: Option<(u16, u16, u32, u16)> = None;
    let mut data_info: Option<(u64, usize)> = None;
    loop {
        let mut chunk_h = [0u8; 8];
        match reader.read_exact(&mut chunk_h) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(Error::PlainIo(e)),
        }
        let id = &chunk_h[0..4];
        let size = u32::from_le_bytes(chunk_h[4..8].try_into().unwrap()) as usize;
        let body_pos = reader.stream_position()?;
        if id == b"fmt " {
            if size < 16 {
                return Err(Error::Audio("truncated fmt chunk".into()));
            }
            let mut fmt = vec![0u8; size];
            reader.read_exact(&mut fmt)?;
            let format_tag = u16::from_le_bytes(fmt[0..2].try_into().unwrap());
            let channels = u16::from_le_bytes(fmt[2..4].try_into().unwrap());
            let rate = u32::from_le_bytes(fmt[4..8].try_into().unwrap());
            let bits = u16::from_le_bytes(fmt[14..16].try_into().unwrap());
            format = Some((format_tag, channels, rate, bits));
        } else if id == b"data" {
            data_info = Some((body_pos, size));
            // Do not walk the PCM payload — seek back to it after the header scan.
            if format.is_some() {
                break;
            }
            reader.seek(SeekFrom::Current(size as i64))?;
            if size & 1 != 0 {
                reader.seek(SeekFrom::Current(1))?;
            }
            continue;
        } else {
            reader.seek(SeekFrom::Current(size as i64))?;
        }
        if size & 1 != 0 {
            reader.seek(SeekFrom::Current(1))?;
        }
        if format.is_some() && data_info.is_some() {
            break;
        }
    }
    let (format_tag, channels_u16, sample_rate, bits) =
        format.ok_or_else(|| Error::Audio("missing fmt chunk".into()))?;
    let (data_pos, data_len) =
        data_info.ok_or_else(|| Error::Audio("missing data chunk".into()))?;
    let channels = channels_u16 as usize;
    if channels == 0 {
        return Err(Error::Audio("zero channels".into()));
    }
    reader.seek(SeekFrom::Start(data_pos))?;
    Ok((format_tag, channels, sample_rate, bits, data_len, data_pos))
}

/// Decode one interleaved PCM frame into `out[ch]`.
pub(crate) fn decode_frame(format_tag: u16, bits: u16, frame: &[u8], channels: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), channels);
    match (format_tag, bits) {
        (3, 32) | (0xFFFE, 32) => {
            for (ch, slot) in out.iter_mut().enumerate() {
                let o = ch * 4;
                *slot = f32::from_le_bytes(frame[o..o + 4].try_into().unwrap());
            }
        }
        (3, 64) => {
            for (ch, slot) in out.iter_mut().enumerate() {
                let o = ch * 8;
                *slot = f64::from_le_bytes(frame[o..o + 8].try_into().unwrap()) as f32;
            }
        }
        (1, 16) => {
            for (ch, slot) in out.iter_mut().enumerate() {
                let o = ch * 2;
                *slot = i16::from_le_bytes(frame[o..o + 2].try_into().unwrap()) as f32 / 32768.0;
            }
        }
        (1, 24) => {
            for (ch, slot) in out.iter_mut().enumerate() {
                let o = ch * 3;
                let raw = (frame[o] as i32) | ((frame[o + 1] as i32) << 8) | ((frame[o + 2] as i32) << 16);
                let signed = (raw << 8) >> 8;
                *slot = signed as f32 / 8_388_608.0;
            }
        }
        (1, 32) => {
            for (ch, slot) in out.iter_mut().enumerate() {
                let o = ch * 4;
                *slot = i32::from_le_bytes(frame[o..o + 4].try_into().unwrap()) as f32 / 2_147_483_648.0;
            }
        }
        _ => unreachable!("format validated earlier"),
    }
}

fn read_wav_reader(reader: &mut (impl Read + Seek)) -> Result<Audio> {
    let (format_tag, channels, sample_rate, bits, data_len, _data_pos) = scan_wav_header(reader)?;
    match (format_tag, bits) {
        (3, 32) | (3, 64) | (1, 16) | (1, 24) | (1, 32) | (0xFFFE, 32) => {}
        (tag, bits) => {
            return Err(Error::Audio(format!(
                "unsupported WAV format {tag} with {bits} bits per sample"
            )))
        }
    }
    let bpf = wav_bytes_per_frame(channels, bits)?;
    if data_len % bpf != 0 {
        return Err(Error::Audio(format!(
            "data chunk of {data_len} bytes is not a whole number of {channels}-channel frames"
        )));
    }
    let frames = data_len / bpf;
    let mut data = vec![0.0f32; frames * channels];
    // Windowed interleaved scratch — O(segment), not O(track).
    const WINDOW_FRAMES: usize = 4096;
    let mut scratch = vec![0u8; WINDOW_FRAMES * bpf];
    let mut frame_vals = vec![0.0f32; channels];
    let mut frame = 0usize;
    while frame < frames {
        let n = (frames - frame).min(WINDOW_FRAMES);
        let bytes = n * bpf;
        reader.read_exact(&mut scratch[..bytes])?;
        for i in 0..n {
            let raw = &scratch[i * bpf..(i + 1) * bpf];
            decode_frame(format_tag, bits, raw, channels, &mut frame_vals);
            let t = frame + i;
            for ch in 0..channels {
                data[ch * frames + t] = frame_vals[ch];
            }
        }
        frame += n;
    }
    Ok(Audio {
        sample_rate,
        samples: frames,
        data,
    })
}

/// In-memory WAV parse used by unit tests (same decode path as the file reader).
fn parse_wav(bytes: &[u8]) -> Result<Audio> {
    let mut cursor = std::io::Cursor::new(bytes);
    read_wav_reader(&mut cursor)
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
    let data_bytes = (frames as u64)
        .checked_mul(block_align as u64)
        .ok_or_else(|| Error::Audio("WAV data chunk too large".into()))?;
    if data_bytes > u32::MAX as u64 || 36 + data_bytes > u32::MAX as u64 {
        return Err(Error::Audio(
            "WAV larger than 4 GiB is not supported by classic RIFF".into(),
        ));
    }
    let data_bytes_u32 = data_bytes as u32;

    // Stream frames through a buffered writer — never build a whole-file Vec.
    let file = std::fs::File::create(path).with_path(path)?;
    let mut out = std::io::BufWriter::with_capacity(1 << 20, file);
    out.write_all(b"RIFF").with_path(path)?;
    out.write_all(&(36 + data_bytes_u32).to_le_bytes()).with_path(path)?;
    out.write_all(b"WAVE").with_path(path)?;
    out.write_all(b"fmt ").with_path(path)?;
    out.write_all(&16u32.to_le_bytes()).with_path(path)?;
    out.write_all(&format_tag.to_le_bytes()).with_path(path)?;
    out.write_all(&(channels as u16).to_le_bytes()).with_path(path)?;
    out.write_all(&sample_rate.to_le_bytes()).with_path(path)?;
    out.write_all(&(sample_rate * block_align).to_le_bytes())
        .with_path(path)?;
    out.write_all(&(block_align as u16).to_le_bytes()).with_path(path)?;
    out.write_all(&(bytes_per_sample as u16 * 8).to_le_bytes())
        .with_path(path)?;
    out.write_all(b"data").with_path(path)?;
    out.write_all(&data_bytes_u32.to_le_bytes()).with_path(path)?;
    let mut frame_buf = vec![0u8; block_align as usize];
    for frame in 0..frames {
        let mut o = 0usize;
        for channel in 0..channels {
            let value = data[channel * frames + frame];
            match subtype {
                Subtype::Float => {
                    frame_buf[o..o + 4].copy_from_slice(&value.to_le_bytes());
                    o += 4;
                }
                Subtype::Pcm16 => {
                    frame_buf[o..o + 2].copy_from_slice(&quantize(value, 16).to_le_bytes()[..2]);
                    o += 2;
                }
                Subtype::Pcm24 => {
                    frame_buf[o..o + 3].copy_from_slice(&quantize(value, 24).to_le_bytes()[..3]);
                    o += 3;
                }
            }
        }
        out.write_all(&frame_buf).with_path(path)?;
    }
    out.flush().with_path(path)?;
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

/// Incremental IEEE float32 WAV writer: header with placeholder sizes, then
/// interleaved frames via a buffered writer, then a seek to patch RIFF/data
/// sizes. Peak RAM is O(buffer), not O(T).
pub struct FloatWavWriter {
    path: std::path::PathBuf,
    file: std::io::BufWriter<std::fs::File>,
    sample_rate: u32,
    channels: usize,
    frames: u64,
}

impl FloatWavWriter {
    /// Creates `path` and writes a float32 WAV header with zeroed sizes.
    pub fn create(path: impl AsRef<Path>, sample_rate: u32, channels: usize) -> Result<Self> {
        let path_ref = path.as_ref();
        if channels == 0 {
            return Err(Error::Audio("cannot write zero channels".into()));
        }
        let file = std::fs::File::create(path_ref).with_path(path_ref)?;
        let mut out = std::io::BufWriter::with_capacity(1 << 20, file);
        let block_align = (channels as u32) * 4;
        out.write_all(b"RIFF").with_path(path_ref)?;
        out.write_all(&0u32.to_le_bytes()).with_path(path_ref)?; // patch later
        out.write_all(b"WAVE").with_path(path_ref)?;
        out.write_all(b"fmt ").with_path(path_ref)?;
        out.write_all(&16u32.to_le_bytes()).with_path(path_ref)?;
        out.write_all(&3u16.to_le_bytes()).with_path(path_ref)?; // IEEE float
        out.write_all(&(channels as u16).to_le_bytes()).with_path(path_ref)?;
        out.write_all(&sample_rate.to_le_bytes()).with_path(path_ref)?;
        out.write_all(&(sample_rate * block_align).to_le_bytes())
            .with_path(path_ref)?;
        out.write_all(&(block_align as u16).to_le_bytes()).with_path(path_ref)?;
        out.write_all(&32u16.to_le_bytes()).with_path(path_ref)?;
        out.write_all(b"data").with_path(path_ref)?;
        out.write_all(&0u32.to_le_bytes()).with_path(path_ref)?; // patch later
        Ok(Self {
            path: path_ref.to_path_buf(),
            file: out,
            sample_rate,
            channels,
            frames: 0,
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn frames_written(&self) -> u64 {
        self.frames
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Writes one interleaved frame (`frame.len() == channels`).
    pub fn write_frame(&mut self, frame: &[f32]) -> Result<()> {
        if frame.len() != self.channels {
            return Err(Error::Audio(format!(
                "expected {} channels, got {}",
                self.channels,
                frame.len()
            )));
        }
        for &value in frame {
            self.file
                .write_all(&value.to_le_bytes())
                .with_path(&self.path)?;
        }
        self.frames += 1;
        Ok(())
    }

    /// Writes `n` interleaved frames from a flat `[frame0_ch0, frame0_ch1, ...]`.
    pub fn write_interleaved(&mut self, interleaved: &[f32]) -> Result<()> {
        if interleaved.len() % self.channels != 0 {
            return Err(Error::Audio(format!(
                "{} samples do not divide into {} channels",
                interleaved.len(),
                self.channels
            )));
        }
        for frame in interleaved.chunks_exact(self.channels) {
            self.write_frame(frame)?;
        }
        Ok(())
    }

    /// Flushes, seeks back, and patches RIFF / data chunk sizes.
    pub fn finish(mut self) -> Result<()> {
        use std::io::{Seek, SeekFrom};
        self.file.flush().with_path(&self.path)?;
        let data_bytes = self
            .frames
            .checked_mul(self.channels as u64 * 4)
            .ok_or_else(|| Error::Audio("WAV data chunk too large".into()))?;
        if data_bytes > u32::MAX as u64 || 36 + data_bytes > u32::MAX as u64 {
            return Err(Error::Audio(
                "WAV larger than 4 GiB is not supported by classic RIFF".into(),
            ));
        }
        let file = self.file.get_mut();
        file.seek(SeekFrom::Start(4)).with_path(&self.path)?;
        file.write_all(&(36 + data_bytes as u32).to_le_bytes())
            .with_path(&self.path)?;
        file.seek(SeekFrom::Start(40)).with_path(&self.path)?;
        file.write_all(&(data_bytes as u32).to_le_bytes())
            .with_path(&self.path)?;
        file.flush().with_path(&self.path)?;
        Ok(())
    }
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
        let stereo = audio.clone().ensure_channels(2).unwrap();
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

    #[test]
    fn probe_audio_matches_decoded_frames() {
        let dir = std::env::temp_dir().join("demucs-audio-probe-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("probe.wav");
        write_wav_f32(&path, 8000, 2, &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let info = probe_audio(&path).unwrap();
        let audio = read_audio(&path).unwrap();
        assert_eq!(info.sample_rate, audio.sample_rate);
        assert_eq!(info.channels, audio.channels());
        assert_eq!(info.frames, audio.samples);
    }

    #[test]
    fn float_wav_writer_matches_write_wav_f32() {
        let dir = std::env::temp_dir().join("demucs-audio-stream-test");
        std::fs::create_dir_all(&dir).unwrap();
        let streamed = dir.join("streamed.wav");
        let bulk = dir.join("bulk.wav");
        // channel-major for bulk: L=[0.25,-0.75,0.5], R=[1.0,-1.0,0.0]
        let channel_major = [0.25f32, -0.75, 0.5, 1.0, -1.0, 0.0];
        write_wav_f32(&bulk, 22050, 2, &channel_major).unwrap();
        let mut w = FloatWavWriter::create(&streamed, 22050, 2).unwrap();
        w.write_frame(&[0.25, 1.0]).unwrap();
        w.write_frame(&[-0.75, -1.0]).unwrap();
        w.write_frame(&[0.5, 0.0]).unwrap();
        w.finish().unwrap();
        let a = read_audio(&streamed).unwrap();
        let b = read_audio(&bulk).unwrap();
        assert_eq!(a.sample_rate, b.sample_rate);
        assert_eq!(a.channels(), b.channels());
        assert_eq!(a.channel(0), b.channel(0));
        assert_eq!(a.channel(1), b.channel(1));
    }
}
