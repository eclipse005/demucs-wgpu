//! wgpu device layer.
//!
//! The shape of this module follows the two prior Rust+wgpu ports in this
//! series (the Qwen3 ASR and forced-aligner ports, Apache-2.0), which solved the
//! same problems on the same machine: adapter ranking, 16-byte-padded storage
//! buffers, a `push_error_scope` pipeline builder that turns WGSL validation
//! errors into a labelled `Result`, and bounded staging so a multi-hundred-MB
//! upload cannot blow up WDDM. See `NOTICE` for attribution.

pub mod arena;
pub mod htdemucs;
pub mod kernels;
pub mod shaders;

use crate::error::{Error, Result};

/// True when a 32-lane xor butterfly reduces correctly on this adapter.
///
/// The `SUBGROUP` feature bit alone is not enough: a driver may grant a *width
/// range* (8..=32), and a shuffle that assumes 32 lanes then folds across the
/// wrong ones, producing plausible-looking garbage with no error raised
/// anywhere. Only an exact 32..=32 promise makes the shuffle path safe; anything
/// else must take the shared-memory fallback, which is bit-identical.
fn shuffle_reduction_is_safe(info: &wgpu::AdapterInfo, features: wgpu::Features) -> bool {
    features.contains(wgpu::Features::SUBGROUP)
        && info.subgroup_min_size == 32
        && info.subgroup_max_size == 32
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DeviceSelector {
    /// Best discrete GPU, then integrated, then whatever is left.
    #[default]
    Auto,
    /// A specific runtime and the n-th adapter of it, e.g. `vulkan:0`.
    Runtime { api: wgpu::Backend, index: usize },
    /// A substring of the adapter name, e.g. `nvidia`.
    Name(String),
}

impl DeviceSelector {
    pub fn parse(spec: &str) -> Result<Self> {
        let spec = spec.trim();
        if spec.is_empty() || spec.eq_ignore_ascii_case("auto") {
            return Ok(DeviceSelector::Auto);
        }
        // `<runtime>[:<index>]`
        let (head, tail) = match spec.split_once(':') {
            Some((head, tail)) => (head, Some(tail)),
            None => (spec, None),
        };
        let runtime = match head.to_ascii_lowercase().as_str() {
            "vulkan" => Some(wgpu::Backend::Vulkan),
            "metal" => Some(wgpu::Backend::Metal),
            "dx12" => Some(wgpu::Backend::Dx12),
            "gl" => Some(wgpu::Backend::Gl),
            _ => None,
        };
        if let Some(api) = runtime {
            let index = match tail {
                Some(text) => text
                    .parse::<usize>()
                    .map_err(|e| Error::Gpu(format!("bad adapter index `{text}`: {e}")))?,
                None => 0,
            };
            return Ok(DeviceSelector::Runtime { api, index });
        }
        if tail.is_some() {
            return Err(Error::Gpu(format!(
                "`{spec}` is neither a runtime (`vulkan`, `metal`, `dx12`, `gl`) nor a plain name"
            )));
        }
        Ok(DeviceSelector::Name(spec.to_string()))
    }

    fn backends(&self) -> wgpu::Backends {
        match self {
            DeviceSelector::Runtime { api, .. } => wgpu::Backends::from(*api),
            _ => wgpu::Backends::all(),
        }
    }
}

fn rank(info: &wgpu::AdapterInfo) -> (u8, u8) {
    let class = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        _ => 3,
    };
    let api = match info.backend {
        wgpu::Backend::Vulkan => 0,
        wgpu::Backend::Metal => 1,
        wgpu::Backend::Dx12 => 2,
        _ => 3,
    };
    (class, api)
}

/// Facts about the chosen adapter that decide which kernels may run.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub adapter: wgpu::AdapterInfo,
    pub shuffle_reduction: bool,
    pub max_binding_bytes: u64,
    pub max_buffer_bytes: u64,
    pub subgroup_min: u32,
    pub subgroup_max: u32,
}

impl DeviceInfo {
    pub fn describe(&self) -> String {
        format!(
            "{:?} {} ({} {}, driver {}) binding {} MiB, subgroup {}",
            self.adapter.backend,
            self.adapter.name,
            self.adapter.vendor,
            match self.adapter.device_type {
                wgpu::DeviceType::DiscreteGpu => "dGPU",
                wgpu::DeviceType::IntegratedGpu => "iGPU",
                wgpu::DeviceType::Cpu => "CPU",
                wgpu::DeviceType::VirtualGpu => "vGPU",
                _ => "other",
            },
            if self.adapter.driver_info.is_empty() {
                self.adapter.driver.as_str()
            } else {
                self.adapter.driver_info.as_str()
            },
            self.max_binding_bytes / (1024 * 1024),
            if self.subgroup_min == self.subgroup_max {
                format!("{}", self.subgroup_min)
            } else {
                format!("{}..{}", self.subgroup_min, self.subgroup_max)
            }
        )
    }
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: DeviceInfo,
}

/// One adapter as `demucs adapters` shows it. `index` is the one a
/// `DeviceSelector::Runtime` addresses within `runtime`.
#[derive(Debug, Clone)]
pub struct AdapterEntry {
    pub runtime: String,
    pub index: usize,
    pub name: String,
    pub kind: String,
}

impl Gpu {
    pub fn new() -> Result<Self> {
        Self::with_selector(DeviceSelector::Auto)
    }

    /// Every adapter wgpu can see, grouped by runtime and ranked within it, so a
    /// caller can show what `--device` accepts. The index is the one
    /// `DeviceSelector::Runtime` addresses (`vulkan:1` picks the second Vulkan
    /// adapter); the first entry of each runtime is also what a bare `vulkan`
    /// picks.
    pub fn list_adapters() -> Vec<AdapterEntry> {
        let instance = wgpu::Instance::default();
        let all: Vec<wgpu::AdapterInfo> = pollster::block_on(
            instance.enumerate_adapters(wgpu::Backends::all()),
        )
        .into_iter()
        .map(|adapter| adapter.get_info())
        .collect();
        let mut ranked: Vec<((u8, u8), wgpu::AdapterInfo)> = all
            .into_iter()
            .map(|info| (rank(&info), info))
            .collect();
        ranked.sort_by_key(|(rank, _)| *rank);
        let mut out: Vec<AdapterEntry> = Vec::new();
        for backend in [
            wgpu::Backend::Vulkan,
            wgpu::Backend::Dx12,
            wgpu::Backend::Gl,
            wgpu::Backend::Metal,
        ] {
            let mut index = 0usize;
            for (_, info) in ranked.iter().filter(|(_, info)| info.backend == backend) {
                out.push(AdapterEntry {
                    runtime: format!("{backend:?}").to_lowercase(),
                    index,
                    name: info.name.clone(),
                    kind: match info.device_type {
                        wgpu::DeviceType::DiscreteGpu => "dGPU",
                        wgpu::DeviceType::IntegratedGpu => "iGPU",
                        wgpu::DeviceType::Cpu => "CPU",
                        wgpu::DeviceType::VirtualGpu => "vGPU",
                        _ => "other",
                    }
                    .to_string(),
                });
                index += 1;
            }
        }
        out
    }

    pub fn with_selector(selector: DeviceSelector) -> Result<Self> {
        pollster::block_on(Self::with_selector_async(selector))
    }

    pub async fn with_selector_async(selector: DeviceSelector) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let mut hits: Vec<wgpu::Adapter> = instance
            .enumerate_adapters(selector.backends())
            .await
            .into_iter()
            .filter(|adapter| match &selector {
                DeviceSelector::Auto => true,
                DeviceSelector::Runtime { api, .. } => adapter.get_info().backend == *api,
                DeviceSelector::Name(needle) => adapter
                    .get_info()
                    .name
                    .to_ascii_lowercase()
                    .contains(&needle.to_ascii_lowercase()),
            })
            .collect();

        if hits.is_empty() {
            return Err(Error::Gpu(format!(
                "no adapter matched {selector:?}; run `demucs kernels` to list what is available"
            )));
        }
        hits.sort_by_key(|a| rank(&a.get_info()));

        let adapter = match &selector {
            DeviceSelector::Runtime { index, .. } => hits.get(*index).ok_or_else(|| {
                Error::Gpu(format!(
                    "adapter index {index} does not exist ({} found)",
                    hits.len()
                ))
            })?,
            _ => &hits[0],
        };

        let adapter_info = adapter.get_info();
        let features = adapter.features();
        let limits = adapter.limits();
        // Asking for exactly what the adapter reports means request_device cannot
        // fail on limits; anything the kernels need beyond this is checked by the
        // caller against `DeviceInfo`.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("demucs-wgpu"),
                required_features: features
                    & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SUBGROUP),
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Gpu(format!("request_device: {e}")))?;

        device.on_uncaptured_error(std::sync::Arc::new(|error| {
            eprintln!("wgpu error: {error}");
        }));

        let info = DeviceInfo {
            shuffle_reduction: shuffle_reduction_is_safe(&adapter_info, features),
            max_binding_bytes: limits.max_storage_buffer_binding_size as u64,
            max_buffer_bytes: limits.max_buffer_size,
            subgroup_min: adapter_info.subgroup_min_size,
            subgroup_max: adapter_info.subgroup_max_size,
            adapter: adapter_info,
        };

        Ok(Self {
            device,
            queue,
            info,
        })
    }

    /// Storage buffer, 16-byte aligned and padded, with the copy usages a
    /// readback or an upload needs.
    pub fn storage(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        let size = bytes.div_ceil(16) * 16;
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// A buffer the arena hands out regions of: storage for activations, uniform
    /// for the per-dispatch parameter blocks, and copyable in both directions for
    /// the single upload and readback a forward pass needs.
    ///
    /// One buffer cannot be a `storage` and a `uniform` binding unless it declares
    /// both usages, which is why the arena does not reuse [`Gpu::storage`].
    ///
    /// The out-of-memory scope is the point: a buffer the driver refused to back
    /// otherwise surfaces much later as a cascade of "bind group ... buffer is
    /// invalid" validation errors, none of which name the allocation that failed.
    pub fn scratch(&self, label: &str, bytes: u64) -> std::result::Result<wgpu::Buffer, String> {
        let size = bytes.div_ceil(256) * 256;
        let guard = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(256),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        match pollster::block_on(guard.pop()) {
            Some(error) => Err(format!(
                "{label} needs {} MiB of device memory: {error}",
                size.max(256) / (1024 * 1024)
            )),
            None => Ok(buffer),
        }
    }

    pub fn uniform(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        let size = bytes.div_ceil(16) * 16;
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn upload(&self, buffer: &wgpu::Buffer, data: &[u8]) {
        self.queue.write_buffer(buffer, 0, data);
    }

    /// Waits for every submitted command to finish.
    pub fn flush(&self) -> Result<()> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("poll: {e}")))?;
        Ok(())
    }

    /// Copies `bytes` back from `buffer` through a staging buffer.
    pub fn readback(&self, buffer: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>> {
        let started = std::time::Instant::now();
        let out = self.readback_inner(buffer, bytes);
        if std::env::var("DEMUCS_STAGE_TIMING").is_ok() {
            eprintln!(
                "[stg]   readback {} MiB {:?}",
                bytes / (1024 * 1024),
                started.elapsed()
            );
        }
        out
    }

    fn readback_inner(&self, buffer: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>> {
        let size = bytes.div_ceil(4) * 4;
        let staging = self.staging(size.max(4));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size.max(4));
        self.queue.submit([encoder.finish()]);
        self.flush()?;
        self.mapped_bytes(&staging, bytes)
    }

    /// A host-visible buffer sized for `bytes` of readback.
    ///
    /// Public so a pass can record its own copy into it (`Recorder::
    /// copy_to_staging`) and the wait for that pass can happen later, which is
    /// what lets one chunk's readback overlap the next chunk's forward.
    pub fn staging(&self, bytes: u64) -> wgpu::Buffer {
        let size = bytes.div_ceil(4) * 4;
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: size.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Waits for one specific submission rather than for the whole queue.
    pub fn wait_for(&self, submission: wgpu::SubmissionIndex) -> Result<()> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|e| Error::Gpu(format!("poll for submission: {e}")))?;
        Ok(())
    }

    /// Maps a staging buffer filled by an earlier `copy_to_staging` and copies
    /// it out. Requires the submission that recorded the copy to have completed.
    pub fn mapped_bytes(&self, staging: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>> {
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("poll for readback: {e}")))?;
        rx.recv()
            .map_err(|_| Error::Gpu("map callback dropped".into()))?
            .map_err(|e| Error::Gpu(format!("map buffer: {e}")))?;

        let data = slice
            .get_mapped_range()
            .map_err(|e| Error::Gpu(format!("mapped range: {e}")))?
            .to_vec();
        staging.unmap();
        let mut data = data;
        data.truncate(bytes as usize);
        Ok(data)
    }

    /// Builds a compute pipeline, surfacing WGSL validation errors as errors
    /// rather than as an opaque panic on first dispatch.
    pub fn pipeline(
        &self,
        label: &str,
        wgsl: &str,
        entry: &str,
    ) -> Result<wgpu::ComputePipeline> {
        let guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
        if let Some(error) = pollster::block_on(guard.pop()) {
            return Err(Error::Gpu(format!("pipeline {label} failed validation: {error}")));
        }
        Ok(pipeline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_parses_runtimes_and_names() {
        assert_eq!(DeviceSelector::parse("auto").unwrap(), DeviceSelector::Auto);
        assert_eq!(DeviceSelector::parse("").unwrap(), DeviceSelector::Auto);
        assert_eq!(
            DeviceSelector::parse("vulkan").unwrap(),
            DeviceSelector::Runtime {
                api: wgpu::Backend::Vulkan,
                index: 0
            }
        );
        assert_eq!(
            DeviceSelector::parse("vulkan:1").unwrap(),
            DeviceSelector::Runtime {
                api: wgpu::Backend::Vulkan,
                index: 1
            }
        );
        assert_eq!(
            DeviceSelector::parse("nvidia").unwrap(),
            DeviceSelector::Name("nvidia".into())
        );
    }

    #[test]
    fn selector_rejects_an_index_on_a_name() {
        assert!(DeviceSelector::parse("nvidia:2").is_err());
        assert!(DeviceSelector::parse("vulkan:x").is_err());
    }

    /// `AdapterInfo` has no `Default` and keeps gaining fields, so build one from
    /// a real adapter rather than listing them.
    fn adapter_info_with_width(min: u32, max: u32) -> wgpu::AdapterInfo {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
            .into_iter()
            .next()
            .expect("a test machine running this suite has an adapter");
        let mut info = adapter.get_info();
        info.subgroup_min_size = min;
        info.subgroup_max_size = max;
        info
    }

    #[test]
    fn shuffle_reduction_requires_an_exact_32_lane_promise() {
        // Exactly 32 and the feature: safe.
        assert!(shuffle_reduction_is_safe(
            &adapter_info_with_width(32, 32),
            wgpu::Features::SUBGROUP
        ));
        // The trap: a range that includes 32 still folds across the wrong lanes.
        assert!(!shuffle_reduction_is_safe(
            &adapter_info_with_width(8, 32),
            wgpu::Features::SUBGROUP
        ));
        // No feature bit, no shuffle.
        assert!(!shuffle_reduction_is_safe(
            &adapter_info_with_width(32, 32),
            wgpu::Features::empty()
        ));
    }
}
