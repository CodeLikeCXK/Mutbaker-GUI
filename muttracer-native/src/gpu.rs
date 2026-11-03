use anyhow::{Context, Result};
use std::path::PathBuf;
use std::time::Duration;

pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    pub fn new_dx12() -> Result<Self> {
        let backends = wgpu::Backends::DX12;
        let mut dx12_options = wgpu::Dx12BackendOptions::default();

        let dxc_candidates = [
            PathBuf::from("dxcompiler.dll"),
            std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.join("dxcompiler.dll"))).unwrap_or_default(),
            PathBuf::from(r"C:\Bevy\dxcompiler.dll"),
        ];

        for candidate in &dxc_candidates {
            if candidate.is_file() {
                eprintln!("[gpu] Using DXC compiler: {}", candidate.display());
                dx12_options.shader_compiler = wgpu::Dx12Compiler::DynamicDxc {
                    dxc_path: candidate.to_string_lossy().to_string(),
                };
                break;
            }
        }

        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = backends;
        desc.backend_options.dx12 = dx12_options;
        let instance = wgpu::Instance::new(desc);

        let adapters = block_on(instance.enumerate_adapters(backends));
        if adapters.is_empty() {
            anyhow::bail!("No DX12 adapters found");
        }

        eprintln!("[gpu] Available DX12 adapters:");
        for (i, a) in adapters.iter().enumerate() {
            let info = a.get_info();
            eprintln!("  [{i}] {} ({:?}, driver: {})", info.name, info.device_type, info.driver);
        }

        // Prefer discrete or integrated hardware GPU over CPU WARP
        let adapter = adapters
            .into_iter()
            .find(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
            .or_else(|| {
                block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                    ..Default::default()
                }))
                .ok()
            })
            .context("No suitable DX12 GPU adapter available")?;

        let adapter_info = adapter.get_info();
        eprintln!(
            "[gpu] Selected: {} ({:?}, backend: {:?})",
            adapter_info.name, adapter_info.device_type, adapter_info.backend
        );

        let limits = adapter.limits();
        eprintln!(
            "[gpu] Limits: max_buffer_size = {} MB, max_storage_buffer_binding_size = {} MB",
            limits.max_buffer_size / (1024 * 1024),
            limits.max_storage_buffer_binding_size / (1024 * 1024)
        );

        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("muttracer-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: Default::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }))
        .context("Failed to create wgpu device on DX12")?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            adapter_info,
        })
    }

    pub fn wait(&self, timeout: Duration) -> Result<()> {
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(timeout),
        });
        Ok(())
    }
}

pub fn block_on<F: std::future::Future>(mut future: F) -> F::Output {
    use std::pin::Pin;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    fn dummy_raw_waker() -> RawWaker {
        fn no_op(_: *const ()) {}
        fn clone(_: *const ()) -> RawWaker {
            dummy_raw_waker()
        }
        let vtable = &RawWakerVTable::new(clone, no_op, no_op, no_op);
        RawWaker::new(std::ptr::null(), vtable)
    }

    let waker = unsafe { Waker::from_raw(dummy_raw_waker()) };
    let mut cx = Context::from_waker(&waker);
    let mut future = unsafe { Pin::new_unchecked(&mut future) };
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(val) => return val,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

