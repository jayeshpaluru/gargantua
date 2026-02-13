use anyhow::{anyhow, Context, Result};
use bytemuck::{Pod, Zeroable};
use clap::{Parser, ValueEnum};
use egui::{Color32, RichText};
use egui_wgpu::wgpu;
use image::{ImageBuffer, Rgba};
use std::f32::consts::PI;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowBuilder;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Backend {
    Cpu,
    Gpu,
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Kerr black hole ray tracer (CPU/GPU + realtime)")]
struct Cli {
    #[arg(long, default_value_t = 1280)]
    width: u32,
    #[arg(long, default_value_t = 720)]
    height: u32,
    #[arg(long, default_value_t = 65.0)]
    fov_deg: f32,
    #[arg(long, default_value_t = 0.92)]
    spin: f32,
    #[arg(long, default_value_t = 70.0)]
    observer_r: f32,
    #[arg(long, default_value_t = 60.0)]
    inclination_deg: f32,
    #[arg(long, default_value_t = 3000)]
    max_steps: u32,
    #[arg(long, default_value_t = 0.03)]
    step_size: f32,
    #[arg(long, default_value_t = 1.7)]
    disk_inner: f32,
    #[arg(long, default_value_t = 20.0)]
    disk_outer: f32,
    #[arg(long, default_value_t = 2.2)]
    emissivity_power: f32,
    #[arg(long, default_value_t = 4)]
    spp: u32,
    #[arg(long, default_value = "gargantua.png")]
    output: String,
    #[arg(long, value_enum, default_value_t = Backend::Gpu)]
    backend: Backend,
    #[arg(long, default_value_t = false)]
    realtime: bool,
}

#[derive(Clone, Copy)]
struct Physics {
    spin: f32,
    observer_r: f32,
    inclination_rad: f32,
    step_size: f32,
    max_steps: u32,
    disk_inner: f32,
    disk_outer: f32,
    emissivity_power: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct OfflineGpuUniforms {
    width: u32,
    height: u32,
    max_steps: u32,
    spp: u32,
    seed: u32,
    _pad0: [u32; 4],
    fov_y_rad: f32,
    spin: f32,
    observer_r: f32,
    inclination_rad: f32,
    step_size: f32,
    disk_inner: f32,
    disk_outer: f32,
    emissivity_power: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RealtimeUniforms {
    width: u32,
    height: u32,
    max_steps: u32,
    spp: u32,
    frame_index: u32,
    _pad0: [u32; 3],
    jitter: [f32; 2],
    _pad1: [f32; 2],
    fov_y_rad: f32,
    spin: f32,
    observer_r: f32,
    inclination_rad: f32,
    step_size: f32,
    disk_inner: f32,
    disk_outer: f32,
    emissivity_power: f32,
    exposure: f32,
    glow_strength: f32,
    camera_yaw: f32,
    camera_pitch: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RealtimeGpuUniforms {
    dims: [u32; 4],       // width, height, max_steps, spp
    frame: [u32; 4],      // frame_index + padding
    jitter: [f32; 4],     // jitter.xy + padding
    camera0: [f32; 4],    // fov, spin, observer_r, inclination
    camera1: [f32; 4],    // step, inner, outer, emissivity
    camera2: [f32; 4],    // exposure, glow, yaw, pitch
}

impl RealtimeUniforms {
    fn as_gpu(self) -> RealtimeGpuUniforms {
        RealtimeGpuUniforms {
            dims: [self.width, self.height, self.max_steps, self.spp],
            frame: [self.frame_index, 0, 0, 0],
            jitter: [self.jitter[0], self.jitter[1], 0.0, 0.0],
            camera0: [self.fov_y_rad, self.spin, self.observer_r, self.inclination_rad],
            camera1: [self.step_size, self.disk_inner, self.disk_outer, self.emissivity_power],
            camera2: [self.exposure, self.glow_strength, self.camera_yaw, self.camera_pitch],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn normalize(self) -> Self {
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z;
        let inv = len2.max(1e-12).sqrt().recip();
        Self {
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
        }
    }

    fn mul(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    fn add(self, o: Self) -> Self {
        Self {
            x: self.x + o.x,
            y: self.y + o.y,
            z: self.z + o.z,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    validate(&cli)?;

    if cli.realtime {
        if let Backend::Cpu = cli.backend {
            return Err(anyhow!("realtime mode requires --backend gpu"));
        }
        return run_realtime(cli);
    }

    let (pixels, actual_backend) = match cli.backend {
        Backend::Cpu => (render_cpu(&cli), Backend::Cpu),
        Backend::Gpu => match pollster::block_on(render_gpu_offline(&cli)) {
            Ok(px) => (px, Backend::Gpu),
            Err(err) => {
                eprintln!("GPU render failed, falling back to CPU: {err:#}");
                (render_cpu(&cli), Backend::Cpu)
            }
        },
    };

    let img = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(cli.width, cli.height, pixels)
        .context("failed to build output image")?;
    img.save(&cli.output)
        .with_context(|| format!("failed to save {}", cli.output))?;

    println!(
        "saved {} ({}x{}) using {:?}",
        cli.output, cli.width, cli.height, actual_backend
    );

    Ok(())
}

fn validate(cli: &Cli) -> Result<()> {
    if cli.spin.abs() >= 1.0 {
        return Err(anyhow!("spin must satisfy |a| < 1"));
    }
    if cli.disk_inner <= 0.0 || cli.disk_outer <= cli.disk_inner {
        return Err(anyhow!("disk radii must satisfy 0 < disk_inner < disk_outer"));
    }
    if cli.spp == 0 {
        return Err(anyhow!("spp must be >= 1"));
    }
    if cli.step_size <= 0.0 || cli.max_steps == 0 {
        return Err(anyhow!("step_size and max_steps must be positive"));
    }
    Ok(())
}

fn physics_from_cli(cli: &Cli) -> Physics {
    Physics {
        spin: cli.spin,
        observer_r: cli.observer_r,
        inclination_rad: cli.inclination_deg.to_radians(),
        step_size: cli.step_size,
        max_steps: cli.max_steps,
        disk_inner: cli.disk_inner,
        disk_outer: cli.disk_outer,
        emissivity_power: cli.emissivity_power,
    }
}

fn render_cpu(cli: &Cli) -> Vec<u8> {
    let mut out = vec![0u8; (cli.width * cli.height * 4) as usize];
    let phys = physics_from_cli(cli);

    for y in 0..cli.height {
        for x in 0..cli.width {
            let mut color = Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            };

            for s in 0..cli.spp {
                let (jx, jy) = sample_jitter_u32(x, y, s, 0);
                let dir = pixel_dir_camera(x, y, cli.width, cli.height, cli.fov_deg.to_radians(), jx, jy);
                color = color.add(trace_ray(dir, phys));
            }
            color = color.mul(1.0 / cli.spp as f32);

            let idx = ((y * cli.width + x) * 4) as usize;
            out[idx] = to_u8(color.x);
            out[idx + 1] = to_u8(color.y);
            out[idx + 2] = to_u8(color.z);
            out[idx + 3] = 255;
        }
    }

    out
}

fn create_instance() -> wgpu::Instance {
    #[cfg(target_os = "macos")]
    {
        wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..Default::default()
        })
    }
    #[cfg(not(target_os = "macos"))]
    {
        wgpu::Instance::default()
    }
}

async fn render_gpu_offline(cli: &Cli) -> Result<Vec<u8>> {
    let instance = create_instance();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .context("no GPU adapter found")?;
    let info = adapter.get_info();
    println!("using GPU adapter: {} ({:?})", info.name, info.backend);

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gargantua-offline-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await?;

    let uniforms = OfflineGpuUniforms {
        width: cli.width,
        height: cli.height,
        max_steps: cli.max_steps,
        spp: cli.spp,
        seed: 1,
        _pad0: [0; 4],
        fov_y_rad: cli.fov_deg.to_radians(),
        spin: cli.spin,
        observer_r: cli.observer_r,
        inclination_rad: cli.inclination_deg.to_radians(),
        step_size: cli.step_size,
        disk_inner: cli.disk_inner,
        disk_outer: cli.disk_outer,
        emissivity_power: cli.emissivity_power,
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("offline-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let pixel_count = (cli.width * cli.height) as usize;
    let out_size = (pixel_count * std::mem::size_of::<u32>()) as u64;

    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("offline-storage"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("offline-readback"),
        size: out_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("offline-raytrace-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/raytrace.wgsl").into()),
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("offline-bind-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("offline-bind-group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: storage_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("offline-pipeline-layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("offline-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("offline-encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("offline-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(cli.width.div_ceil(8), cli.height.div_ceil(8), 1);
    }

    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &readback, 0, out_size);
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().context("failed waiting for GPU map")??;

    let data = slice.get_mapped_range();
    let packed = bytemuck::cast_slice::<u8, u32>(&data);
    let mut out = vec![0u8; pixel_count * 4];

    for (i, px) in packed.iter().enumerate() {
        let b = (px & 0xff) as u8;
        let g = ((px >> 8) & 0xff) as u8;
        let r = ((px >> 16) & 0xff) as u8;
        let a = ((px >> 24) & 0xff) as u8;
        let base = i * 4;
        out[base] = r;
        out[base + 1] = g;
        out[base + 2] = b;
        out[base + 3] = a;
    }

    drop(data);
    readback.unmap();

    Ok(out)
}

fn run_realtime(cli: Cli) -> Result<()> {
    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Gargantua - Realtime Kerr Raytracing")
            .with_inner_size(PhysicalSize::new(cli.width, cli.height))
            .build(&event_loop)
            .context("failed creating window")?,
    );

    let mut state = pollster::block_on(RealtimeState::new(window.clone(), cli.clone()))?;

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                let consumed = state.on_window_event(&event);
                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(size) => {
                        state.resize(size.width, size.height);
                    }
                    WindowEvent::RedrawRequested => {
                        if let Err(err) = state.render() {
                            eprintln!("render error: {err:#}");
                            elwt.exit();
                        }
                    }
                    _ => {
                        if consumed {
                            window.request_redraw();
                        }
                    }
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    })?;

    Ok(())
}

struct RealtimeState {
    window: Arc<winit::window::Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    uniforms: RealtimeUniforms,
    uniform_buffer: wgpu::Buffer,
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    empty_bg: wgpu::BindGroup,
    accum_a: wgpu::Texture,
    accum_b: wgpu::Texture,
    display_tex: wgpu::Texture,
    display_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    compute_bg_ab: wgpu::BindGroup,
    compute_bg_ba: wgpu::BindGroup,
    render_bg: wgpu::BindGroup,
    frame_index: u32,
    ping: bool,
    paused: bool,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    egui_screen_desc: egui_wgpu::ScreenDescriptor,
    frame_timer: Instant,
    fps_window_start: Instant,
    fps_frames: u32,
    fps_value: f32,
    needs_reset: bool,
    mouse_look: bool,
    last_cursor: Option<(f64, f64)>,
    export_frames: u32,
    export_prefix: String,
}

impl RealtimeState {
    async fn new(window: Arc<winit::window::Window>, cli: Cli) -> Result<Self> {
        let instance = create_instance();
        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .context("no adapter for realtime surface")?;
        let info = adapter.get_info();
        println!("using realtime GPU adapter: {} ({:?})", info.name, info.backend);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("realtime-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let uniforms = RealtimeUniforms {
            width: config.width,
            height: config.height,
            max_steps: cli.max_steps,
            spp: cli.spp,
            frame_index: 0,
            _pad0: [0; 3],
            jitter: [0.0, 0.0],
            _pad1: [0.0; 2],
            fov_y_rad: cli.fov_deg.to_radians(),
            spin: cli.spin,
            observer_r: cli.observer_r,
            inclination_rad: cli.inclination_deg.to_radians(),
            step_size: cli.step_size,
            disk_inner: cli.disk_inner,
            disk_outer: cli.disk_outer,
            emissivity_power: cli.emissivity_power,
            exposure: 1.15,
            glow_strength: 0.35,
            camera_yaw: 0.0,
            camera_pitch: 0.0,
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("realtime-uniforms"),
            contents: bytemuck::bytes_of(&uniforms.as_gpu()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("realtime-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/realtime.wgsl").into()),
        });

        let (accum_a, accum_b, display_tex, display_view) =
            create_realtime_textures(&device, config.width, config.height);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("display-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let accum_a_view = accum_a.create_view(&Default::default());
        let accum_b_view = accum_b.create_view(&Default::default());

        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("realtime-compute-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let display_storage_view = display_tex.create_view(&Default::default());

        let compute_bg_ab = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute-bg-ab"),
            layout: &compute_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&accum_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&accum_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&display_storage_view),
                },
            ],
        });

        let compute_bg_ba = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute-bg-ba"),
            layout: &compute_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&accum_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&accum_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&display_storage_view),
                },
            ],
        });

        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("realtime-compute-pipeline-layout"),
            bind_group_layouts: &[&compute_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("realtime-compute-pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "cs_main",
        });

        let render_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("realtime-render-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let empty_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("realtime-empty-layout"),
            entries: &[],
        });
        let empty_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("realtime-empty-bg"),
            layout: &empty_layout,
            entries: &[],
        });

        let render_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("realtime-render-bg"),
            layout: &render_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&display_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("realtime-render-pipeline-layout"),
            bind_group_layouts: &[&empty_layout, &render_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("realtime-render-pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let egui_ctx = egui::Context::default();
        egui_ctx.set_visuals(egui::Visuals {
            window_fill: Color32::from_rgb(8, 11, 18),
            panel_fill: Color32::from_rgb(10, 14, 23),
            override_text_color: Some(Color32::from_rgb(220, 228, 242)),
            ..egui::Visuals::dark()
        });
        let max_texture_side = device.limits().max_texture_dimension_2d as usize;
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            Some(max_texture_side),
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1);
        let initial_size = [config.width, config.height];

        let mut state = Self {
            window,
            surface,
            device,
            queue,
            config,
            uniforms,
            uniform_buffer,
            compute_pipeline,
            render_pipeline,
            accum_a,
            accum_b,
            display_tex,
            display_view,
            sampler,
            empty_bg,
            compute_bg_ab,
            compute_bg_ba,
            render_bg,
            frame_index: 0,
            ping: false,
            paused: false,
            egui_ctx,
            egui_state,
            egui_renderer,
            egui_screen_desc: egui_wgpu::ScreenDescriptor {
                size_in_pixels: initial_size,
                pixels_per_point: 1.0,
            },
            frame_timer: Instant::now(),
            fps_window_start: Instant::now(),
            fps_frames: 0,
            fps_value: 0.0,
            needs_reset: false,
            mouse_look: false,
            last_cursor: None,
            export_frames: 240,
            export_prefix: "frames/gargantua".to_string(),
        };
        state.egui_screen_desc.pixels_per_point = state.window.scale_factor() as f32;

        state.clear_accum();
        Ok(state)
    }

    fn on_window_event(&mut self, event: &WindowEvent) -> bool {
        let consumed = self.egui_state.on_window_event(&self.window, event).consumed;
        if !consumed {
            self.handle_camera_input(event);
        }
        consumed
    }

    fn handle_camera_input(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::MouseInput { state, button, .. } if *button == MouseButton::Right => {
                self.mouse_look = *state == ElementState::Pressed;
                if !self.mouse_look {
                    self.last_cursor = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } if self.mouse_look => {
                if let Some((lx, ly)) = self.last_cursor {
                    let dx = (position.x - lx) as f32;
                    let dy = (position.y - ly) as f32;
                    self.uniforms.camera_yaw += dx * 0.003;
                    self.uniforms.camera_pitch = (self.uniforms.camera_pitch + dy * 0.003).clamp(-1.35, 1.35);
                    self.needs_reset = true;
                }
                self.last_cursor = Some((position.x, position.y));
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    let mut changed = false;
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::KeyW) => {
                            self.uniforms.observer_r -= 2.0;
                            changed = true;
                        }
                        PhysicalKey::Code(KeyCode::KeyS) => {
                            self.uniforms.observer_r += 2.0;
                            changed = true;
                        }
                        PhysicalKey::Code(KeyCode::KeyA) => {
                            self.uniforms.inclination_rad -= 2.0_f32.to_radians();
                            changed = true;
                        }
                        PhysicalKey::Code(KeyCode::KeyD) => {
                            self.uniforms.inclination_rad += 2.0_f32.to_radians();
                            changed = true;
                        }
                        PhysicalKey::Code(KeyCode::Space) => {
                            self.paused = !self.paused;
                        }
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            self.needs_reset = true;
                        }
                        _ => {}
                    }
                    if changed {
                        self.sanitize_uniforms();
                        self.needs_reset = true;
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.04,
                };
                self.uniforms.fov_y_rad =
                    (self.uniforms.fov_y_rad - amount.to_radians() * 0.8).clamp(20f32.to_radians(), 110f32.to_radians());
                self.needs_reset = true;
            }
            _ => {}
        }
    }

    fn clear_accum(&mut self) {
        let byte_count = (self.config.width as usize) * (self.config.height as usize) * 8;
        let zeros = vec![0u8; byte_count];
        for tex in [&self.accum_a, &self.accum_b] {
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &zeros,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(8 * self.config.width),
                    rows_per_image: Some(self.config.height),
                },
                wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        self.frame_index = 0;
        self.ping = false;
    }

    fn resize(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if self.config.width == width && self.config.height == height {
            return;
        }

        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.egui_screen_desc.size_in_pixels = [width, height];
        self.egui_screen_desc.pixels_per_point = self.window.scale_factor() as f32;

        let (accum_a, accum_b, display_tex, display_view) =
            create_realtime_textures(&self.device, width, height);
        self.accum_a = accum_a;
        self.accum_b = accum_b;
        self.display_tex = display_tex;
        self.display_view = display_view;

        let accum_a_view = self.accum_a.create_view(&Default::default());
        let accum_b_view = self.accum_b.create_view(&Default::default());
        let display_storage_view = self.display_tex.create_view(&Default::default());

        let compute_layout = self.compute_pipeline.get_bind_group_layout(0);
        self.compute_bg_ab = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute-bg-ab"),
            layout: &compute_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&accum_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&accum_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&display_storage_view),
                },
            ],
        });
        self.compute_bg_ba = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute-bg-ba"),
            layout: &compute_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&accum_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&accum_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&display_storage_view),
                },
            ],
        });

        let render_layout = self.render_pipeline.get_bind_group_layout(1);
        self.render_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("realtime-render-bg"),
            layout: &render_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.display_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        self.clear_accum();
    }

    fn render(&mut self) -> Result<()> {
        self.update_fps();
        let mut ui_uniforms = self.uniforms;
        let mut ui_paused = self.paused;
        let mut ui_reset_clicked = false;
        let mut ui_export_clicked = false;
        let mut ui_export_frames = self.export_frames;
        let mut ui_export_prefix = self.export_prefix.clone();

        let raw_input = self.egui_state.take_egui_input(&self.window);
        let egui_output = self.egui_ctx.run(raw_input, |ctx| {
            Self::draw_controls_ui(
                ctx,
                &mut ui_uniforms,
                &mut ui_paused,
                &mut ui_reset_clicked,
                &mut ui_export_clicked,
                &mut ui_export_frames,
                &mut ui_export_prefix,
                self.fps_value,
                self.frame_index,
            );
        });
        self.egui_state.handle_platform_output(&self.window, egui_output.platform_output);

        for (id, delta) in &egui_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }

        let ui_changed = ui_uniforms != self.uniforms || ui_paused != self.paused || ui_reset_clicked;
        self.uniforms = ui_uniforms;
        self.paused = ui_paused;
        self.export_frames = ui_export_frames.clamp(1, 2400);
        self.export_prefix = ui_export_prefix;

        if ui_changed || self.needs_reset {
            self.sanitize_uniforms();
            self.clear_accum();
            self.needs_reset = false;
        }

        if ui_export_clicked {
            let export_frames = self.export_frames;
            let export_prefix = self.export_prefix.clone();
            self.export_sequence(export_frames, &export_prefix)?;
        }

        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            Err(wgpu::SurfaceError::Timeout) => {
                return Ok(());
            }
            Err(e) => return Err(anyhow!(e)),
        };

        let (jx, jy) = halton_2d(self.frame_index + 1);
        self.uniforms.width = self.config.width;
        self.uniforms.height = self.config.height;
        self.uniforms.frame_index = self.frame_index;
        self.uniforms.jitter = [jx - 0.5, jy - 0.5];

        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms.as_gpu()));

        let swap_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("realtime-encoder"),
            });

        if !self.paused {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("realtime-compute-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_pipeline);
            let bg = if self.ping {
                &self.compute_bg_ba
            } else {
                &self.compute_bg_ab
            };
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(self.config.width.div_ceil(8), self.config.height.div_ceil(8), 1);
        }

        let paint_jobs = self
            .egui_ctx
            .tessellate(egui_output.shapes, self.egui_screen_desc.pixels_per_point);
        let _cbs = self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &self.egui_screen_desc,
        );

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("realtime-render-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &swap_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.empty_bg, &[]);
            pass.set_bind_group(1, &self.render_bg, &[]);
            pass.draw(0..3, 0..1);
            self.egui_renderer
                .render(&mut pass, &paint_jobs, &self.egui_screen_desc);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();

        if !self.paused {
            self.frame_index = self.frame_index.saturating_add(1);
            self.ping = !self.ping;
        }

        for id in &egui_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.window.request_redraw();
        Ok(())
    }

    fn sanitize_uniforms(&mut self) {
        self.uniforms.spin = self.uniforms.spin.clamp(-0.999, 0.999);
        self.uniforms.inclination_rad = self.uniforms.inclination_rad.clamp(0.01, PI - 0.01);
        self.uniforms.observer_r = self.uniforms.observer_r.clamp(2.0, 500.0);
        self.uniforms.fov_y_rad = self.uniforms.fov_y_rad.clamp(15.0f32.to_radians(), 120.0f32.to_radians());
        self.uniforms.step_size = self.uniforms.step_size.clamp(0.003, 0.1);
        self.uniforms.max_steps = self.uniforms.max_steps.clamp(200, 12_000);
        self.uniforms.spp = self.uniforms.spp.clamp(1, 32);
        self.uniforms.disk_inner = self.uniforms.disk_inner.clamp(1.0, 50.0);
        self.uniforms.disk_outer = self.uniforms.disk_outer.clamp(self.uniforms.disk_inner + 0.2, 150.0);
        self.uniforms.emissivity_power = self.uniforms.emissivity_power.clamp(1.0, 5.0);
        self.uniforms.exposure = self.uniforms.exposure.clamp(0.2, 3.0);
        self.uniforms.glow_strength = self.uniforms.glow_strength.clamp(0.0, 2.0);
    }

    fn draw_controls_ui(
        ctx: &egui::Context,
        uniforms: &mut RealtimeUniforms,
        paused: &mut bool,
        reset_clicked: &mut bool,
        export_clicked: &mut bool,
        export_frames: &mut u32,
        export_prefix: &mut String,
        fps_value: f32,
        frame_index: u32,
    ) {
        egui::TopBottomPanel::top("stats").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Gargantua Live").strong().color(Color32::from_rgb(255, 213, 124)));
                ui.separator();
                ui.label(format!("{fps_value:.1} FPS"));
                ui.separator();
                ui.label(format!("Accumulated frames: {frame_index}"));
                if *paused {
                    ui.separator();
                    ui.label(RichText::new("paused").color(Color32::from_rgb(255, 180, 70)));
                }
            });
        });

        egui::SidePanel::left("controls")
            .resizable(false)
            .default_width(300.0)
            .show(ctx, |ui| {
                ui.heading("Realtime Controls");
                ui.label("Physical and render controls update instantly.");
                ui.separator();

                ui.add(egui::Slider::new(&mut uniforms.spin, -0.998..=0.998).text("Spin a"));

                let mut inc_deg = uniforms.inclination_rad.to_degrees();
                if ui
                    .add(egui::Slider::new(&mut inc_deg, 1.0..=179.0).text("Inclination (deg)"))
                    .changed()
                {
                    uniforms.inclination_rad = inc_deg.to_radians();
                }

                ui.add(egui::Slider::new(&mut uniforms.observer_r, 6.0..=180.0).text("Observer r/M"));
                let mut fov = uniforms.fov_y_rad.to_degrees();
                if ui
                    .add(egui::Slider::new(&mut fov, 20.0..=110.0).text("FOV Y (deg)"))
                    .changed()
                {
                    uniforms.fov_y_rad = fov.to_radians();
                }

                ui.separator();
                ui.add(egui::Slider::new(&mut uniforms.disk_inner, 1.1..=30.0).text("Disk inner"));
                ui.add(egui::Slider::new(&mut uniforms.disk_outer, 2.0..=80.0).text("Disk outer"));
                ui.add(egui::Slider::new(&mut uniforms.emissivity_power, 1.0..=4.5).text("Emissivity p"));

                ui.separator();
                ui.add(egui::Slider::new(&mut uniforms.max_steps, 400..=8000).text("Max geodesic steps"));
                ui.add(egui::Slider::new(&mut uniforms.step_size, 0.005..=0.08).text("Step size"));
                ui.add(egui::Slider::new(&mut uniforms.spp, 1..=16).text("Samples / pixel / frame"));
                ui.add(egui::Slider::new(&mut uniforms.exposure, 0.4..=2.5).text("Exposure"));
                ui.add(egui::Slider::new(&mut uniforms.glow_strength, 0.0..=1.3).text("Glow"));

                ui.separator();
                if ui.button(if *paused { "Resume accumulation" } else { "Pause accumulation" }).clicked() {
                    *paused = !*paused;
                }
                if ui.button("Reset accumulation").clicked() {
                    *reset_clicked = true;
                }

                ui.separator();
                ui.label("Camera: RMB drag to orbit, wheel to zoom, WASD to adjust.");
                ui.add(egui::Slider::new(export_frames, 30..=1200).text("Export frames"));
                ui.text_edit_singleline(export_prefix);
                if ui.button("Export PNG sequence").clicked() {
                    *export_clicked = true;
                }
            });
    }

    fn update_fps(&mut self) {
        self.fps_frames += 1;
        let elapsed = self.fps_window_start.elapsed();
        if elapsed >= Duration::from_millis(300) {
            self.fps_value = self.fps_frames as f32 / elapsed.as_secs_f32();
            self.fps_frames = 0;
            self.fps_window_start = Instant::now();
        }
        self.frame_timer = Instant::now();
    }

    fn export_sequence(&mut self, frames: u32, prefix: &str) -> Result<()> {
        if frames == 0 {
            return Ok(());
        }

        let prefix_path = Path::new(prefix);
        if let Some(parent) = prefix_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create export dir {}", parent.display()))?;
            }
        }

        let start_yaw = self.uniforms.camera_yaw;
        let step = 2.0 * PI / frames as f32;
        let was_paused = self.paused;
        self.paused = true;

        for i in 0..frames {
            self.uniforms.camera_yaw = start_yaw + step * i as f32;
            self.uniforms.width = self.config.width;
            self.uniforms.height = self.config.height;
            self.uniforms.frame_index = self.frame_index;
            let (jx, jy) = halton_2d(self.frame_index + 1);
            self.uniforms.jitter = [jx - 0.5, jy - 0.5];
            self.queue
                .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms.as_gpu()));

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("export-encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("export-compute"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.compute_pipeline);
                let bg = if self.ping { &self.compute_bg_ba } else { &self.compute_bg_ab };
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(self.config.width.div_ceil(8), self.config.height.div_ceil(8), 1);
            }
            self.queue.submit(Some(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);

            let pixels = self.read_display_texture()?;
            let img = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(self.config.width, self.config.height, pixels)
                .context("failed to build export image")?;
            let path = format!("{prefix}_{i:05}.png");
            img.save(&path)
                .with_context(|| format!("failed to save export frame {path}"))?;

            self.frame_index = self.frame_index.saturating_add(1);
            self.ping = !self.ping;
        }

        self.uniforms.camera_yaw = start_yaw;
        self.paused = was_paused;
        self.needs_reset = true;
        Ok(())
    }

    fn read_display_texture(&self) -> Result<Vec<u8>> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let unpadded = width * 4;
        let padded = align_to(unpadded, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
        let size = (padded * height) as u64;

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("display-readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("display-readback-encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.display_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded as u32),
                    rows_per_image: Some(self.config.height),
                },
            },
            wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().context("failed waiting for display readback")??;

        let mapped = slice.get_mapped_range();
        let mut out = vec![0u8; width * height * 4];
        for row in 0..height {
            let src_start = row * padded;
            let src_end = src_start + unpadded;
            let dst_start = row * unpadded;
            let dst_end = dst_start + unpadded;
            out[dst_start..dst_end].copy_from_slice(&mapped[src_start..src_end]);
        }
        drop(mapped);
        readback.unmap();
        Ok(out)
    }
}

fn align_to(v: usize, align: usize) -> usize {
    ((v + align - 1) / align) * align
}

fn create_realtime_textures(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::Texture, wgpu::Texture, wgpu::TextureView) {
    let accum_desc = wgpu::TextureDescriptor {
        label: Some("accum-tex"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    };

    let display_desc = wgpu::TextureDescriptor {
        label: Some("display-tex"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };

    let a = device.create_texture(&accum_desc);
    let b = device.create_texture(&accum_desc);
    let display = device.create_texture(&display_desc);
    let view = display.create_view(&Default::default());
    (a, b, display, view)
}

fn pixel_dir_camera(
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    fov_y: f32,
    jx: f32,
    jy: f32,
) -> Vec3 {
    let aspect = width as f32 / height as f32;
    let tan_half = (0.5 * fov_y).tan();

    let nx = ((x as f32 + 0.5 + jx) / width as f32) * 2.0 - 1.0;
    let ny = 1.0 - ((y as f32 + 0.5 + jy) / height as f32) * 2.0;

    Vec3 {
        x: nx * tan_half * aspect,
        y: ny * tan_half,
        z: 1.0,
    }
    .normalize()
}

fn trace_ray(dir_cam: Vec3, phys: Physics) -> Vec3 {
    let Some(init) = tetrad_init(dir_cam, phys.spin, phys.observer_r, phys.inclination_rad) else {
        return background(dir_cam);
    };

    let horizon = 1.0 + (1.0 - phys.spin * phys.spin).sqrt();

    let mut r = phys.observer_r;
    let mut theta = phys.inclination_rad;
    let mut phi = 0.0f32;
    let mut sigma_r = if init.k_r >= 0.0 { 1.0 } else { -1.0 };
    let mut sigma_theta = if init.k_theta >= 0.0 { 1.0 } else { -1.0 };

    let mut prev_theta = theta;
    let mut prev_r = r;

    for _ in 0..phys.max_steps {
        let sin_t = theta.sin().abs().max(1e-4);
        let cos_t = theta.cos();
        let sin2 = sin_t * sin_t;
        let cos2 = cos_t * cos_t;

        let a = phys.spin;
        let delta = r * r - 2.0 * r + a * a;
        let sigma = r * r + a * a * cos2;

        let p = (r * r + a * a) - a * init.lambda;
        let mut r_pot = p * p - delta * (init.eta + (init.lambda - a) * (init.lambda - a));
        let mut t_pot = init.eta + a * a * cos2 - (init.lambda * init.lambda * cos2) / sin2;

        if r_pot < 0.0 {
            r_pot = 0.0;
            sigma_r = -sigma_r;
        }
        if t_pot < 0.0 {
            t_pot = 0.0;
            sigma_theta = -sigma_theta;
        }

        let dr = sigma_r * r_pot.sqrt() / sigma;
        let dtheta = sigma_theta * t_pot.sqrt() / sigma;
        let dphi = (init.lambda / sin2 - a + a * p / delta.max(1e-5)) / sigma;

        r += phys.step_size * dr;
        theta += phys.step_size * dtheta;
        phi += phys.step_size * dphi;

        if r <= horizon {
            return Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            };
        }

        let centered_prev = prev_theta - PI * 0.5;
        let centered_curr = theta - PI * 0.5;

        if centered_prev * centered_curr <= 0.0 {
            let mix = (centered_prev.abs() / (centered_prev.abs() + centered_curr.abs() + 1e-6)).clamp(0.0, 1.0);
            let r_hit = prev_r * (1.0 - mix) + r * mix;

            if r_hit >= phys.disk_inner && r_hit <= phys.disk_outer {
                return disk_radiance(r_hit, wrap_angle(phi), init.lambda, phys);
            }
        }

        if r > phys.observer_r * 1.2 {
            return background(dir_cam);
        }

        prev_r = r;
        prev_theta = theta;
    }

    background(dir_cam)
}

#[derive(Clone, Copy)]
struct RayInit {
    lambda: f32,
    eta: f32,
    k_r: f32,
    k_theta: f32,
}

fn tetrad_init(dir_cam: Vec3, a: f32, r: f32, theta: f32) -> Option<RayInit> {
    let sin_t = theta.sin().abs().max(1e-5);
    let cos_t = theta.cos();
    let sin2 = sin_t * sin_t;
    let cos2 = cos_t * cos_t;

    let sigma = r * r + a * a * cos2;
    let delta = r * r - 2.0 * r + a * a;
    if delta <= 0.0 {
        return None;
    }

    let a_big = (r * r + a * a) * (r * r + a * a) - a * a * delta * sin2;
    let lapse = (sigma * delta / a_big).sqrt();
    let omega = 2.0 * a * r / a_big;

    let e_t_t = 1.0 / lapse;
    let e_t_phi = omega / lapse;
    let e_r_r = (delta / sigma).sqrt();
    let e_th_th = 1.0 / sigma.sqrt();
    let e_ph_phi = (sigma / a_big).sqrt() / sin_t;

    let n_r = -dir_cam.z;
    let n_th = dir_cam.y;
    let n_ph = dir_cam.x;

    let k_t = e_t_t;
    let k_r = n_r * e_r_r;
    let k_theta = n_th * e_th_th;
    let k_phi = e_t_phi + n_ph * e_ph_phi;

    let g_tt = -(1.0 - 2.0 * r / sigma);
    let g_tphi = -2.0 * a * r * sin2 / sigma;
    let g_phiphi = (r * r + a * a + 2.0 * a * a * r * sin2 / sigma) * sin2;

    let p_t = g_tt * k_t + g_tphi * k_phi;
    let p_phi = g_tphi * k_t + g_phiphi * k_phi;
    let p_theta = sigma * k_theta;

    let e = -p_t;
    if e <= 1e-6 {
        return None;
    }

    let lz = p_phi;
    let lambda = lz / e;
    let q = p_theta * p_theta + cos2 * ((lz * lz) / sin2 - a * a * e * e);
    let eta = (q / (e * e)).max(0.0);

    Some(RayInit {
        lambda,
        eta,
        k_r,
        k_theta,
    })
}

fn disk_radiance(r: f32, phi: f32, lambda: f32, phys: Physics) -> Vec3 {
    let a = phys.spin;
    let omega = 1.0 / (r.powf(1.5) + a);
    let g_tt = -(1.0 - 2.0 / r);
    let g_tphi = -2.0 * a / r;
    let g_phiphi = r * r + a * a + 2.0 * a * a / r;

    let denom = -(g_tt + 2.0 * omega * g_tphi + omega * omega * g_phiphi).max(-1e-5);
    let u_t = 1.0 / denom.sqrt();
    let doppler = (1.0 - omega * lambda).abs().max(1e-3);
    let g = (1.0 / (u_t * doppler)).clamp(0.05, 3.0);

    let radial = (r / phys.disk_inner).powf(-phys.emissivity_power);
    let fade = ((phys.disk_outer - r) / (phys.disk_outer - phys.disk_inner)).clamp(0.0, 1.0);
    let az = 0.8 + 0.2 * (6.0 * phi).cos();

    let intensity = radial * fade * az * g.powf(3.0);
    let base = thermal_color((1.0 / r.sqrt()).clamp(0.0, 1.0));

    Vec3 {
        x: tone_map(base.x * intensity),
        y: tone_map(base.y * intensity),
        z: tone_map(base.z * intensity),
    }
}

fn background(dir: Vec3) -> Vec3 {
    let d = (dir.x * dir.x + dir.y * dir.y).sqrt();
    let t = (0.65 - 0.55 * d).clamp(0.0, 1.0);
    let vignette = (1.0 - 0.7 * d).clamp(0.0, 1.0);

    Vec3 {
        x: tone_map((0.04 + 0.20 * t) * vignette),
        y: tone_map((0.05 + 0.24 * t) * vignette),
        z: tone_map((0.08 + 0.32 * t) * vignette),
    }
}

fn thermal_color(x: f32) -> Vec3 {
    let x = x.clamp(0.0, 1.0);
    let warm = Vec3 {
        x: 1.35,
        y: 0.90,
        z: 0.45,
    };
    let hot = Vec3 {
        x: 0.90,
        y: 1.00,
        z: 1.20,
    };
    warm.mul(1.0 - x).add(hot.mul(x))
}

fn tone_map(v: f32) -> f32 {
    let mapped = v / (1.0 + v.abs());
    mapped.powf(1.0 / 2.2).clamp(0.0, 1.0)
}

fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn wrap_angle(phi: f32) -> f32 {
    let two_pi = 2.0 * PI;
    let mut x = phi % two_pi;
    if x < 0.0 {
        x += two_pi;
    }
    x
}

fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

fn sample_jitter_u32(x: u32, y: u32, s: u32, frame: u32) -> (f32, f32) {
    let h1 = hash_u32(x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663) ^ s ^ frame);
    let h2 = hash_u32(h1 ^ 0x9e37_79b9);
    let jx = ((h1 as f32) / (u32::MAX as f32)) - 0.5;
    let jy = ((h2 as f32) / (u32::MAX as f32)) - 0.5;
    (jx, jy)
}

fn halton(mut i: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut r = 0.0;
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

fn halton_2d(i: u32) -> (f32, f32) {
    (halton(i, 2), halton(i, 3))
}
