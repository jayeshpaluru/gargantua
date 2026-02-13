# Gargantua

Realtime Kerr black hole renderer in Rust with a GPU compute path (`wgpu`) and a CPU fallback.

The project targets an Interstellar-inspired look while keeping the core transport model physically grounded:
- Kerr null geodesic tracing with conserved quantities
- Observer tetrad ray initialization
- Thin relativistic accretion disk with redshift/beaming terms
- Temporal accumulation for realtime viewing

## Reference Target

The included `gargantua.jpeg` is the visual reference this repo is tuned toward.

## Requirements

- Rust toolchain (`rustup`, `cargo`)
- macOS + Apple Silicon GPU recommended for realtime (`Metal`)

## Build

```bash
cargo check
cargo run --release -- --help
```

## Realtime Viewer

```bash
cargo run --release -- \
  --realtime --backend gpu \
  --width 1600 --height 900 \
  --spp 2 --max-steps 3500 --step-size 0.025 \
  --spin 0.92 --inclination-deg 63 --observer-r 75
```

### Realtime Controls

- `RMB drag`: orbit camera
- `Mouse wheel`: zoom (`FOV`)
- `W/S`: move observer radius in/out
- `A/D`: adjust inclination
- `Space`: pause/resume accumulation
- `R`: reset accumulation

UI sliders also expose:
- black hole spin, disk radii, emissivity
- geodesic step controls
- exposure + glow
- export frame sequence options

## Offline Render (PNG)

```bash
cargo run --release -- \
  --backend gpu \
  --width 1920 --height 1080 \
  --spp 16 \
  --spin 0.92 --inclination-deg 63 --observer-r 75 \
  --max-steps 4000 --step-size 0.025 \
  --output gargantua.png
```

If GPU initialization fails, the binary falls back to CPU and prints a warning.

## Scientific Notes

- This is a physically motivated renderer, not full GRMHD + radiative transfer.
- The disk model is thin and parametric.
- Visual tuning includes cinematic post terms (e.g., glow/streak behavior) to approach the film look.

## Repository Layout

- `src/main.rs`: app, CPU tracer, realtime window/UI, export
- `shaders/realtime.wgsl`: realtime compute + fullscreen display shader
- `shaders/raytrace.wgsl`: offline GPU compute shader
- `gargantua.jpeg`: visual reference target
