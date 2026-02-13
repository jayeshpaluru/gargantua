struct Uniforms {
  dims: vec4<u32>,      // width, height, max_steps, spp
  frame: vec4<u32>,     // frame_index + padding
  jitter: vec4<f32>,    // jitter.xy + padding
  camera0: vec4<f32>,   // fov, spin, observer_r, inclination
  camera1: vec4<f32>,   // step_size, disk_inner, disk_outer, emissivity_power
  camera2: vec4<f32>,   // exposure, glow_strength, camera_yaw, camera_pitch
}

struct RayInit {
  lambda: f32,
  eta: f32,
  k_r: f32,
  k_theta: f32,
}

@group(0) @binding(0)
var<uniform> U: Uniforms;
@group(0) @binding(1)
var accum_in: texture_2d<f32>;
@group(0) @binding(2)
var accum_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3)
var display_out: texture_storage_2d<rgba8unorm, write>;

@group(1) @binding(0)
var display_tex: texture_2d<f32>;
@group(1) @binding(1)
var display_sampler: sampler;

fn U_width() -> u32 { return U.dims.x; }
fn U_height() -> u32 { return U.dims.y; }
fn U_max_steps() -> u32 { return U.dims.z; }
fn U_spp() -> u32 { return U.dims.w; }
fn U_frame_index() -> u32 { return U.frame.x; }
fn U_checkerboard() -> u32 { return U.frame.y; }
fn U_fov_y_rad() -> f32 { return U.camera0.x; }
fn U_spin() -> f32 { return U.camera0.y; }
fn U_observer_r() -> f32 { return U.camera0.z; }
fn U_inclination_rad() -> f32 { return U.camera0.w; }
fn U_step_size() -> f32 { return U.camera1.x; }
fn U_disk_inner() -> f32 { return U.camera1.y; }
fn U_disk_outer() -> f32 { return U.camera1.z; }
fn U_emissivity_power() -> f32 { return U.camera1.w; }
fn U_exposure() -> f32 { return U.camera2.x; }
fn U_glow_strength() -> f32 { return U.camera2.y; }
fn U_camera_yaw() -> f32 { return U.camera2.z; }
fn U_camera_pitch() -> f32 { return U.camera2.w; }
fn U_temporal_alpha() -> f32 { return U.jitter.z; }

fn tone_map(v: f32) -> f32 {
  let x = max(v * U_exposure(), 0.0);
  let mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
  return clamp(mapped, 0.0, 1.0);
}

fn thermal_color(x_in: f32) -> vec3<f32> {
  let x = clamp(x_in, 0.0, 1.0);
  let warm = vec3<f32>(1.35, 0.90, 0.45);
  let hot = vec3<f32>(0.90, 1.00, 1.20);
  return warm * (1.0 - x) + hot * x;
}

fn starfield(dir: vec3<f32>) -> vec3<f32> {
  let p = floor((normalize(dir) + vec3<f32>(1.0)) * 300.0);
  let h = fract(sin(dot(p, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453);
  let s = step(0.9975, h) * (0.4 + 0.6 * fract(h * 97.0));
  return vec3<f32>(0.6, 0.75, 1.0) * s;
}

fn background(dir: vec3<f32>) -> vec3<f32> {
  let d = length(dir.xy);
  let t = clamp(0.9 - 0.95 * d, 0.0, 1.0);
  let vignette = clamp(1.0 - 0.85 * d, 0.0, 1.0);
  let nebula = vec3<f32>(
    (0.004 + 0.014 * t) * vignette,
    (0.006 + 0.020 * t) * vignette,
    (0.011 + 0.028 * t) * vignette
  );
  return nebula + starfield(dir) * 0.75;
}

fn wrap_angle(phi: f32) -> f32 {
  let two_pi = 6.28318530718;
  var x = phi % two_pi;
  if (x < 0.0) {
    x = x + two_pi;
  }
  return x;
}

fn disk_radiance(r: f32, phi: f32, lambda: f32) -> vec3<f32> {
  let a = U_spin();
  let z1 = 1.0 + pow(1.0 - a * a, 1.0 / 3.0) * (pow(1.0 + a, 1.0 / 3.0) + pow(1.0 - a, 1.0 / 3.0));
  let z2 = sqrt(3.0 * a * a + z1 * z1);
  let risco = max(1.0, 3.0 + z2 - sqrt(max((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 0.0)));
  let r_in = max(U_disk_inner(), risco);
  let omega = 1.0 / (pow(r, 1.5) + a);
  let g_tt = -(1.0 - 2.0 / r);
  let g_tphi = -2.0 * a / r;
  let g_phiphi = r * r + a * a + 2.0 * a * a / r;

  let denom = max(-(g_tt + 2.0 * omega * g_tphi + omega * omega * g_phiphi), 1e-5);
  let u_t = 1.0 / sqrt(denom);
  let doppler = max(abs(1.0 - omega * lambda), 1e-3);
  let g = clamp(1.0 / (u_t * doppler), 0.03, 8.0);

  let radial = pow(r / r_in, -U_emissivity_power());
  let nt = clamp(1.0 - sqrt(r_in / max(r, r_in + 1e-4)), 0.0, 1.0);
  let fade = clamp((U_disk_outer() - r) / (U_disk_outer() - r_in), 0.0, 1.0);
  let az = 0.75 + 0.25 * cos(6.0 * phi);
  let beaming = clamp(1.0 + 1.35 * omega * lambda, 0.2, 4.5);
  let intensity = radial * nt * fade * az * beaming * pow(g, 3.6);

  let base = thermal_color(clamp(1.35 / sqrt(r), 0.0, 1.0));
  let white_hot = vec3<f32>(1.15, 1.08, 1.0);
  let hot_mix = clamp(pow(g, 0.8) * 0.45, 0.0, 0.65);
  let color = base * (1.0 - hot_mix) + white_hot * hot_mix;
  return vec3<f32>(color.x * intensity * 2.6, color.y * intensity * 2.15, color.z * intensity * 1.75);
}

fn hash_u32(x_in: u32) -> u32 {
  var x = x_in;
  x = x ^ (x >> 16u);
  x = x * 0x7feb352du;
  x = x ^ (x >> 15u);
  x = x * 0x846ca68bu;
  x = x ^ (x >> 16u);
  return x;
}

fn rand01(seed: u32) -> f32 {
  return f32(seed) / 4294967295.0;
}

fn tetrad_init(dir: vec3<f32>) -> RayInit {
  let a = U_spin();
  let r = U_observer_r();
  let theta = U_inclination_rad();

  let sin_t = max(abs(sin(theta)), 1e-5);
  let cos_t = cos(theta);
  let sin2 = sin_t * sin_t;
  let cos2 = cos_t * cos_t;

  let sigma = r * r + a * a * cos2;
  let delta = r * r - 2.0 * r + a * a;
  let big_a = (r * r + a * a) * (r * r + a * a) - a * a * delta * sin2;
  let lapse = sqrt(max(sigma * delta / big_a, 1e-7));
  let omega = 2.0 * a * r / big_a;

  let e_t_t = 1.0 / lapse;
  let e_t_phi = omega / lapse;
  let e_r_r = sqrt(max(delta / sigma, 1e-7));
  let e_th_th = 1.0 / sqrt(sigma);
  let e_ph_phi = sqrt(max(sigma / big_a, 1e-7)) / sin_t;

  let n_r = -dir.z;
  let n_th = dir.y;
  let n_ph = dir.x;

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

  let e = max(-p_t, 1e-6);
  let lz = p_phi;
  let lambda = lz / e;
  let q = p_theta * p_theta + cos2 * ((lz * lz) / sin2 - a * a * e * e);
  let eta = max(q / (e * e), 0.0);

  return RayInit(lambda, eta, k_r, k_theta);
}

fn rotate_dir(dir: vec3<f32>) -> vec3<f32> {
  let cy = cos(U_camera_yaw());
  let sy = sin(U_camera_yaw());
  let cp = cos(U_camera_pitch());
  let sp = sin(U_camera_pitch());
  let yawed = vec3<f32>(cy * dir.x + sy * dir.z, dir.y, -sy * dir.x + cy * dir.z);
  return normalize(vec3<f32>(yawed.x, cp * yawed.y - sp * yawed.z, sp * yawed.y + cp * yawed.z));
}

fn trace_ray(dir: vec3<f32>) -> vec3<f32> {
  let d = rotate_dir(dir);
  let init = tetrad_init(d);
  let a = U_spin();
  let horizon = 1.0 + sqrt(max(1.0 - a * a, 1e-7));

  var r = U_observer_r();
  var theta = U_inclination_rad();
  var phi = 0.0;

  var sigma_r = select(-1.0, 1.0, init.k_r >= 0.0);
  var sigma_theta = select(-1.0, 1.0, init.k_theta >= 0.0);

  var prev_theta = theta;
  var prev_r = r;

  var accum = vec3<f32>(0.0);
  var trans = 1.0;
  var hits: u32 = 0u;
  var step: u32 = 0u;
  loop {
    if (step >= U_max_steps()) {
      break;
    }

    let sin_t = max(abs(sin(theta)), 1e-4);
    let cos_t = cos(theta);
    let sin2 = sin_t * sin_t;
    let cos2 = cos_t * cos_t;

    let delta = r * r - 2.0 * r + a * a;
    let sigma = r * r + a * a * cos2;

    let p = (r * r + a * a) - a * init.lambda;
    var r_pot = p * p - delta * (init.eta + (init.lambda - a) * (init.lambda - a));
    var t_pot = init.eta + a * a * cos2 - (init.lambda * init.lambda * cos2) / sin2;

    if (r_pot < 0.0) {
      r_pot = 0.0;
      sigma_r = -sigma_r;
    }
    if (t_pot < 0.0) {
      t_pot = 0.0;
      sigma_theta = -sigma_theta;
    }

    let dr = sigma_r * sqrt(r_pot) / sigma;
    let dtheta = sigma_theta * sqrt(t_pot) / sigma;
    let dphi = (init.lambda / sin2 - a + a * p / max(delta, 1e-5)) / sigma;

    r = r + U_step_size() * dr;
    theta = theta + U_step_size() * dtheta;
    phi = phi + U_step_size() * dphi;

    if (r <= horizon) {
      return accum;
    }

    let centered_prev = prev_theta - 1.57079632679;
    let centered_curr = theta - 1.57079632679;
    if (centered_prev * centered_curr <= 0.0) {
      let mix = clamp(abs(centered_prev) / (abs(centered_prev) + abs(centered_curr) + 1e-6), 0.0, 1.0);
      let r_hit = prev_r * (1.0 - mix) + r * mix;
      if (r_hit >= U_disk_inner() && r_hit <= U_disk_outer()) {
        let emission = disk_radiance(r_hit, wrap_angle(phi), init.lambda);
        accum = accum + trans * emission;
        trans = trans * 0.45;
        hits = hits + 1u;
        if (hits >= 4u || trans < 0.03) {
          return accum;
        }
      }
    }

    if (r > U_observer_r() * 1.2) {
      return accum + trans * background(d);
    }

    prev_theta = theta;
    prev_r = r;
    step = step + 1u;
  }

  return accum + trans * background(d);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= U_width() || gid.y >= U_height()) {
    return;
  }

  let aspect = f32(U_width()) / f32(U_height());
  let tan_half = tan(0.5 * U_fov_y_rad());
  let idx = gid.y * U_width() + gid.x;
  let prev = textureLoad(accum_in, vec2<i32>(gid.xy), 0).rgb;

  if (U_checkerboard() != 0u) {
    let parity = (gid.x + gid.y + U_frame_index()) & 1u;
    if (parity == 1u) {
      let glow_prev = max(prev - vec3<f32>(0.55), vec3<f32>(0.0)) * U_glow_strength();
      let final_prev = prev + glow_prev * glow_prev;
      textureStore(accum_out, vec2<i32>(gid.xy), vec4<f32>(prev, 1.0));
      textureStore(
        display_out,
        vec2<i32>(gid.xy),
        vec4<f32>(tone_map(final_prev.x), tone_map(final_prev.y), tone_map(final_prev.z), 1.0)
      );
      return;
    }
  }

  var frame_sum = vec3<f32>(0.0, 0.0, 0.0);
  for (var s: u32 = 0u; s < U_spp(); s = s + 1u) {
    let h1 = hash_u32(gid.x * 73856093u ^ gid.y * 19349663u ^ s ^ U_frame_index());
    let h2 = hash_u32(h1 ^ 0x9e3779b9u);

    let jx = (rand01(h1) - 0.5 + U.jitter.x) / f32(U_width());
    let jy = (rand01(h2) - 0.5 + U.jitter.y) / f32(U_height());

    let nx = ((f32(gid.x) + 0.5) / f32(U_width())) * 2.0 - 1.0 + 2.0 * jx;
    let ny = 1.0 - ((f32(gid.y) + 0.5) / f32(U_height())) * 2.0 + 2.0 * jy;
    let dir = normalize(vec3<f32>(nx * tan_half * aspect, ny * tan_half, 1.0));
    frame_sum = frame_sum + trace_ray(dir);
  }

  var sample = frame_sum / f32(U_spp());
  let alpha = (((f32(gid.x) + 0.5) / f32(U_width())) * 2.0 - 1.0) * tan_half * aspect * U_observer_r();
  let beta = (1.0 - ((f32(gid.y) + 0.5) / f32(U_height())) * 2.0) * tan_half * U_observer_r();
  let b = sqrt(alpha * alpha + beta * beta);
  let b_crit = 5.2;
  let ring = exp(-pow((b - b_crit) / 0.42, 2.0));
  sample = sample + vec3<f32>(1.15, 0.94, 0.78) * ring * 0.22;

  let ny = 1.0 - ((f32(gid.y) + 0.5) / f32(U_height())) * 2.0;
  let luma = dot(sample, vec3<f32>(0.2126, 0.7152, 0.0722));
  let streak = pow(clamp(1.0 - abs(ny) * 28.0, 0.0, 1.0), 2.2) * smoothstep(0.22, 1.0, luma);
  sample = sample + vec3<f32>(1.0, 0.72, 0.48) * streak * 0.42;

  let blend_alpha = clamp(U_temporal_alpha(), 0.03, 1.0);
  let accum = prev * (1.0 - blend_alpha) + sample * blend_alpha;

  let glow = max(accum - vec3<f32>(0.55), vec3<f32>(0.0)) * U_glow_strength();
  let final_rgb = accum + glow * glow;
  textureStore(accum_out, vec2<i32>(gid.xy), vec4<f32>(accum, 1.0));
  textureStore(display_out, vec2<i32>(gid.xy), vec4<f32>(tone_map(final_rgb.x), tone_map(final_rgb.y), tone_map(final_rgb.z), 1.0));

  _ = idx;
}

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  var out: VSOut;
  let x = f32((vi << 1u) & 2u);
  let y = f32(vi & 2u);
  out.uv = vec2<f32>(x, 1.0 - y);
  out.pos = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  return textureSample(display_tex, display_sampler, in.uv);
}
