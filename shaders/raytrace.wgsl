struct Uniforms {
  width: u32,
  height: u32,
  max_steps: u32,
  spp: u32,
  seed: u32,
  _pad0: vec4<u32>,
  fov_y_rad: f32,
  spin: f32,
  observer_r: f32,
  inclination_rad: f32,
  step_size: f32,
  disk_inner: f32,
  disk_outer: f32,
  emissivity_power: f32,
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
var<storage, read_write> out_pixels: array<u32>;

fn tone_map(v: f32) -> f32 {
  let x = max(v, 0.0);
  let mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
  return clamp(mapped, 0.0, 1.0);
}

fn thermal_color(x_in: f32) -> vec3<f32> {
  let x = clamp(x_in, 0.0, 1.0);
  let warm = vec3<f32>(1.35, 0.90, 0.45);
  let hot = vec3<f32>(0.90, 1.00, 1.20);
  return warm * (1.0 - x) + hot * x;
}

fn background(dir: vec3<f32>) -> vec3<f32> {
  let d = length(dir.xy);
  let t = clamp(0.9 - 0.95 * d, 0.0, 1.0);
  let vignette = clamp(1.0 - 0.85 * d, 0.0, 1.0);
  return vec3<f32>(
    (0.004 + 0.014 * t) * vignette,
    (0.006 + 0.020 * t) * vignette,
    (0.011 + 0.028 * t) * vignette
  );
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
  let a = U.spin;
  let z1 = 1.0 + pow(1.0 - a * a, 1.0 / 3.0) * (pow(1.0 + a, 1.0 / 3.0) + pow(1.0 - a, 1.0 / 3.0));
  let z2 = sqrt(3.0 * a * a + z1 * z1);
  let risco = max(1.0, 3.0 + z2 - sqrt(max((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 0.0)));
  let r_in = max(U.disk_inner, risco);
  let omega = 1.0 / (pow(r, 1.5) + a);
  let g_tt = -(1.0 - 2.0 / r);
  let g_tphi = -2.0 * a / r;
  let g_phiphi = r * r + a * a + 2.0 * a * a / r;

  let denom = max(-(g_tt + 2.0 * omega * g_tphi + omega * omega * g_phiphi), 1e-5);
  let u_t = 1.0 / sqrt(denom);
  let doppler = max(abs(1.0 - omega * lambda), 1e-3);
  let g = clamp(1.0 / (u_t * doppler), 0.03, 8.0);

  let radial = pow(r / r_in, -U.emissivity_power);
  let nt = clamp(1.0 - sqrt(r_in / max(r, r_in + 1e-4)), 0.0, 1.0);
  let fade = clamp((U.disk_outer - r) / (U.disk_outer - r_in), 0.0, 1.0);
  let az = 0.75 + 0.25 * cos(6.0 * phi);
  let beaming = clamp(1.0 + 1.35 * omega * lambda, 0.2, 4.5);
  let intensity = radial * nt * fade * az * beaming * pow(g, 3.6);

  let base = thermal_color(clamp(1.35 / sqrt(r), 0.0, 1.0));
  let white_hot = vec3<f32>(1.15, 1.08, 1.0);
  let hot_mix = clamp(pow(g, 0.8) * 0.45, 0.0, 0.65);
  let color = base * (1.0 - hot_mix) + white_hot * hot_mix;
  return vec3<f32>(
    tone_map(color.x * intensity * 2.6),
    tone_map(color.y * intensity * 2.15),
    tone_map(color.z * intensity * 1.75)
  );
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
  let a = U.spin;
  let r = U.observer_r;
  let theta = U.inclination_rad;

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

fn trace_ray(dir: vec3<f32>) -> vec3<f32> {
  let init = tetrad_init(dir);
  let a = U.spin;
  let horizon = 1.0 + sqrt(max(1.0 - a * a, 1e-7));

  var r = U.observer_r;
  var theta = U.inclination_rad;
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
    if (step >= U.max_steps) {
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

    r = r + U.step_size * dr;
    theta = theta + U.step_size * dtheta;
    phi = phi + U.step_size * dphi;

    if (r <= horizon) {
      return accum;
    }

    let centered_prev = prev_theta - 1.57079632679;
    let centered_curr = theta - 1.57079632679;
    if (centered_prev * centered_curr <= 0.0) {
      let mix = clamp(abs(centered_prev) / (abs(centered_prev) + abs(centered_curr) + 1e-6), 0.0, 1.0);
      let r_hit = prev_r * (1.0 - mix) + r * mix;
      if (r_hit >= U.disk_inner && r_hit <= U.disk_outer) {
        let emission = disk_radiance(r_hit, wrap_angle(phi), init.lambda);
        accum = accum + trans * emission;
        trans = trans * 0.45;
        hits = hits + 1u;
        if (hits >= 4u || trans < 0.03) {
          return accum;
        }
      }
    }

    if (r > U.observer_r * 1.2) {
      return accum + trans * background(dir);
    }

    prev_theta = theta;
    prev_r = r;
    step = step + 1u;
  }

  return accum + trans * background(dir);
}

fn pack_rgba8(color: vec3<f32>) -> u32 {
  let r = u32(clamp(color.x * 255.0 + 0.5, 0.0, 255.0));
  let g = u32(clamp(color.y * 255.0 + 0.5, 0.0, 255.0));
  let b = u32(clamp(color.z * 255.0 + 0.5, 0.0, 255.0));
  return (255u << 24u) | (r << 16u) | (g << 8u) | b;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= U.width || gid.y >= U.height) {
    return;
  }

  let aspect = f32(U.width) / f32(U.height);
  let tan_half = tan(0.5 * U.fov_y_rad);

  let idx = gid.y * U.width + gid.x;
  var accum = vec3<f32>(0.0, 0.0, 0.0);

  for (var s: u32 = 0u; s < U.spp; s = s + 1u) {
    let h1 = hash_u32(gid.x * 73856093u ^ gid.y * 19349663u ^ s ^ U.seed);
    let h2 = hash_u32(h1 ^ 0x9e3779b9u);

    let jx = rand01(h1) - 0.5;
    let jy = rand01(h2) - 0.5;

    let nx = ((f32(gid.x) + 0.5 + jx) / f32(U.width)) * 2.0 - 1.0;
    let ny = 1.0 - ((f32(gid.y) + 0.5 + jy) / f32(U.height)) * 2.0;

    let dir = normalize(vec3<f32>(nx * tan_half * aspect, ny * tan_half, 1.0));
    accum = accum + trace_ray(dir);
  }

  var color = accum / f32(U.spp);
  let alpha = (((f32(gid.x) + 0.5) / f32(U.width)) * 2.0 - 1.0) * tan_half * aspect * U.observer_r;
  let beta = (1.0 - ((f32(gid.y) + 0.5) / f32(U.height)) * 2.0) * tan_half * U.observer_r;
  let b = sqrt(alpha * alpha + beta * beta);
  let ring = exp(-pow((b - 5.2) / 0.42, 2.0));
  color = color + vec3<f32>(1.15, 0.94, 0.78) * ring * 0.22;

  let ny = 1.0 - ((f32(gid.y) + 0.5) / f32(U.height)) * 2.0;
  let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
  let streak = pow(clamp(1.0 - abs(ny) * 28.0, 0.0, 1.0), 2.2) * smoothstep(0.22, 1.0, luma);
  color = color + vec3<f32>(1.0, 0.72, 0.48) * streak * 0.42;

  out_pixels[idx] = pack_rgba8(color);
}
