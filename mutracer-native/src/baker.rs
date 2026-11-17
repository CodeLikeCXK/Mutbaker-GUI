use crate::bvh::{FlatBvh, GpuPointLight};
use crate::gpu::GpuContext;
use crate::ktx2_writer::{create_ktx2_3d_rgb9e5, pack_rgb9e5};
use crate::scene::EnvironmentMap;
use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use glam::{UVec3, Vec3};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BakeUniforms {
    pub bounds_min: [f32; 4],
    pub bounds_max: [f32; 4],
    pub resolution: [u32; 4], // rx, ry, rz, spp
    pub params: [u32; 4],     // max_bounces, rng_seed, total_voxels, point_light_count
    pub sun_dir: [f32; 4],    // xyz = sun direction, w = enabled flag (1.0 or 0.0)
    pub sun_color: [f32; 4],  // rgb = sun color, w = intensity
    pub env_params: [f32; 4], // x = enabled (1/0), y = intensity, z = rotation_y_rad, w = has_cdf (1/0)
    pub env_dims: [u32; 4],   // cdf_width, cdf_height, tex_width, tex_height
}

#[derive(Clone, Debug)]
pub struct BakeSettings {
    pub resolution: UVec3, // Rx, Ry, Rz
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub spp: u32,
    pub max_bounces: u32,
    pub sun_dir: Option<Vec3>,
    pub sun_color: Option<Vec3>,
    pub sun_intensity: Option<f32>,
    pub env_map: Option<Arc<EnvironmentMap>>,
    pub env_intensity: f32,
    pub env_rotation_y: f32,
}

pub struct BakeResult {
    pub resolution: UVec3,
    pub texture_dimensions: UVec3, // Rx, 2*Ry, 3*Rz
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub ktx2_bytes: Vec<u8>,
    pub raw_floats: Vec<[f32; 4]>,
    pub duration: Duration,
}

const BAKER_SHADER_SRC: &str = r#"
struct BakeUniforms {
    bounds_min: vec4f,
    bounds_max: vec4f,
    resolution: vec4u, // rx, ry, rz, spp
    params: vec4u,     // max_bounces, rng_seed, total_voxels, point_light_count
    sun_dir: vec4f,    // xyz = sun direction, w = enabled flag (1.0 or 0.0)
    sun_color: vec4f,  // rgb = sun color, w = intensity
    env_params: vec4f, // x = enabled (1/0), y = intensity, z = rotation_y_rad, w = has_cdf (1/0)
    env_dims: vec4u,   // cdf_width, cdf_height, tex_width, tex_height
};

struct GpuBvhNode {
    min: vec3f,
    left_or_first_tri: u32,
    max: vec3f,
    tri_count: u32,
};

struct GpuTriangle {
    p0: vec3f,
    material_idx: u32,
    p1: vec3f,
    flags: u32,
    p2: vec3f,
    pad0: u32,
    n0: vec3f,
    pad1: u32,
    n1: vec3f,
    pad2: u32,
    n2: vec3f,
    pad3: u32,
    uv0: vec2f,
    uv1: vec2f,
    uv2: vec2f,
    pad4: vec2f,
};

struct GpuMaterial {
    base_color: vec4f,
    emissive: vec4f,
    roughness_metallic: vec4f,
    tex_index: vec4i,
};

struct GpuPointLight {
    position_range: vec4f,  // xyz = world position, w = inverse_range_squared (0.0 if unbounded)
    color_intensity: vec4f, // rgb = linear color, w = intensity (candela)
};

@group(0) @binding(0) var<uniform> u: BakeUniforms;
@group(0) @binding(1) var<storage, read> bvh_nodes: array<GpuBvhNode>;
@group(0) @binding(2) var<storage, read> triangles: array<GpuTriangle>;
@group(0) @binding(3) var<storage, read> materials: array<GpuMaterial>;
@group(0) @binding(4) var<storage, read_write> out_irradiance: array<vec4f>;
@group(0) @binding(5) var tex_array: texture_2d_array<f32>;
@group(0) @binding(6) var tex_sampler: sampler;
@group(0) @binding(7) var<storage, read> point_lights: array<GpuPointLight>;
@group(0) @binding(8) var env_tex: texture_2d<f32>;
@group(0) @binding(9) var env_sampler: sampler;
@group(0) @binding(10) var<storage, read> env_cdf: array<f32>;

const PI: f32 = 3.14159265359;
const INV_PI: f32 = 0.31830988618;
const TWO_PI: f32 = 6.28318530718;

fn rotate_y(v: vec3f, angle: f32) -> vec3f {
    if (abs(angle) < 1e-6) { return v; }
    let s = sin(angle);
    let c = cos(angle);
    return vec3f(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}

fn sample_hdri(world_dir: vec3f) -> vec3f {
    let d = normalize(rotate_y(world_dir, -u.env_params.z));
    let u_coord = atan2(d.z, d.x) / TWO_PI + 0.5;
    let v_coord = acos(clamp(d.y, -1.0, 1.0)) * INV_PI;
    return textureSampleLevel(env_tex, env_sampler, vec2f(u_coord, v_coord), 0.0).rgb * u.env_params.y;
}

fn hdri_pdf(world_dir: vec3f) -> f32 {
    if (u.env_params.w < 0.5) {
        return 0.0;
    }
    let cw = u.env_dims.x;
    let ch = u.env_dims.y;
    if (cw == 0u || ch == 0u) { return 0.0; }
    let d = normalize(rotate_y(world_dir, -u.env_params.z));
    let u_coord = fract(atan2(d.z, d.x) / TWO_PI + 0.5);
    let v_coord = clamp(acos(clamp(d.y, -1.0, 1.0)) * INV_PI, 0.0, 0.999999);

    let x = min(u32(floor(u_coord * f32(cw))), cw - 1u);
    let y = min(u32(floor(v_coord * f32(ch))), ch - 1u);

    let p_row = max(0.0, env_cdf[y + 1u] - env_cdf[y]);
    let cond_base = (ch + 1u) + y * (cw + 1u);
    let p_col = max(0.0, env_cdf[cond_base + x + 1u] - env_cdf[cond_base + x]);
    let sin_theta = max(sqrt(max(1.0 - d.y * d.y, 0.0)), 1e-5);
    return (p_row * p_col * f32(cw * ch)) / (2.0 * PI * PI * sin_theta);
}

fn sample_hdri_importance(u1: f32, u2: f32) -> vec4f {
    let cw = u.env_dims.x;
    let ch = u.env_dims.y;
    var lo = 0u;
    var hi = ch;
    while (lo + 1u < hi) {
        let mid = (lo + hi) >> 1u;
        if (env_cdf[mid] <= u1) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let y = lo;
    let m0 = env_cdf[y];
    let m1 = env_cdf[y + 1u];
    let p_row = max(0.0, m1 - m0);
    let dy = clamp((u1 - m0) / max(p_row, 1e-8), 0.0, 0.999999);
    let v_coord = (f32(y) + dy) / f32(ch);

    let cond_base = (ch + 1u) + y * (cw + 1u);
    var clo = 0u;
    var chi = cw;
    while (clo + 1u < chi) {
        let mid = (clo + chi) >> 1u;
        if (env_cdf[cond_base + mid] <= u2) {
            clo = mid;
        } else {
            chi = mid;
        }
    }
    let x = clo;
    let c0 = env_cdf[cond_base + x];
    let c1 = env_cdf[cond_base + x + 1u];
    let p_col = max(0.0, c1 - c0);
    let dx = clamp((u2 - c0) / max(p_col, 1e-8), 0.0, 0.999999);
    let u_coord = (f32(x) + dx) / f32(cw);

    let phi = (u_coord - 0.5) * TWO_PI;
    let theta = v_coord * PI;
    let sin_theta = sin(theta);
    let env_dir = vec3f(sin_theta * cos(phi), cos(theta), sin_theta * sin(phi));
    let world_dir = normalize(rotate_y(env_dir, u.env_params.z));
    let pdf_omega = (p_row * p_col * f32(cw * ch)) / (2.0 * PI * PI * max(sin_theta, 1e-5));
    return vec4f(world_dir, pdf_omega);
}

struct HitRecord {
    t: f32,
    u: f32,
    v: f32,
    tri_idx: u32,
};

fn intersect_box(o: vec3f, inv_d: vec3f, bmin: vec3f, bmax: vec3f, t_max: f32) -> bool {
    let t0 = (bmin - o) * inv_d;
    let t1 = (bmax - o) * inv_d;
    let tnear = min(t0, t1);
    let tfar = max(t0, t1);
    let enter = max(max(tnear.x, tnear.y), tnear.z);
    let exit = min(min(tfar.x, tfar.y), tfar.z);
    return enter <= exit && exit > 0.001 && enter < t_max;
}

fn intersect_tri(o: vec3f, d: vec3f, tri: GpuTriangle, t_max: f32) -> vec4f {
    let e1 = tri.p1 - tri.p0;
    let e2 = tri.p2 - tri.p0;
    let h = cross(d, e2);
    let a = dot(e1, h);
    if (abs(a) < 1e-7) { return vec4f(0.0); }
    let f = 1.0 / a;
    let s = o - tri.p0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) { return vec4f(0.0); }
    let q = cross(s, e1);
    let v = f * dot(d, q);
    if (v < 0.0 || u + v > 1.0) { return vec4f(0.0); }
    let t = f * dot(e2, q);
    if (t > 0.002 && t < t_max) {
        return vec4f(t, u, v, 1.0);
    }
    return vec4f(0.0);
}

fn trace_ray(o: vec3f, d: vec3f, max_dist: f32) -> HitRecord {
    var hit: HitRecord;
    hit.t = max_dist;
    hit.tri_idx = 0xFFFFFFFFu;

    var inv_d = vec3f(1.0) / d;
    if (abs(d.x) < 1e-6) { inv_d.x = 1e6; }
    if (abs(d.y) < 1e-6) { inv_d.y = 1e6; }
    if (abs(d.z) < 1e-6) { inv_d.z = 1e6; }

    var stack: array<u32, 32>;
    var stack_top = 0;
    stack[0] = 0u;

    while (stack_top >= 0) {
        let node_idx = stack[stack_top];
        stack_top--;

        let node = bvh_nodes[node_idx];
        if (intersect_box(o, inv_d, node.min, node.max, hit.t)) {
            if (node.tri_count > 0u) {
                let first = node.left_or_first_tri;
                for (var i = 0u; i < node.tri_count; i++) {
                    let tid = first + i;
                    let tri = triangles[tid];
                    let h = intersect_tri(o, d, tri, hit.t);
                    if (h.w > 0.5) {
                        hit.t = h.x;
                        hit.u = h.y;
                        hit.v = h.z;
                        hit.tri_idx = tid;
                    }
                }
            } else {
                stack_top++;
                stack[stack_top] = node.left_or_first_tri + 1u;
                stack_top++;
                stack[stack_top] = node.left_or_first_tri;
            }
        }
    }

    return hit;
}

// PCG hash based random numbers
fn pcg_hash(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(seed: ptr<function, u32>) -> f32 {
    *seed = pcg_hash(*seed);
    return f32(*seed) / 4294967295.0;
}

fn point_light_attenuation(dist_sq: f32, inv_range_sq: f32) -> f32 {
    var range_falloff = 1.0;
    if (inv_range_sq > 0.0) {
        let factor = dist_sq * inv_range_sq;
        if (factor >= 1.0) {
            return 0.0;
        }
        let smooth_factor = clamp(1.0 - factor * factor, 0.0, 1.0);
        range_falloff = smooth_factor * smooth_factor;
    }
    return range_falloff / max(dist_sq, 0.0001);
}

fn eval_point_lights(hit_p: vec3f, hit_n: vec3f, eps: f32) -> vec3f {
    var total_radiance = vec3f(0.0);
    let count = u.params.w;
    let inv_pi = 0.318309886;
    for (var i = 0u; i < count; i++) {
        let pl = point_lights[i];
        let to_light = pl.position_range.xyz - hit_p;
        let dist_sq = dot(to_light, to_light);
        let atten = point_light_attenuation(dist_sq, pl.position_range.w);
        if (atten <= 0.0) {
            continue;
        }
        let dist = sqrt(dist_sq);
        let light_dir = select(vec3f(0.0, 1.0, 0.0), to_light / dist, dist > 1e-4);
        let n_dot_l = max(0.0, dot(hit_n, light_dir));
        if (n_dot_l <= 0.0) {
            continue;
        }
        var occluded = false;
        if (dist > eps * 2.0) {
            let shadow_hit = trace_ray(hit_p, light_dir, dist - eps);
            occluded = (shadow_hit.tri_idx != 0xFFFFFFFFu);
        }
        if (!occluded) {
            total_radiance += pl.color_intensity.rgb * (pl.color_intensity.w * atten * n_dot_l * inv_pi);
        }
    }
    return total_radiance;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let task_id = gid.x;
    let total_tasks = u.params.z * 6u;
    if (task_id >= total_tasks) { return; }

    let voxel_id = task_id / 6u;
    let face_id = task_id % 6u;

    let rx = u.resolution.x;
    let ry = u.resolution.y;
    let rz = u.resolution.z;

    let vx = voxel_id % rx;
    let vy = (voxel_id / rx) % ry;
    let vz = voxel_id / (rx * ry);

    // Compute voxel center in world space
    let t_coord = (vec3f(f32(vx) + 0.5, f32(vy) + 0.5, f32(vz) + 0.5)) / vec3f(f32(rx), f32(ry), f32(rz));
    let p_voxel = u.bounds_min.xyz + t_coord * (u.bounds_max.xyz - u.bounds_min.xyz);

    // Ambient cube basis vectors
    var N: vec3f;
    var T: vec3f;
    var B: vec3f;

    switch (face_id) {
        case 0u: { // +X
            N = vec3f(1.0, 0.0, 0.0);
            T = vec3f(0.0, 1.0, 0.0);
            B = vec3f(0.0, 0.0, 1.0);
        }
        case 1u: { // -X
            N = vec3f(-1.0, 0.0, 0.0);
            T = vec3f(0.0, 1.0, 0.0);
            B = vec3f(0.0, 0.0, -1.0);
        }
        case 2u: { // +Y
            N = vec3f(0.0, 1.0, 0.0);
            T = vec3f(1.0, 0.0, 0.0);
            B = vec3f(0.0, 0.0, 1.0);
        }
        case 3u: { // -Y
            N = vec3f(0.0, -1.0, 0.0);
            T = vec3f(1.0, 0.0, 0.0);
            B = vec3f(0.0, 0.0, -1.0);
        }
        case 4u: { // +Z
            N = vec3f(0.0, 0.0, 1.0);
            T = vec3f(1.0, 0.0, 0.0);
            B = vec3f(0.0, 1.0, 0.0);
        }
        default: { // 5u: -Z
            N = vec3f(0.0, 0.0, -1.0);
            T = vec3f(-1.0, 0.0, 0.0);
            B = vec3f(0.0, 1.0, 0.0);
        }
    }

    var seed = task_id + u.params.y * 1973u;
    let spp = u.resolution.w;
    let max_bounces = u.params.x;

    var accum_irradiance = vec3f(0.0);

    for (var s = 0u; s < spp; s++) {
        let u1 = rand_f32(&seed);
        let u2 = rand_f32(&seed);

        // Cosine-weighted hemisphere sampling
        let phi = 6.28318530718 * u1;
        let r = sqrt(u2);
        let x = r * cos(phi);
        let y = r * sin(phi);
        let z = sqrt(max(0.0, 1.0 - u2));

        let ray_dir = normalize(x * T + y * B + z * N);
        var ray_o = p_voxel;
        var throughput = vec3f(3.14159265); // pi factor for cosine estimator

        var ray_rad = vec3f(0.0);
        var cur_dir = ray_dir;

        let extent_scale = max(max(u.bounds_max.x - u.bounds_min.x, u.bounds_max.y - u.bounds_min.y), u.bounds_max.z - u.bounds_min.z);
        let eps = max(0.003, extent_scale * 0.0003);

        // Primary probe-to-sky importance sample (MIS paired with cosine hemisphere sample)
        if (u.env_params.x > 0.5 && u.env_params.w > 0.5) {
            let eu0_1 = rand_f32(&seed);
            let eu0_2 = rand_f32(&seed);
            let env_s0 = sample_hdri_importance(eu0_1, eu0_2);
            let env_dir0 = env_s0.xyz;
            let p_env0 = env_s0.w;
            let n_dot_env0 = dot(N, env_dir0);
            if (p_env0 > 1e-7 && n_dot_env0 > 0.0) {
                let sky_hit = trace_ray(p_voxel, env_dir0, 1e4);
                if (sky_hit.tri_idx == 0xFFFFFFFFu) {
                    let env_rad0 = sample_hdri(env_dir0);
                    let p_cos0 = n_dot_env0 * INV_PI;
                    let w_env0 = (p_env0 * p_env0) / max(p_env0 * p_env0 + p_cos0 * p_cos0, 1e-12);
                    ray_rad += env_rad0 * (n_dot_env0 / p_env0) * w_env0;
                }
            }
        }

        var prev_bsdf_pdf = z * INV_PI;

        for (var b = 0u; b <= max_bounces; b++) {
            let hit = trace_ray(ray_o, cur_dir, 1e4);
            if (hit.tri_idx == 0xFFFFFFFFu) {
                if (u.env_params.x > 0.5) {
                    var env_rad = sample_hdri(cur_dir);
                    if (u.env_params.w > 0.5) {
                        let p_env = hdri_pdf(cur_dir);
                        let w_bsdf = (prev_bsdf_pdf * prev_bsdf_pdf) / max(prev_bsdf_pdf * prev_bsdf_pdf + p_env * p_env, 1e-12);
                        env_rad *= w_bsdf;
                    }
                    ray_rad += throughput * env_rad;
                } else {
                    // Fallback sky gradient
                    let sky_grad = clamp(cur_dir.y * 0.5 + 0.5, 0.0, 1.0);
                    let sky = mix(vec3f(0.12, 0.14, 0.18), vec3f(0.35, 0.42, 0.55), sky_grad);
                    ray_rad += throughput * sky;
                }
                break;
            }

            let tri = triangles[hit.tri_idx];
            let mat = materials[tri.material_idx];

            let w = 1.0 - hit.u - hit.v;
            let uv = fract(w * tri.uv0 + hit.u * tri.uv1 + hit.v * tri.uv2);

            var albedo = mat.base_color.rgb;
            if (mat.tex_index.x >= 0) {
                let s = textureSampleLevel(tex_array, tex_sampler, uv, mat.tex_index.x, 0.0);
                albedo *= s.rgb;
            }

            // Emissive contribution (e.g. ceiling area light)
            ray_rad += throughput * min(mat.emissive.rgb, vec3f(15.0));

            // Interpolate normal
            var hit_n = normalize(w * tri.n0 + hit.u * tri.n1 + hit.v * tri.n2);
            if (dot(hit_n, cur_dir) > 0.0) {
                hit_n = -hit_n;
            }

            let hit_p = ray_o + cur_dir * hit.t + hit_n * eps;

            // Directional Sunlight (dynamic from uniforms/scene)
            if (u.sun_dir.w > 0.5) {
                let sun_dir = normalize(u.sun_dir.xyz);
                let n_dot_sun = max(0.0, dot(hit_n, sun_dir));
                if (n_dot_sun > 0.0) {
                    let shadow_hit = trace_ray(hit_p, sun_dir, 1e4);
                    if (shadow_hit.tri_idx == 0xFFFFFFFFu) {
                        let sun_radiance = u.sun_color.rgb * (u.sun_color.w * n_dot_sun);
                        ray_rad += throughput * albedo * sun_radiance;
                    }
                }
            }

            // Punctual Point Lights (KHR_lights_punctual)
            if (u.params.w > 0u) {
                ray_rad += throughput * albedo * eval_point_lights(hit_p, hit_n, eps);
            }

            // HDRI Next Event Estimation (NEE + MIS)
            if (u.env_params.x > 0.5 && u.env_params.w > 0.5) {
                let eu1 = rand_f32(&seed);
                let eu2 = rand_f32(&seed);
                let env_s = sample_hdri_importance(eu1, eu2);
                let env_dir = env_s.xyz;
                let p_env = env_s.w;
                let n_dot_env = dot(hit_n, env_dir);
                if (p_env > 1e-7 && n_dot_env > 0.0) {
                    let shadow_hit = trace_ray(hit_p, env_dir, 1e4);
                    if (shadow_hit.tri_idx == 0xFFFFFFFFu) {
                        let env_rad = sample_hdri(env_dir);
                        let p_bsdf = n_dot_env * INV_PI;
                        let w_nee = (p_env * p_env) / max(p_env * p_env + p_bsdf * p_bsdf, 1e-12);
                        ray_rad += throughput * albedo * (env_rad * n_dot_env * INV_PI / p_env) * w_nee;
                    }
                }
            } else if (u.env_params.x < 0.5) {
                // Subtle fallback ambient bounce when no HDRI is active
                let ambient_fill = vec3f(0.10, 0.12, 0.15) * (0.4 * (hit_n.y * 0.5 + 0.5));
                ray_rad += throughput * albedo * ambient_fill;
            }

            if (b == max_bounces) {
                break;
            }

            // Diffuse bounce
            let bu1 = rand_f32(&seed);
            let bu2 = rand_f32(&seed);
            let bphi = 6.28318530718 * bu1;
            let br = sqrt(bu2);
            let bx = br * cos(bphi);
            let by = br * sin(bphi);
            let bz = sqrt(max(0.0, 1.0 - bu2));

            // Construct local basis around hit_n
            let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(hit_n.y) > 0.9);
            let bt = normalize(cross(hit_n, up));
            let bb = cross(hit_n, bt);
            let next_dir = normalize(bx * bt + by * bb + bz * hit_n);

            throughput *= albedo;
            if (max(max(throughput.r, throughput.g), throughput.b) < 1e-4) {
                break;
            }

            prev_bsdf_pdf = bz * INV_PI;
            ray_o = hit_p;
            cur_dir = next_dir;
        }

        accum_irradiance += ray_rad;
    }

    accum_irradiance /= f32(spp);

    // Bevy ambient cube 3D texture layout mapping:
    // Dimensions: (Rx, 2*Ry, 3*Rz)
    // S in {+X, -X}: p = vz, t = vy (if +X) or vy + Ry (if -X)
    // S in {+Y, -Y}: p = vz + Rz, t = vy (if +Y) or vy + Ry (if -Y)
    // S in {+Z, -Z}: p = vz + 2*Rz, t = vy (if +Z) or vy + Ry (if -Z)
    var s_coord = vx;
    var t_coord_tex = vy;
    var p_coord = vz;

    switch (face_id) {
        case 0u: { // +X
            t_coord_tex = vy;
            p_coord = vz;
        }
        case 1u: { // -X
            t_coord_tex = vy + ry;
            p_coord = vz;
        }
        case 2u: { // +Y
            t_coord_tex = vy;
            p_coord = vz + rz;
        }
        case 3u: { // -Y
            t_coord_tex = vy + ry;
            p_coord = vz + rz;
        }
        case 4u: { // +Z
            t_coord_tex = vy;
            p_coord = vz + 2u * rz;
        }
        default: { // 5u: -Z
            t_coord_tex = vy + ry;
            p_coord = vz + 2u * rz;
        }
    }

    let tex_w = rx;
    let tex_h = 2u * ry;
    let flat_idx = s_coord + t_coord_tex * tex_w + p_coord * (tex_w * tex_h);

    out_irradiance[flat_idx] = vec4f(accum_irradiance, 1.0);
}
"#;

fn create_point_lights_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    lights: &[GpuPointLight],
    label: &str,
) -> wgpu::Buffer {
    let fallback = [GpuPointLight::zeroed()];
    let slice = if lights.is_empty() { &fallback[..] } else { lights };
    let bytes = bytemuck::cast_slice(slice);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytes);
    buf
}

struct EnvGpuResources {
    tex_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    cdf_buf: wgpu::Buffer,
    env_params: [f32; 4],
    env_dims: [u32; 4],
}

fn create_env_gpu_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    env_map: Option<&EnvironmentMap>,
    intensity: f32,
    rotation_y_rad: f32,
    label_prefix: &str,
) -> EnvGpuResources {
    let (tex_w, tex_h, tex_bytes, cdf_floats, env_params, env_dims) = if let Some(env) = env_map {
        let mut cdf = Vec::with_capacity(env.marginal_cdf.len() + env.conditional_cdf.len());
        cdf.extend_from_slice(&env.marginal_cdf);
        cdf.extend_from_slice(&env.conditional_cdf);
        let has_cdf = if env.has_importance_cdf { 1.0 } else { 0.0 };
        (
            env.width,
            env.height,
            bytemuck::cast_slice(&env.rgba16f_data).to_vec(),
            cdf,
            [1.0, intensity.max(0.0), rotation_y_rad, has_cdf],
            [env.cdf_width, env.cdf_height, env.width, env.height],
        )
    } else {
        let fallback_px = [0u16; 4];
        (
            1,
            1,
            bytemuck::cast_slice(&fallback_px).to_vec(),
            vec![0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [1, 1, 1, 1],
        )
    };

    let env_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&format!("{label_prefix}-env-tex")),
        size: wgpu::Extent3d {
            width: tex_w,
            height: tex_h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        env_tex.as_image_copy(),
        &tex_bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(tex_w * 8),
            rows_per_image: Some(tex_h),
        },
        wgpu::Extent3d {
            width: tex_w,
            height: tex_h,
            depth_or_array_layers: 1,
        },
    );
    let tex_view = env_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    let cdf_bytes = bytemuck::cast_slice(&cdf_floats);
    let cdf_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label_prefix}-env-cdf")),
        size: cdf_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&cdf_buf, 0, cdf_bytes);

    EnvGpuResources {
        tex_view,
        sampler,
        cdf_buf,
        env_params,
        env_dims,
    }
}

pub fn bake_irradiance_volume(
    gpu: &GpuContext,
    bvh: &FlatBvh,
    settings: &BakeSettings,
) -> Result<BakeResult> {
    let start_time = Instant::now();

    let rx = settings.resolution.x;
    let ry = settings.resolution.y;
    let rz = settings.resolution.z;
    let tex_w = rx;
    let tex_h = 2 * ry;
    let tex_d = 3 * rz;
    let total_texels = (tex_w * tex_h * tex_d) as usize;
    let total_voxels = rx * ry * rz;
    let total_tasks = total_voxels * 6;

    eprintln!(
        "[baker] Starting Irradiance Volume Bake: {}x{}x{} voxels ({} total)",
        rx, ry, rz, total_voxels
    );
    eprintln!(
        "[baker] 3D Texture Dimensions: {}x{}x{} ({} texels), SPP: {}, Bounces: {}",
        tex_w, tex_h, tex_d, total_texels, settings.spp, settings.max_bounces
    );

    let (sun_d, sun_enabled) = match settings.sun_dir {
        Some(d) if d.length_squared() > 1e-4 => (d.normalize(), 1.0f32),
        _ => (Vec3::Y, 0.0f32),
    };
    let sun_col = settings.sun_color.unwrap_or(Vec3::new(1.0, 0.98, 0.92));
    let sun_int = match settings.sun_intensity {
        Some(i) if i > 100.0 => (i / 100_000.0) * 3.0,
        Some(i) => i,
        None => 2.8,
    };

    let env_gpu = create_env_gpu_resources(
        &gpu.device,
        &gpu.queue,
        settings.env_map.as_deref(),
        settings.env_intensity,
        settings.env_rotation_y,
        "baker",
    );

    let uniforms = BakeUniforms {
        bounds_min: [settings.bounds_min.x, settings.bounds_min.y, settings.bounds_min.z, 0.0],
        bounds_max: [settings.bounds_max.x, settings.bounds_max.y, settings.bounds_max.z, 0.0],
        resolution: [rx, ry, rz, settings.spp],
        params: [settings.max_bounces, 1337, total_voxels, bvh.point_lights.len() as u32],
        sun_dir: [sun_d.x, sun_d.y, sun_d.z, sun_enabled],
        sun_color: [sun_col.x, sun_col.y, sun_col.z, sun_int],
        env_params: env_gpu.env_params,
        env_dims: env_gpu.env_dims,
    };

    let uniform_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bake-uniforms"),
        size: std::mem::size_of::<BakeUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

    let nodes_bytes = bytemuck::cast_slice(&bvh.nodes);
    let nodes_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bvh-nodes"),
        size: nodes_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&nodes_buf, 0, nodes_bytes);

    let tris_bytes = bytemuck::cast_slice(&bvh.triangles);
    let tris_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("triangles"),
        size: tris_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&tris_buf, 0, tris_bytes);

    let mats_bytes = bytemuck::cast_slice(&bvh.materials);
    let mats_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("materials"),
        size: mats_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&mats_buf, 0, mats_bytes);

    let point_lights_buf = create_point_lights_buffer(
        &gpu.device,
        &gpu.queue,
        &bvh.point_lights,
        "baker-point-lights",
    );

    let output_byte_len = (total_texels * std::mem::size_of::<[f32; 4]>()) as u64;
    let output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out-irradiance"),
        size: output_byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback-irradiance"),
        size: output_byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let tex_array = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("baker-tex-array"),
        size: wgpu::Extent3d {
            width: bvh.texture_size,
            height: bvh.texture_size,
            depth_or_array_layers: bvh.texture_slices,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        tex_array.as_image_copy(),
        &bvh.texture_array,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bvh.texture_size * 4),
            rows_per_image: Some(bvh.texture_size),
        },
        wgpu::Extent3d {
            width: bvh.texture_size,
            height: bvh.texture_size,
            depth_or_array_layers: bvh.texture_slices,
        },
    );
    let tex_view = tex_array.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let tex_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("baker-shader"),
        source: wgpu::ShaderSource::Wgsl(BAKER_SHADER_SRC.into()),
    });

    let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("baker-bgl"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("baker-bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: nodes_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: tris_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: mats_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: output_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&tex_view) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&tex_sampler) },
            wgpu::BindGroupEntry { binding: 7, resource: point_lights_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&env_gpu.tex_view) },
            wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&env_gpu.sampler) },
            wgpu::BindGroupEntry { binding: 10, resource: env_gpu.cdf_buf.as_entire_binding() },
        ],
    });

    let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("baker-pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("baker-pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let workgroups = (total_tasks + 63) / 64;
    eprintln!("[baker] Dispatching {workgroups} workgroups for {total_tasks} face tasks...");

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("baker-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("baker-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &readback_buf, 0, output_byte_len);
    gpu.queue.submit(Some(encoder.finish()));

    let slice = readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    gpu.wait(Duration::from_secs(60))
        .context("Timeout waiting for GPU bake to complete")?;
    rx.recv()
        .context("Channel error waiting for map_async")?
        .context("Failed to map readback buffer")?;

    let view = slice.get_mapped_range()?;
    let raw_data: &[[f32; 4]] = bytemuck::cast_slice(&view);
    let raw_floats: Vec<[f32; 4]> = raw_data.to_vec();
    drop(view);
    readback_buf.unmap();

    // Pack floats to RGB9E5
    let mut packed_pixels: Vec<u32> = Vec::with_capacity(raw_floats.len());
    for p in &raw_floats {
        packed_pixels.push(pack_rgb9e5(p[0], p[1], p[2]));
    }

    let ktx2_bytes = create_ktx2_3d_rgb9e5(tex_w, tex_h, tex_d, &packed_pixels);

    let duration = start_time.elapsed();
    eprintln!("[baker] Bake complete in {:.2?}", duration);

    Ok(BakeResult {
        resolution: settings.resolution,
        texture_dimensions: UVec3::new(tex_w, tex_h, tex_d),
        bounds_min: settings.bounds_min,
        bounds_max: settings.bounds_max,
        ktx2_bytes,
        raw_floats,
        duration,
    })
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ViewportPtUniforms {
    pub eye: [f32; 4],
    pub forward: [f32; 4],
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub bounds_min: [f32; 4],
    pub bounds_max: [f32; 4],
    pub dims: [u32; 4],      // width, height, spp, max_bounces
    pub params: [u32; 4],    // rng_seed, mode (0=PT, 1=Raster+BakedGI), has_irradiance, pad
    pub grid_res: [u32; 4],  // rx, ry, rz, point_light_count
    pub sun_dir: [f32; 4],   // xyz = sun direction, w = enabled flag (1.0 or 0.0)
    pub sun_color: [f32; 4], // rgb = sun color, w = intensity
    pub env_params: [f32; 4], // x = enabled (1/0), y = intensity, z = rotation_y_rad, w = has_cdf (1/0)
    pub env_dims: [u32; 4],   // cdf_width, cdf_height, tex_width, tex_height
}

#[derive(Clone, Debug)]
pub struct ViewportPtSettings {
    pub eye: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub width: u32,
    pub height: u32,
    pub spp: u32,
    pub max_bounces: u32,
    pub seed: u32,
    pub mode: u32,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub resolution: UVec3,
    pub irradiance_voxels: Option<Vec<[f32; 4]>>,
    pub sun_dir: Option<Vec3>,
    pub sun_color: Option<Vec3>,
    pub sun_intensity: Option<f32>,
    pub env_map: Option<Arc<EnvironmentMap>>,
    pub env_intensity: f32,
    pub env_rotation_y: f32,
}

const VIEWPORT_SHADER_SRC: &str = r#"
struct ViewportPtUniforms {
    eye: vec4f,
    forward: vec4f,
    right: vec4f,
    up: vec4f,
    bounds_min: vec4f,
    bounds_max: vec4f,
    dims: vec4u,     // width, height, spp, max_bounces
    params: vec4u,   // rng_seed, mode, has_irradiance, pad
    grid_res: vec4u, // rx, ry, rz, point_light_count
    sun_dir: vec4f,
    sun_color: vec4f,
    env_params: vec4f,
    env_dims: vec4u,
};

struct GpuBvhNode {
    min: vec3f,
    left_or_first_tri: u32,
    max: vec3f,
    tri_count: u32,
};

struct GpuTriangle {
    p0: vec3f,
    material_idx: u32,
    p1: vec3f,
    flags: u32,
    p2: vec3f,
    pad0: u32,
    n0: vec3f,
    pad1: u32,
    n1: vec3f,
    pad2: u32,
    n2: vec3f,
    pad3: u32,
    uv0: vec2f,
    uv1: vec2f,
    uv2: vec2f,
    pad4: vec2f,
};

struct GpuMaterial {
    base_color: vec4f,
    emissive: vec4f,
    roughness_metallic: vec4f,
    tex_index: vec4i,
};

struct GpuPointLight {
    position_range: vec4f,  // xyz = world position, w = inverse_range_squared (0.0 if unbounded)
    color_intensity: vec4f, // rgb = linear color, w = intensity (candela)
};

@group(0) @binding(0) var<uniform> u: ViewportPtUniforms;
@group(0) @binding(1) var<storage, read> bvh_nodes: array<GpuBvhNode>;
@group(0) @binding(2) var<storage, read> triangles: array<GpuTriangle>;
@group(0) @binding(3) var<storage, read> materials: array<GpuMaterial>;
@group(0) @binding(4) var<storage, read_write> out_pixels: array<u32>;
@group(0) @binding(5) var tex_array: texture_2d_array<f32>;
@group(0) @binding(6) var tex_sampler: sampler;
@group(0) @binding(7) var<storage, read> irradiance_voxels: array<vec4f>;
@group(0) @binding(8) var<storage, read> point_lights: array<GpuPointLight>;
@group(0) @binding(9) var env_tex: texture_2d<f32>;
@group(0) @binding(10) var env_sampler: sampler;
@group(0) @binding(11) var<storage, read> env_cdf: array<f32>;

const PI: f32 = 3.14159265359;
const INV_PI: f32 = 0.31830988618;
const TWO_PI: f32 = 6.28318530718;

fn rotate_y(v: vec3f, angle: f32) -> vec3f {
    if (abs(angle) < 1e-6) { return v; }
    let s = sin(angle);
    let c = cos(angle);
    return vec3f(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}

fn sample_hdri(world_dir: vec3f) -> vec3f {
    let d = normalize(rotate_y(world_dir, -u.env_params.z));
    let u_coord = atan2(d.z, d.x) / TWO_PI + 0.5;
    let v_coord = acos(clamp(d.y, -1.0, 1.0)) * INV_PI;
    return textureSampleLevel(env_tex, env_sampler, vec2f(u_coord, v_coord), 0.0).rgb * u.env_params.y;
}

fn hdri_pdf(world_dir: vec3f) -> f32 {
    if (u.env_params.w < 0.5) {
        return 0.0;
    }
    let cw = u.env_dims.x;
    let ch = u.env_dims.y;
    if (cw == 0u || ch == 0u) { return 0.0; }
    let d = normalize(rotate_y(world_dir, -u.env_params.z));
    let u_coord = fract(atan2(d.z, d.x) / TWO_PI + 0.5);
    let v_coord = clamp(acos(clamp(d.y, -1.0, 1.0)) * INV_PI, 0.0, 0.999999);

    let x = min(u32(floor(u_coord * f32(cw))), cw - 1u);
    let y = min(u32(floor(v_coord * f32(ch))), ch - 1u);

    let p_row = max(0.0, env_cdf[y + 1u] - env_cdf[y]);
    let cond_base = (ch + 1u) + y * (cw + 1u);
    let p_col = max(0.0, env_cdf[cond_base + x + 1u] - env_cdf[cond_base + x]);
    let sin_theta = max(sqrt(max(1.0 - d.y * d.y, 0.0)), 1e-5);
    return (p_row * p_col * f32(cw * ch)) / (2.0 * PI * PI * sin_theta);
}

fn sample_hdri_importance(u1: f32, u2: f32) -> vec4f {
    let cw = u.env_dims.x;
    let ch = u.env_dims.y;
    var lo = 0u;
    var hi = ch;
    while (lo + 1u < hi) {
        let mid = (lo + hi) >> 1u;
        if (env_cdf[mid] <= u1) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let y = lo;
    let m0 = env_cdf[y];
    let m1 = env_cdf[y + 1u];
    let p_row = max(0.0, m1 - m0);
    let dy = clamp((u1 - m0) / max(p_row, 1e-8), 0.0, 0.999999);
    let v_coord = (f32(y) + dy) / f32(ch);

    let cond_base = (ch + 1u) + y * (cw + 1u);
    var clo = 0u;
    var chi = cw;
    while (clo + 1u < chi) {
        let mid = (clo + chi) >> 1u;
        if (env_cdf[cond_base + mid] <= u2) {
            clo = mid;
        } else {
            chi = mid;
        }
    }
    let x = clo;
    let c0 = env_cdf[cond_base + x];
    let c1 = env_cdf[cond_base + x + 1u];
    let p_col = max(0.0, c1 - c0);
    let dx = clamp((u2 - c0) / max(p_col, 1e-8), 0.0, 0.999999);
    let u_coord = (f32(x) + dx) / f32(cw);

    let phi = (u_coord - 0.5) * TWO_PI;
    let theta = v_coord * PI;
    let sin_theta = sin(theta);
    let env_dir = vec3f(sin_theta * cos(phi), cos(theta), sin_theta * sin(phi));
    let world_dir = normalize(rotate_y(env_dir, u.env_params.z));
    let pdf_omega = (p_row * p_col * f32(cw * ch)) / (2.0 * PI * PI * max(sin_theta, 1e-5));
    return vec4f(world_dir, pdf_omega);
}

fn sample_face_trilinear(face: u32, stp: vec3f) -> vec3f {
    let rx = u.grid_res.x;
    let ry = u.grid_res.y;
    let rz = u.grid_res.z;
    if (rx == 0u || ry == 0u || rz == 0u) {
        return vec3f(0.0);
    }
    let tex_w = rx;
    let tex_h = 2u * ry;

    let fx = clamp(stp.x, 0.0, f32(rx - 1u));
    let fy = clamp(stp.y, 0.0, f32(ry - 1u));
    let fz = clamp(stp.z, 0.0, f32(rz - 1u));

    let x0 = u32(floor(fx));
    let x1 = min(x0 + 1u, rx - 1u);
    let wx = fx - f32(x0);

    let y0 = u32(floor(fy));
    let y1 = min(y0 + 1u, ry - 1u);
    let wy = fy - f32(y0);

    let z0 = u32(floor(fz));
    let z1 = min(z0 + 1u, rz - 1u);
    let wz = fz - f32(z0);

    var t_off = 0u;
    if ((face & 1u) == 1u) {
        t_off = ry;
    }
    let p_off = (face / 2u) * rz;

    let slice_stride = tex_w * tex_h;

    let c000 = irradiance_voxels[x0 + (y0 + t_off) * tex_w + (z0 + p_off) * slice_stride].rgb;
    let c100 = irradiance_voxels[x1 + (y0 + t_off) * tex_w + (z0 + p_off) * slice_stride].rgb;
    let c010 = irradiance_voxels[x0 + (y1 + t_off) * tex_w + (z0 + p_off) * slice_stride].rgb;
    let c110 = irradiance_voxels[x1 + (y1 + t_off) * tex_w + (z0 + p_off) * slice_stride].rgb;
    let c001 = irradiance_voxels[x0 + (y0 + t_off) * tex_w + (z1 + p_off) * slice_stride].rgb;
    let c101 = irradiance_voxels[x1 + (y0 + t_off) * tex_w + (z1 + p_off) * slice_stride].rgb;
    let c011 = irradiance_voxels[x0 + (y1 + t_off) * tex_w + (z1 + p_off) * slice_stride].rgb;
    let c111 = irradiance_voxels[x1 + (y1 + t_off) * tex_w + (z1 + p_off) * slice_stride].rgb;

    let c00 = mix(c000, c100, wx);
    let c10 = mix(c010, c110, wx);
    let c01 = mix(c001, c101, wx);
    let c11 = mix(c011, c111, wx);

    let c0 = mix(c00, c10, wy);
    let c1 = mix(c01, c11, wy);

    return mix(c0, c1, wz);
}

fn sample_irradiance_volume(p: vec3f, n: vec3f) -> vec3f {
    let bmin = u.bounds_min.xyz;
    let bmax = u.bounds_max.xyz;
    let bsize = max(bmax - bmin, vec3f(1e-4));
    let rel_p = clamp((p - bmin) / bsize, vec3f(0.0), vec3f(1.0));

    let rx = f32(u.grid_res.x);
    let ry = f32(u.grid_res.y);
    let rz = f32(u.grid_res.z);
    let res = vec3f(rx, ry, rz);

    let stp = clamp(rel_p * res, vec3f(0.5), res - vec3f(0.5)) - vec3f(0.5);

    let face_x = select(1u, 0u, n.x >= 0.0);
    let face_y = select(3u, 2u, n.y >= 0.0);
    let face_z = select(5u, 4u, n.z >= 0.0);

    let col_x = sample_face_trilinear(face_x, stp);
    let col_y = sample_face_trilinear(face_y, stp);
    let col_z = sample_face_trilinear(face_z, stp);

    let NN = n * n;
    return col_x * NN.x + col_y * NN.y + col_z * NN.z;
}

struct HitRecord {
    t: f32,
    u: f32,
    v: f32,
    tri_idx: u32,
};

fn intersect_box(o: vec3f, inv_d: vec3f, bmin: vec3f, bmax: vec3f, t_max: f32) -> bool {
    let t0 = (bmin - o) * inv_d;
    let t1 = (bmax - o) * inv_d;
    let tnear = min(t0, t1);
    let tfar = max(t0, t1);
    let enter = max(max(tnear.x, tnear.y), tnear.z);
    let exit = min(min(tfar.x, tfar.y), tfar.z);
    return enter <= exit && exit > 0.001 && enter < t_max;
}

fn intersect_tri(o: vec3f, d: vec3f, tri: GpuTriangle, t_max: f32) -> vec4f {
    let e1 = tri.p1 - tri.p0;
    let e2 = tri.p2 - tri.p0;
    let h = cross(d, e2);
    let a = dot(e1, h);
    if (abs(a) < 1e-7) { return vec4f(0.0); }
    let f = 1.0 / a;
    let s = o - tri.p0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) { return vec4f(0.0); }
    let q = cross(s, e1);
    let v = f * dot(d, q);
    if (v < 0.0 || u + v > 1.0) { return vec4f(0.0); }
    let t = f * dot(e2, q);
    if (t > 0.002 && t < t_max) {
        return vec4f(t, u, v, 1.0);
    }
    return vec4f(0.0);
}

fn trace_ray(o: vec3f, d: vec3f, max_dist: f32) -> HitRecord {
    var hit: HitRecord;
    hit.t = max_dist;
    hit.tri_idx = 0xFFFFFFFFu;

    var inv_d = vec3f(1.0) / d;
    if (abs(d.x) < 1e-6) { inv_d.x = 1e6; }
    if (abs(d.y) < 1e-6) { inv_d.y = 1e6; }
    if (abs(d.z) < 1e-6) { inv_d.z = 1e6; }

    var stack: array<u32, 32>;
    var stack_top = 0;
    stack[0] = 0u;

    while (stack_top >= 0) {
        let node_idx = stack[stack_top];
        stack_top--;

        let node = bvh_nodes[node_idx];
        if (intersect_box(o, inv_d, node.min, node.max, hit.t)) {
            if (node.tri_count > 0u) {
                let first = node.left_or_first_tri;
                for (var i = 0u; i < node.tri_count; i++) {
                    let tid = first + i;
                    let tri = triangles[tid];
                    let h = intersect_tri(o, d, tri, hit.t);
                    if (h.w > 0.5) {
                        hit.t = h.x;
                        hit.u = h.y;
                        hit.v = h.z;
                        hit.tri_idx = tid;
                    }
                }
            } else {
                stack_top++;
                stack[stack_top] = node.left_or_first_tri + 1u;
                stack_top++;
                stack[stack_top] = node.left_or_first_tri;
            }
        }
    }

    return hit;
}

fn pcg_hash(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(seed: ptr<function, u32>) -> f32 {
    *seed = pcg_hash(*seed);
    return f32(*seed) / 4294967295.0;
}

fn point_light_attenuation(dist_sq: f32, inv_range_sq: f32) -> f32 {
    var range_falloff = 1.0;
    if (inv_range_sq > 0.0) {
        let factor = dist_sq * inv_range_sq;
        if (factor >= 1.0) {
            return 0.0;
        }
        let smooth_factor = clamp(1.0 - factor * factor, 0.0, 1.0);
        range_falloff = smooth_factor * smooth_factor;
    }
    return range_falloff / max(dist_sq, 0.0001);
}

fn eval_point_lights(hit_p: vec3f, hit_n: vec3f, eps: f32) -> vec3f {
    var total_radiance = vec3f(0.0);
    let count = u.grid_res.w;
    let inv_pi = 0.318309886;
    for (var i = 0u; i < count; i++) {
        let pl = point_lights[i];
        let to_light = pl.position_range.xyz - hit_p;
        let dist_sq = dot(to_light, to_light);
        let atten = point_light_attenuation(dist_sq, pl.position_range.w);
        if (atten <= 0.0) {
            continue;
        }
        let dist = sqrt(dist_sq);
        let light_dir = select(vec3f(0.0, 1.0, 0.0), to_light / dist, dist > 1e-4);
        let n_dot_l = max(0.0, dot(hit_n, light_dir));
        if (n_dot_l <= 0.0) {
            continue;
        }
        var occluded = false;
        if (dist > eps * 2.0) {
            let shadow_hit = trace_ray(hit_p, light_dir, dist - eps);
            occluded = (shadow_hit.tri_idx != 0xFFFFFFFFu);
        }
        if (!occluded) {
            total_radiance += pl.color_intensity.rgb * (pl.color_intensity.w * atten * n_dot_l * inv_pi);
        }
    }
    return total_radiance;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let px = gid.x;
    let py = gid.y;
    let width = u.dims.x;
    let height = u.dims.y;
    if (px >= width || py >= height) { return; }

    let mode = u.params.y; // 0 = Pathtracer Preview, 1 = Raster + Baked GI
    let has_irr = u.params.z;

    let tan_half_fov = 0.41421356f;
    let aspect = f32(width) / f32(height);

    if (mode == 1u) {
        // Mode 1: Raster + Baked GI (Fast deterministic, single primary ray + direct sun + baked irradiance volume)
        let ndc_x = ((f32(px) + 0.5) / f32(width)) * 2.0 - 1.0;
        let ndc_y = 1.0 - ((f32(py) + 0.5) / f32(height)) * 2.0;

        let ray_dir = normalize(u.forward.xyz + u.right.xyz * (ndc_x * aspect * tan_half_fov) + u.up.xyz * (ndc_y * tan_half_fov));
        let ray_o = u.eye.xyz;

        let hit = trace_ray(ray_o, ray_dir, 1e5);
        var col_linear = vec3f(0.0);

        if (hit.tri_idx == 0xFFFFFFFFu) {
            if (u.env_params.x > 0.5) {
                col_linear = sample_hdri(ray_dir);
            } else {
                let bg_grad = clamp(ray_dir.y * 0.5 + 0.5, 0.0, 1.0);
                col_linear = mix(vec3f(0.067, 0.071, 0.078), vec3f(0.09, 0.10, 0.12), bg_grad);
            }
        } else {
            let tri = triangles[hit.tri_idx];
            let mat = materials[tri.material_idx];

            let w = 1.0 - hit.u - hit.v;
            let uv = fract(w * tri.uv0 + hit.u * tri.uv1 + hit.v * tri.uv2);

            var albedo = mat.base_color.rgb;
            if (mat.tex_index.x >= 0) {
                let s = textureSampleLevel(tex_array, tex_sampler, uv, mat.tex_index.x, 0.0);
                albedo *= s.rgb;
            }

            var hit_n = normalize(w * tri.n0 + hit.u * tri.n1 + hit.v * tri.n2);
            if (dot(hit_n, ray_dir) > 0.0) {
                hit_n = -hit_n;
            }

            let eps = max(0.003, hit.t * 0.0005);
            let hit_p = ray_o + ray_dir * hit.t + hit_n * eps;

            // Direct Sunlight (dynamic from uniforms/scene)
            var sun_radiance = vec3f(0.0);
            if (u.sun_dir.w > 0.5) {
                let sun_dir = normalize(u.sun_dir.xyz);
                let n_dot_sun = max(0.0, dot(hit_n, sun_dir));
                if (n_dot_sun > 0.0) {
                    let shadow_hit = trace_ray(hit_p, sun_dir, 1e5);
                    if (shadow_hit.tri_idx == 0xFFFFFFFFu) {
                        sun_radiance = u.sun_color.rgb * (u.sun_color.w * n_dot_sun);
                    }
                }
            }

            // Punctual Point Lights (KHR_lights_punctual)
            var point_radiance = vec3f(0.0);
            if (u.grid_res.w > 0u) {
                point_radiance = eval_point_lights(hit_p, hit_n, eps);
            }

            // Indirect Baked GI
            var indirect_gi = vec3f(0.08, 0.09, 0.11) * (hit_n.y * 0.3 + 0.7);
            if (has_irr == 1u) {
                indirect_gi = sample_irradiance_volume(hit_p, hit_n);
            }

            let emissive = min(mat.emissive.rgb, vec3f(15.0));
            col_linear = emissive + albedo * (sun_radiance + point_radiance + indirect_gi);
        }

        var col = col_linear / (col_linear + vec3f(1.0));
        col = pow(col, vec3f(1.0 / 2.2));

        let r = u32(clamp(col.r * 255.0, 0.0, 255.0));
        let g = u32(clamp(col.g * 255.0, 0.0, 255.0));
        let b = u32(clamp(col.b * 255.0, 0.0, 255.0));
        let pixel_idx = py * width + px;
        out_pixels[pixel_idx] = r | (g << 8u) | (b << 16u) | (255u << 24u);
        return;
    }

    let spp = max(1u, u.dims.z);
    let max_bounces = u.dims.w;
    var seed = px + py * width + u.params.x * 1973u;

    var accum_col = vec3f(0.0);

    for (var s = 0u; s < spp; s++) {
        let jitter_x = (rand_f32(&seed) - 0.5) / f32(width);
        let jitter_y = (rand_f32(&seed) - 0.5) / f32(height);
        let ndc_x = ((f32(px) + 0.5) / f32(width) + jitter_x) * 2.0 - 1.0;
        let ndc_y = 1.0 - ((f32(py) + 0.5) / f32(height) + jitter_y) * 2.0;

        let ray_dir = normalize(u.forward.xyz + u.right.xyz * (ndc_x * aspect * tan_half_fov) + u.up.xyz * (ndc_y * tan_half_fov));
        var ray_o = u.eye.xyz;
        var throughput = vec3f(1.0);
        var ray_rad = vec3f(0.0);
        var cur_dir = ray_dir;
        var prev_bsdf_pdf = 0.0f;

        for (var b = 0u; b <= max_bounces; b++) {
            let hit = trace_ray(ray_o, cur_dir, 1e5);
            if (hit.tri_idx == 0xFFFFFFFFu) {
                if (u.env_params.x > 0.5) {
                    var env_rad = sample_hdri(cur_dir);
                    if (b > 0u && u.env_params.w > 0.5) {
                        let p_env = hdri_pdf(cur_dir);
                        let w_bsdf = (prev_bsdf_pdf * prev_bsdf_pdf) / max(prev_bsdf_pdf * prev_bsdf_pdf + p_env * p_env, 1e-12);
                        env_rad *= w_bsdf;
                    }
                    ray_rad += throughput * env_rad;
                } else {
                    let bg_grad = clamp(cur_dir.y * 0.5 + 0.5, 0.0, 1.0);
                    let bg = mix(vec3f(0.067, 0.071, 0.078), vec3f(0.09, 0.10, 0.12), bg_grad);
                    ray_rad += throughput * bg;
                }
                break;
            }

            let tri = triangles[hit.tri_idx];
            let mat = materials[tri.material_idx];

            let w = 1.0 - hit.u - hit.v;
            let uv = fract(w * tri.uv0 + hit.u * tri.uv1 + hit.v * tri.uv2);

            var albedo = mat.base_color.rgb;
            if (mat.tex_index.x >= 0) {
                let s = textureSampleLevel(tex_array, tex_sampler, uv, mat.tex_index.x, 0.0);
                albedo *= s.rgb;
            }

            ray_rad += throughput * min(mat.emissive.rgb, vec3f(15.0));

            var hit_n = normalize(w * tri.n0 + hit.u * tri.n1 + hit.v * tri.n2);
            if (dot(hit_n, cur_dir) > 0.0) {
                hit_n = -hit_n;
            }

            let eps = max(0.003, hit.t * 0.0005);
            let hit_p = ray_o + cur_dir * hit.t + hit_n * eps;

            // Direct Sunlight (dynamic from uniforms/scene)
            if (u.sun_dir.w > 0.5) {
                let sun_dir = normalize(u.sun_dir.xyz);
                let n_dot_sun = max(0.0, dot(hit_n, sun_dir));
                if (n_dot_sun > 0.0) {
                    let shadow_hit = trace_ray(hit_p, sun_dir, 1e5);
                    if (shadow_hit.tri_idx == 0xFFFFFFFFu) {
                        let sun_radiance = u.sun_color.rgb * (u.sun_color.w * n_dot_sun);
                        ray_rad += throughput * albedo * sun_radiance;
                    }
                }
            }

            // Punctual Point Lights (KHR_lights_punctual)
            if (u.grid_res.w > 0u) {
                ray_rad += throughput * albedo * eval_point_lights(hit_p, hit_n, eps);
            }

            // HDRI Next Event Estimation (NEE + MIS)
            if (u.env_params.x > 0.5 && u.env_params.w > 0.5) {
                let eu1 = rand_f32(&seed);
                let eu2 = rand_f32(&seed);
                let env_s = sample_hdri_importance(eu1, eu2);
                let env_dir = env_s.xyz;
                let p_env = env_s.w;
                let n_dot_env = dot(hit_n, env_dir);
                if (p_env > 1e-7 && n_dot_env > 0.0) {
                    let shadow_hit = trace_ray(hit_p, env_dir, 1e5);
                    if (shadow_hit.tri_idx == 0xFFFFFFFFu) {
                        let env_rad = sample_hdri(env_dir);
                        let p_bsdf = n_dot_env * INV_PI;
                        let w_nee = (p_env * p_env) / max(p_env * p_env + p_bsdf * p_bsdf, 1e-12);
                        ray_rad += throughput * albedo * (env_rad * n_dot_env * INV_PI / p_env) * w_nee;
                    }
                }
            }

            if (b == max_bounces) {
                break;
            }

            // Cosine hemisphere diffuse bounce
            let bu1 = rand_f32(&seed);
            let bu2 = rand_f32(&seed);
            let bphi = 6.28318530718 * bu1;
            let br = sqrt(bu2);
            let bx = br * cos(bphi);
            let by = br * sin(bphi);
            let bz = sqrt(max(0.0, 1.0 - bu2));

            var u_tangent = vec3f(0.0, 1.0, 0.0);
            if (abs(hit_n.y) > 0.9) { u_tangent = vec3f(1.0, 0.0, 0.0); }
            let t_vec = normalize(cross(u_tangent, hit_n));
            let b_vec = cross(hit_n, t_vec);
            let next_dir = normalize(bx * t_vec + by * b_vec + bz * hit_n);

            throughput *= albedo;
            if (max(max(throughput.r, throughput.g), throughput.b) < 0.01) {
                break;
            }

            prev_bsdf_pdf = bz * INV_PI;
            ray_o = hit_p;
            cur_dir = next_dir;
        }

        accum_col += ray_rad;
    }

    var col = accum_col / f32(spp);
    col = col / (col + vec3f(1.0));
    col = pow(col, vec3f(1.0 / 2.2));

    let r = u32(clamp(col.r * 255.0, 0.0, 255.0));
    let g = u32(clamp(col.g * 255.0, 0.0, 255.0));
    let b = u32(clamp(col.b * 255.0, 0.0, 255.0));
    let pixel_idx = py * width + px;
    out_pixels[pixel_idx] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}
"#;

pub fn render_viewport_pt(
    gpu: &GpuContext,
    bvh: &FlatBvh,
    settings: &ViewportPtSettings,
) -> Result<Vec<u8>> {
    let width = settings.width;
    let height = settings.height;
    let total_pixels = (width * height) as usize;
    let output_byte_len = (total_pixels * 4) as u64;

    let has_irr = if settings.irradiance_voxels.as_ref().map_or(false, |v| !v.is_empty()) { 1 } else { 0 };
    let (sun_d, sun_enabled) = match settings.sun_dir {
        Some(d) if d.length_squared() > 1e-4 => (d.normalize(), 1.0f32),
        _ => (Vec3::Y, 0.0f32),
    };
    let sun_col = settings.sun_color.unwrap_or(Vec3::new(1.0, 0.98, 0.92));
    let sun_int = match settings.sun_intensity {
        Some(i) if i > 100.0 => (i / 100_000.0) * 3.0,
        Some(i) => i,
        None => 2.8,
    };

    let env_gpu = create_env_gpu_resources(
        &gpu.device,
        &gpu.queue,
        settings.env_map.as_deref(),
        settings.env_intensity,
        settings.env_rotation_y,
        "vp",
    );

    let uniforms = ViewportPtUniforms {
        eye: [settings.eye.x, settings.eye.y, settings.eye.z, 0.0],
        forward: [settings.forward.x, settings.forward.y, settings.forward.z, 0.0],
        right: [settings.right.x, settings.right.y, settings.right.z, 0.0],
        up: [settings.up.x, settings.up.y, settings.up.z, 0.0],
        bounds_min: [settings.bounds_min.x, settings.bounds_min.y, settings.bounds_min.z, 0.0],
        bounds_max: [settings.bounds_max.x, settings.bounds_max.y, settings.bounds_max.z, 0.0],
        dims: [width, height, settings.spp, settings.max_bounces],
        params: [settings.seed, settings.mode, has_irr, 0],
        grid_res: [settings.resolution.x, settings.resolution.y, settings.resolution.z, bvh.point_lights.len() as u32],
        sun_dir: [sun_d.x, sun_d.y, sun_d.z, sun_enabled],
        sun_color: [sun_col.x, sun_col.y, sun_col.z, sun_int],
        env_params: env_gpu.env_params,
        env_dims: env_gpu.env_dims,
    };

    let uniform_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vp-uniforms"),
        size: std::mem::size_of::<ViewportPtUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

    let nodes_bytes = bytemuck::cast_slice(&bvh.nodes);
    let nodes_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vp-nodes"),
        size: nodes_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&nodes_buf, 0, nodes_bytes);

    let tris_bytes = bytemuck::cast_slice(&bvh.triangles);
    let tris_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vp-triangles"),
        size: tris_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&tris_buf, 0, tris_bytes);

    let mats_bytes = bytemuck::cast_slice(&bvh.materials);
    let mats_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vp-materials"),
        size: mats_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&mats_buf, 0, mats_bytes);

    let output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vp-out-pixels"),
        size: output_byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vp-readback"),
        size: output_byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let tex_array = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vp-tex-array"),
        size: wgpu::Extent3d {
            width: bvh.texture_size,
            height: bvh.texture_size,
            depth_or_array_layers: bvh.texture_slices,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        tex_array.as_image_copy(),
        &bvh.texture_array,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bvh.texture_size * 4),
            rows_per_image: Some(bvh.texture_size),
        },
        wgpu::Extent3d {
            width: bvh.texture_size,
            height: bvh.texture_size,
            depth_or_array_layers: bvh.texture_slices,
        },
    );
    let tex_view = tex_array.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let tex_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    let irradiance_slice: &[[f32; 4]] = if let Some(ref data) = settings.irradiance_voxels {
        if !data.is_empty() {
            data.as_slice()
        } else {
            &[[0.0, 0.0, 0.0, 0.0]]
        }
    } else {
        &[[0.0, 0.0, 0.0, 0.0]]
    };
    let irradiance_bytes = bytemuck::cast_slice(irradiance_slice);
    let irradiance_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vp-irradiance"),
        size: irradiance_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&irradiance_buf, 0, irradiance_bytes);

    let point_lights_buf = create_point_lights_buffer(
        &gpu.device,
        &gpu.queue,
        &bvh.point_lights,
        "vp-point-lights",
    );

    let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vp-pt-shader"),
        source: wgpu::ShaderSource::Wgsl(VIEWPORT_SHADER_SRC.into()),
    });

    let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vp-bgl"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 11,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vp-bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: nodes_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: tris_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: mats_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: output_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&tex_view) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&tex_sampler) },
            wgpu::BindGroupEntry { binding: 7, resource: irradiance_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: point_lights_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&env_gpu.tex_view) },
            wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::Sampler(&env_gpu.sampler) },
            wgpu::BindGroupEntry { binding: 11, resource: env_gpu.cdf_buf.as_entire_binding() },
        ],
    });

    let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("vp-pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("vp-pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let wg_x = (width + 7) / 8;
    let wg_y = (height + 7) / 8;

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vp-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vp-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &readback_buf, 0, output_byte_len);
    gpu.queue.submit(Some(encoder.finish()));

    let slice = readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    gpu.wait(Duration::from_secs(20))
        .context("Timeout waiting for Viewport Path Tracer")?;
    rx.recv()
        .context("Channel error waiting for map_async")?
        .context("Failed to map readback buffer")?;

    let view = slice.get_mapped_range()?;
    let pixels: Vec<u8> = view.to_vec();
    drop(view);
    readback_buf.unmap();

    Ok(pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viewport_pt() {
        let gpu = match GpuContext::new_dx12() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("DX12 init failed: {e:#}");
                return;
            }
        };
        let path = std::path::Path::new("C:/tracer-experiment/mutbaker-gui/assets/models/CornellBox/CornellBox.glb");
        if !path.exists() {
            eprintln!("Path not found: {:?}", path);
            return;
        }
        let scene = match crate::scene::Scene::load_gltf(path) {
            Ok(s) => s,
            Err(e) => {
                panic!("load gltf failed: {e:#}");
            }
        };
        let bvh = FlatBvh::build(&scene);
        let sc_cam_center = glam::Vec3::new(-4.17, 4.11, -4.19);
        let cam_dist = 14.5f32;
        let yaw = 0.85f32;
        let pitch = 0.30f32;
        let visual_scale = 0.015f32;

        let cam_pos = sc_cam_center + glam::Vec3::new(
            yaw.cos() * pitch.cos(),
            pitch.sin(),
            yaw.sin() * pitch.cos(),
        ) * cam_dist;

        let eye = cam_pos / visual_scale;
        let center = sc_cam_center / visual_scale;
        let forward = (center - eye).normalize();
        let right = forward.cross(glam::Vec3::Y).normalize();
        let up = right.cross(forward).normalize();

        println!("Test Cam: eye={:?}, forward={:?}, right={:?}, up={:?}", eye, forward, right, up);

        let settings = ViewportPtSettings {
            eye,
            forward,
            right,
            up,
            width: 480,
            height: 380,
            spp: 4,
            max_bounces: 2,
            seed: 1234,
            mode: 0,
            bounds_min: glam::Vec3::new(-556.0, 0.0, -559.2),
            bounds_max: glam::Vec3::new(0.0, 548.8, 0.0),
            resolution: glam::UVec3::new(4, 4, 4),
            irradiance_voxels: None,
            sun_dir: scene.directional_light.as_ref().map(|s| s.direction),
            sun_color: scene.directional_light.as_ref().map(|s| s.color),
            sun_intensity: scene.directional_light.as_ref().map(|s| s.intensity),
            env_map: None,
            env_intensity: 1.0,
            env_rotation_y: 0.0,
        };
        let res = render_viewport_pt(&gpu, &bvh, &settings);
        match res {
            Ok(pixels) => {
                println!("SUCCESS! Rendered {} pixels", pixels.len());
                let non_zero_count = pixels.chunks(4).filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0).count();
                println!("Non-black pixels: {} / {}", non_zero_count, pixels.len() / 4);
                // Check sample colors
                let mut colors: std::collections::BTreeMap<(u8, u8, u8), usize> = std::collections::BTreeMap::new();
                for chunk in pixels.chunks(4) {
                    *colors.entry((chunk[0], chunk[1], chunk[2])).or_insert(0) += 1;
                }
                println!("Top 5 unique colors:");
                let mut sorted_colors: Vec<_> = colors.into_iter().collect();
                sorted_colors.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
                for (col, count) in sorted_colors.iter().take(5) {
                    println!("  Color {:?}: {} pixels", col, count);
                }
            }
            Err(e) => {
                panic!("render_viewport_pt FAILED: {:#}", e);
            }
        }
    }

    fn make_quad_tris(center: Vec3, half_size: f32, normal_y: f32) -> [crate::scene::Triangle; 2] {
        let n = Vec3::new(0.0, normal_y, 0.0);
        let p0 = center + Vec3::new(-half_size, 0.0, -half_size);
        let p1 = center + Vec3::new(half_size, 0.0, -half_size);
        let p2 = center + Vec3::new(half_size, 0.0, half_size);
        let p3 = center + Vec3::new(-half_size, 0.0, half_size);
        let mk = |a: Vec3, b: Vec3, c: Vec3| crate::scene::Triangle {
            p0: a,
            p1: b,
            p2: c,
            n0: n,
            n1: n,
            n2: n,
            uv0: [0.0, 0.0],
            uv1: [1.0, 0.0],
            uv2: [1.0, 1.0],
            material_id: 0,
        };
        [mk(p0, p1, p2), mk(p0, p2, p3)]
    }

    fn build_synthetic_point_light_scene(
        with_occluder: bool,
        point_lights: Vec<crate::scene::PointLightData>,
    ) -> crate::scene::Scene {
        let mut triangles = Vec::new();
        triangles.extend_from_slice(&make_quad_tris(Vec3::ZERO, 4.0, 1.0));
        if with_occluder {
            triangles.extend_from_slice(&make_quad_tris(Vec3::new(0.0, 1.0, 0.0), 0.75, 1.0));
        }
        let mut mat = crate::scene::Material::default();
        mat.base_color = glam::Vec4::new(0.8, 0.8, 0.8, 1.0);
        crate::scene::Scene {
            triangles,
            materials: vec![mat],
            texture_array: vec![255, 255, 255, 255],
            texture_slices: 1,
            texture_size: 1,
            bounds_min: Vec3::new(-4.0, -0.5, -4.0),
            bounds_max: Vec3::new(4.0, 3.0, 4.0),
            directional_light: None,
            point_lights,
        }
    }

    #[test]
    fn test_point_light_bake_pt_occlusion_and_range() {
        let gpu = match GpuContext::new_dx12() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("DX12 init failed: {e:#}");
                return;
            }
        };

        let lit_scene = build_synthetic_point_light_scene(
            false,
            vec![crate::scene::PointLightData {
                position: Vec3::new(0.0, 2.0, 0.0),
                color: Vec3::new(1.0, 0.5, 0.25),
                intensity: 100.0,
                range: None,
            }],
        );
        let unlit_scene = build_synthetic_point_light_scene(false, vec![]);
        let occluded_scene = build_synthetic_point_light_scene(
            true,
            vec![crate::scene::PointLightData {
                position: Vec3::new(0.0, 2.0, 0.0),
                color: Vec3::new(1.0, 0.5, 0.25),
                intensity: 100.0,
                range: None,
            }],
        );
        let out_of_range_scene = build_synthetic_point_light_scene(
            false,
            vec![crate::scene::PointLightData {
                position: Vec3::new(0.0, 2.0, 0.0),
                color: Vec3::new(1.0, 0.5, 0.25),
                intensity: 100.0,
                range: Some(1.5),
            }],
        );

        let lit_bvh = FlatBvh::build(&lit_scene);
        let unlit_bvh = FlatBvh::build(&unlit_scene);
        let occluded_bvh = FlatBvh::build(&occluded_scene);
        let range_bvh = FlatBvh::build(&out_of_range_scene);
        assert_eq!(lit_bvh.point_lights.len(), 1);
        assert_eq!(unlit_bvh.point_lights.len(), 0);

        let bake_settings = BakeSettings {
            resolution: UVec3::new(2, 2, 2),
            bounds_min: Vec3::new(-2.0, 0.1, -2.0),
            bounds_max: Vec3::new(2.0, 2.5, 2.0),
            spp: 16,
            max_bounces: 1,
            sun_dir: None,
            sun_color: None,
            sun_intensity: None,
            env_map: None,
            env_intensity: 1.0,
            env_rotation_y: 0.0,
        };
        let lit_bake = bake_irradiance_volume(&gpu, &lit_bvh, &bake_settings).unwrap();
        let unlit_bake = bake_irradiance_volume(&gpu, &unlit_bvh, &bake_settings).unwrap();
        let sum_r = |res: &BakeResult| -> f32 { res.raw_floats.iter().map(|v| v[0]).sum() };
        assert!(
            sum_r(&lit_bake) > sum_r(&unlit_bake) * 2.0,
            "Point light should significantly increase baked irradiance: lit={}, unlit={}",
            sum_r(&lit_bake),
            sum_r(&unlit_bake)
        );

        // Camera looking down at (0, 0, 0) from (0, 5, 5)
        let eye = Vec3::new(0.0, 3.0, 4.0);
        let forward = (Vec3::ZERO - eye).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward).normalize();
        let vp_settings = |mode: u32| ViewportPtSettings {
            eye,
            forward,
            right,
            up,
            width: 32,
            height: 32,
            spp: 2,
            max_bounces: 1,
            seed: 42,
            mode,
            bounds_min: bake_settings.bounds_min,
            bounds_max: bake_settings.bounds_max,
            resolution: bake_settings.resolution,
            irradiance_voxels: None,
            sun_dir: None,
            sun_color: None,
            sun_intensity: None,
            env_map: None,
            env_intensity: 1.0,
            env_rotation_y: 0.0,
        };

        let center_px = |pixels: &[u8]| -> [u8; 3] {
            let idx = (16 * 32 + 16) * 4;
            [pixels[idx], pixels[idx + 1], pixels[idx + 2]]
        };

        // Mode 0 (Pathtracer) & Mode 1 (Raster + Baked GI): lit > unlit, occluded shadowed, range=1.5 cut off
        for mode in [0u32, 1u32] {
            let lit_px = render_viewport_pt(&gpu, &lit_bvh, &vp_settings(mode)).unwrap();
            let unlit_px = render_viewport_pt(&gpu, &unlit_bvh, &vp_settings(mode)).unwrap();
            let occ_px = render_viewport_pt(&gpu, &occluded_bvh, &vp_settings(mode)).unwrap();
            let rng_px = render_viewport_pt(&gpu, &range_bvh, &vp_settings(mode)).unwrap();

            let lc = center_px(&lit_px);
            let uc = center_px(&unlit_px);
            let rc = center_px(&rng_px);
            assert!(
                lc[0] > uc[0] + 40 && lc[0] > lc[2],
                "mode {mode}: warm point light should brighten center pixel: lit={lc:?}, unlit={uc:?}"
            );
            assert_eq!(
                rc, uc,
                "mode {mode}: point light with range=1.5 at dist=2.0 should attenuate to zero: rng={rc:?}, unlit={uc:?}"
            );
            // At pixel (16, 26) (lower on screen = front of floor outside occluder? check center under occluder vs unlit)
            let oc = center_px(&occ_px);
            assert!(
                oc[0] < lc[0],
                "mode {mode}: occluder should reduce luminance vs unoccluded lit floor: occ={oc:?}, lit={lc:?}"
            );
        }
    }

    #[test]
    fn test_cubes_glb_point_light_bake_and_pt() {
        let gpu = match GpuContext::new_dx12() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("DX12 init failed: {e:#}");
                return;
            }
        };
        let path = std::path::Path::new("C:/tracer-experiment/mutbaker-gui/assets/models/cubes/Cubes.glb");
        if !path.exists() {
            eprintln!("Path not found: {:?}", path);
            return;
        }
        let scene = crate::scene::Scene::load_gltf(path).expect("load Cubes.glb");
        assert_eq!(scene.point_lights.len(), 1, "Cubes.glb should contain 1 point light");
        let bvh = FlatBvh::build(&scene);
        assert_eq!(bvh.point_lights.len(), 1);

        let bake = bake_irradiance_volume(
            &gpu,
            &bvh,
            &BakeSettings {
                resolution: UVec3::new(4, 4, 4),
                bounds_min: scene.bounds_min,
                bounds_max: scene.bounds_max,
                spp: 16,
                max_bounces: 2,
                sun_dir: None,
                sun_color: None,
                sun_intensity: None,
                env_map: None,
                env_intensity: 1.0,
                env_rotation_y: 0.0,
            },
        )
        .expect("bake Cubes.glb");
        let max_irr = bake
            .raw_floats
            .iter()
            .fold(0.0f32, |acc, v| acc.max(v[0]).max(v[1]).max(v[2]));
        assert!(max_irr > 0.5, "Cubes.glb point light should bake significant irradiance, got {max_irr}");

        let eye = Vec3::new(8.0, 5.0, 8.0);
        let forward = (Vec3::new(0.0, 1.5, 0.0) - eye).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward).normalize();
        let pixels = render_viewport_pt(
            &gpu,
            &bvh,
            &ViewportPtSettings {
                eye,
                forward,
                right,
                up,
                width: 128,
                height: 128,
                spp: 2,
                max_bounces: 2,
                seed: 7,
                mode: 0,
                bounds_min: scene.bounds_min,
                bounds_max: scene.bounds_max,
                resolution: UVec3::new(4, 4, 4),
                irradiance_voxels: None,
                sun_dir: None,
                sun_color: None,
                sun_intensity: None,
                env_map: None,
                env_intensity: 1.0,
                env_rotation_y: 0.0,
            },
        )
        .expect("render Cubes.glb PT");
        let non_black = pixels.chunks(4).filter(|p| p[0] > 10 || p[1] > 10 || p[2] > 10).count();
        assert!(non_black > 100, "Expected lit pixels in Cubes.glb PT preview, got {non_black}");
    }

    #[test]
    fn test_hdri_bake_directionality_rotation_and_pt() {
        let gpu = match GpuContext::new_dx12() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("DX12 init failed: {e:#}");
                return;
            }
        };

        // Create a synthetic 16x8 equirectangular HDRI with a very bright warm light at u=0.5 (+X)
        // since phi = (u - 0.5) * 2*PI => u=0.5 => phi=0 => dir = (+1, 0, 0) at v=0.5.
        let w = 16u32;
        let h = 8u32;
        let mut pixels = vec![[0.02f32, 0.02, 0.04]; (w * h) as usize];
        for y in 3..=4 {
            for x in 7..=8 {
                pixels[(y * w + x) as usize] = [40.0, 12.0, 2.0];
            }
        }
        let env = Arc::new(EnvironmentMap::from_rgb_f32("sun_px.hdr", w, h, &pixels).unwrap());

        let scene = build_synthetic_point_light_scene(false, vec![]);
        let bvh = FlatBvh::build(&scene);

        let bake_0 = bake_irradiance_volume(
            &gpu,
            &bvh,
            &BakeSettings {
                resolution: UVec3::new(1, 1, 1),
                bounds_min: Vec3::new(-0.5, 1.0, -0.5),
                bounds_max: Vec3::new(0.5, 2.0, 0.5),
                spp: 32,
                max_bounces: 1,
                sun_dir: None,
                sun_color: None,
                sun_intensity: None,
                env_map: Some(env.clone()),
                env_intensity: 1.0,
                env_rotation_y: 0.0,
            },
        )
        .expect("HDRI bake rot 0");

        // Layout for 1x1x1 (1x2x3 texels):
        // +X: (0, 0, 0) -> idx 0
        // -X: (0, 1, 0) -> idx 1
        let px_0 = bake_0.raw_floats[0];
        let nx_0 = bake_0.raw_floats[1];
        assert!(
            px_0[0] > nx_0[0] * 4.0 && px_0[0] > px_0[2] * 2.0,
            "HDRI at +X should strongly light +X face with warm color: +X={px_0:?}, -X={nx_0:?}"
        );

        let bake_rot_pi = bake_irradiance_volume(
            &gpu,
            &bvh,
            &BakeSettings {
                resolution: UVec3::new(1, 1, 1),
                bounds_min: Vec3::new(-0.5, 1.0, -0.5),
                bounds_max: Vec3::new(0.5, 2.0, 0.5),
                spp: 32,
                max_bounces: 1,
                sun_dir: None,
                sun_color: None,
                sun_intensity: None,
                env_map: Some(env.clone()),
                env_intensity: 1.0,
                env_rotation_y: std::f32::consts::PI,
            },
        )
        .expect("HDRI bake rot PI");

        let px_pi = bake_rot_pi.raw_floats[0];
        let nx_pi = bake_rot_pi.raw_floats[1];
        assert!(
            nx_pi[0] > px_pi[0] * 4.0 && nx_pi[0] > nx_pi[2] * 2.0,
            "Rotating HDRI by PI should move bright patch to -X face: +X={px_pi:?}, -X={nx_pi:?}"
        );

        // Verify Viewport PT renders HDRI background and floor illumination
        let eye = Vec3::new(0.0, 1.5, 3.0);
        let forward = (Vec3::new(0.0, 0.8, 0.0) - eye).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward).normalize();
        let pt_pixels = render_viewport_pt(
            &gpu,
            &bvh,
            &ViewportPtSettings {
                eye,
                forward,
                right,
                up,
                width: 64,
                height: 64,
                spp: 8,
                max_bounces: 1,
                seed: 1,
                mode: 0,
                bounds_min: scene.bounds_min,
                bounds_max: scene.bounds_max,
                resolution: UVec3::new(2, 2, 2),
                irradiance_voxels: None,
                sun_dir: None,
                sun_color: None,
                sun_intensity: None,
                env_map: Some(env),
                env_intensity: 1.0,
                env_rotation_y: 0.0,
            },
        )
        .expect("HDRI viewport PT");
        let warm_floor = pt_pixels
            .chunks(4)
            .filter(|p| p[0] > p[2] + 20 && p[0] > 80)
            .count();
        assert!(warm_floor > 50, "Expected warm HDRI-lit pixels in PT preview, got {warm_floor}");
    }
}


