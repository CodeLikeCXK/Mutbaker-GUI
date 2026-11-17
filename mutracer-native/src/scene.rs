use anyhow::{Context, Result};
use glam::{Mat4, Vec3, Vec4};
use half::f16;
use image::ImageReader;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::path::Path;

pub const STANDARD_TEX_SIZE: u32 = 512;

#[derive(Debug, Clone)]
pub struct Triangle {
    pub p0: Vec3,
    pub p1: Vec3,
    pub p2: Vec3,
    pub n0: Vec3,
    pub n1: Vec3,
    pub n2: Vec3,
    pub uv0: [f32; 2],
    pub uv1: [f32; 2],
    pub uv2: [f32; 2],
    pub material_id: u32,
}

impl Triangle {
    pub fn centroid(&self) -> Vec3 {
        (self.p0 + self.p1 + self.p2) * (1.0 / 3.0)
    }

    pub fn aabb(&self) -> (Vec3, Vec3) {
        let min = self.p0.min(self.p1).min(self.p2);
        let max = self.p0.max(self.p1).max(self.p2);
        (min, max)
    }
}

#[derive(Debug, Clone)]
pub struct Material {
    pub base_color: Vec4,
    pub emissive: Vec4, // rgb * strength, w = unlit flag
    pub roughness_metallic: Vec4, // x = roughness, y = metallic
    pub tex_slice: i32, // index into 2D texture array, -1 if untextured
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: Vec4::new(0.8, 0.8, 0.8, 1.0),
            emissive: Vec4::ZERO,
            roughness_metallic: Vec4::new(0.5, 0.0, 1.5, 0.0),
            tex_slice: -1,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DirectionalLightData {
    pub direction: Vec3, // Unit vector pointing towards the light source
    pub color: Vec3,
    pub intensity: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct PointLightData {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32, // Luminous intensity in candela (lm/sr) per KHR_lights_punctual
    pub range: Option<f32>,
}

pub struct Scene {
    pub triangles: Vec<Triangle>,
    pub materials: Vec<Material>,
    pub texture_array: Vec<u8>,
    pub texture_slices: u32,
    pub texture_size: u32,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub directional_light: Option<DirectionalLightData>,
    pub point_lights: Vec<PointLightData>,
}

fn sample_pixel_rgba_u8(data: &[u8], format: gltf::image::Format, x: usize, y: usize, w: usize) -> [u8; 4] {
    match format {
        gltf::image::Format::R8G8B8 => {
            let offset = (y * w + x) * 3;
            if offset + 2 < data.len() {
                [data[offset], data[offset + 1], data[offset + 2], 255]
            } else {
                [255, 255, 255, 255]
            }
        }
        gltf::image::Format::R8G8B8A8 => {
            let offset = (y * w + x) * 4;
            if offset + 3 < data.len() {
                [data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]
            } else {
                [255, 255, 255, 255]
            }
        }
        gltf::image::Format::R8G8 => {
            let offset = (y * w + x) * 2;
            if offset + 1 < data.len() {
                [data[offset], data[offset + 1], 0, 255]
            } else {
                [255, 255, 255, 255]
            }
        }
        gltf::image::Format::R8 => {
            let offset = y * w + x;
            if offset < data.len() {
                let v = data[offset];
                [v, v, v, 255]
            } else {
                [255, 255, 255, 255]
            }
        }
        _ => [255, 255, 255, 255],
    }
}

fn resample_image_to_rgba8(
    img: &gltf::image::Data,
    target_dim: usize,
) -> (Vec<u8>, [f32; 3]) {
    let mut out = vec![255u8; target_dim * target_dim * 4];
    let src_w = (img.width as usize).max(1);
    let src_h = (img.height as usize).max(1);

    let mut sum_r = 0.0f32;
    let mut sum_g = 0.0f32;
    let mut sum_b = 0.0f32;
    let mut count = 0.0f32;

    for ty in 0..target_dim {
        let sy = (ty * src_h) / target_dim;
        for tx in 0..target_dim {
            let sx = (tx * src_w) / target_dim;
            let px = sample_pixel_rgba_u8(&img.pixels, img.format, sx, sy, src_w);
            let out_idx = (ty * target_dim + tx) * 4;
            out[out_idx] = px[0];
            out[out_idx + 1] = px[1];
            out[out_idx + 2] = px[2];
            out[out_idx + 3] = px[3];

            // Sample linear color contribution
            let r_lin = ((px[0] as f32) / 255.0).powf(2.2);
            let g_lin = ((px[1] as f32) / 255.0).powf(2.2);
            let b_lin = ((px[2] as f32) / 255.0).powf(2.2);
            sum_r += r_lin;
            sum_g += g_lin;
            sum_b += b_lin;
            count += 1.0;
        }
    }

    let avg = if count > 0.0 {
        [sum_r / count, sum_g / count, sum_b / count]
    } else {
        [1.0, 1.0, 1.0]
    };

    (out, avg)
}

impl Scene {
    pub fn load_gltf<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let (document, buffers, images) = gltf::import(path)
            .with_context(|| format!("Failed to import gltf file: {}", path.display()))?;

        let mut image_slice_map: HashMap<usize, i32> = HashMap::new();
        let mut texture_array_slices: Vec<u8> = Vec::new();
        let mut next_slice_idx: i32 = 0;

        let mut materials = Vec::new();
        for mat in document.materials() {
            let pbr = mat.pbr_metallic_roughness();
            let mut base_color_factor = Vec4::from_array(pbr.base_color_factor());
            let emissive_factor = mat.emissive_factor();
            let emissive_strength = mat.emissive_strength().unwrap_or(1.0);
            let emissive = Vec4::new(
                emissive_factor[0] * emissive_strength,
                emissive_factor[1] * emissive_strength,
                emissive_factor[2] * emissive_strength,
                0.0,
            );
            let roughness = pbr.roughness_factor();
            let metallic = pbr.metallic_factor();

            let mut tex_slice = -1;
            if let Some(tex_info) = pbr.base_color_texture() {
                let img_idx = tex_info.texture().source().index();
                if let Some(&cached_slice) = image_slice_map.get(&img_idx) {
                    tex_slice = cached_slice;
                } else if let Some(img_data) = images.get(img_idx) {
                    let (slice_bytes, avg_color) = resample_image_to_rgba8(img_data, STANDARD_TEX_SIZE as usize);
                    texture_array_slices.extend_from_slice(&slice_bytes);
                    tex_slice = next_slice_idx;
                    image_slice_map.insert(img_idx, next_slice_idx);
                    next_slice_idx += 1;

                    // Tint base_color_factor by average linear color for accurate bounce lighting
                    base_color_factor.x *= avg_color[0];
                    base_color_factor.y *= avg_color[1];
                    base_color_factor.z *= avg_color[2];
                }
            }

            materials.push(Material {
                base_color: base_color_factor,
                emissive,
                roughness_metallic: Vec4::new(roughness, metallic, 1.5, 0.0),
                tex_slice,
            });
        }

        if materials.is_empty() {
            materials.push(Material::default());
        }

        // Fallback texture for scenes without textures (e.g. Cornell Box)
        let (texture_array, texture_slices, texture_size) = if next_slice_idx == 0 {
            (vec![255u8, 255u8, 255u8, 255u8], 1, 1)
        } else {
            (texture_array_slices, next_slice_idx as u32, STANDARD_TEX_SIZE)
        };

        let mut triangles = Vec::new();
        let mut scene_min = Vec3::splat(f32::INFINITY);
        let mut scene_max = Vec3::splat(f32::NEG_INFINITY);
        let mut directional_light: Option<DirectionalLightData> = None;
        let mut point_lights: Vec<PointLightData> = Vec::new();

        for scene in document.scenes() {
            for node in scene.nodes() {
                load_node_recursive(
                    &node,
                    Mat4::IDENTITY,
                    &buffers,
                    materials.len(),
                    &mut triangles,
                    &mut scene_min,
                    &mut scene_max,
                    &mut directional_light,
                    &mut point_lights,
                );
            }
        }

        if triangles.is_empty() {
            anyhow::bail!("No mesh triangles found in gltf scene: {}", path.display());
        }

        eprintln!(
            "[scene] Loaded {} triangles, {} materials, {} texture slices ({}x{})",
            triangles.len(),
            materials.len(),
            texture_slices,
            texture_size,
            texture_size
        );
        eprintln!(
            "[scene] Bounds: min [{}, {}, {}], max [{}, {}, {}]",
            scene_min.x, scene_min.y, scene_min.z, scene_max.x, scene_max.y, scene_max.z
        );
        if let Some(ref sun) = directional_light {
            eprintln!(
                "[scene] Found glTF DirectionalLight: dir {:?}, color {:?}, intensity {}",
                sun.direction, sun.color, sun.intensity
            );
        }
        if !point_lights.is_empty() {
            eprintln!("[scene] Found {} glTF PointLight(s)", point_lights.len());
        }

        Ok(Self {
            triangles,
            materials,
            texture_array,
            texture_slices,
            texture_size,
            bounds_min: scene_min,
            bounds_max: scene_max,
            directional_light,
            point_lights,
        })
    }
}

fn load_node_recursive(
    node: &gltf::Node,
    parent_transform: Mat4,
    buffers: &[gltf::buffer::Data],
    num_materials: usize,
    out_triangles: &mut Vec<Triangle>,
    scene_min: &mut Vec3,
    scene_max: &mut Vec3,
    directional_light: &mut Option<DirectionalLightData>,
    point_lights: &mut Vec<PointLightData>,
) {
    let local_transform = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world_transform = parent_transform * local_transform;
    let normal_matrix = world_transform.inverse().transpose();

    if let Some(light) = node.light() {
        match light.kind() {
            gltf::khr_lights_punctual::Kind::Directional => {
                if directional_light.is_none() {
                    let sun_dir = world_transform.transform_vector3(Vec3::Z).normalize_or_zero();
                    let color = Vec3::from_array(light.color());
                    let intensity = light.intensity();
                    *directional_light = Some(DirectionalLightData {
                        direction: if sun_dir.length_squared() > 1e-4 {
                            sun_dir
                        } else {
                            Vec3::new(0.32, 0.81, 0.49).normalize()
                        },
                        color,
                        intensity,
                    });
                }
            }
            gltf::khr_lights_punctual::Kind::Point => {
                let position = world_transform.transform_point3(Vec3::ZERO);
                let color = Vec3::from_array(light.color());
                let intensity = light.intensity();
                let max_scale = world_transform
                    .x_axis
                    .truncate()
                    .length()
                    .max(world_transform.y_axis.truncate().length())
                    .max(world_transform.z_axis.truncate().length());
                let range = light.range().map(|r| {
                    if max_scale > 1e-6 {
                        r * max_scale
                    } else {
                        r
                    }
                });
                point_lights.push(PointLightData {
                    position,
                    color,
                    intensity,
                    range,
                });
            }
            _ => {}
        }
    }

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|b| Some(&buffers[b.index()]));

            let positions: Vec<Vec3> = match reader.read_positions() {
                Some(iter) => iter.map(|p| world_transform.transform_point3(Vec3::from_array(p))).collect(),
                None => continue,
            };

            let normals: Vec<Vec3> = match reader.read_normals() {
                Some(iter) => iter
                    .map(|n| normal_matrix.transform_vector3(Vec3::from_array(n)).normalize_or_zero())
                    .collect(),
                None => vec![Vec3::Y; positions.len()],
            };

            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

            let material_id = primitive
                .material()
                .index()
                .map(|i| (i as u32).min((num_materials.saturating_sub(1)) as u32))
                .unwrap_or(0);

            if let Some(indices) = reader.read_indices() {
                let idx_vec: Vec<u32> = indices.into_u32().collect();
                for chunk in idx_vec.chunks_exact(3) {
                    let i0 = chunk[0] as usize;
                    let i1 = chunk[1] as usize;
                    let i2 = chunk[2] as usize;
                    if i0 < positions.len() && i1 < positions.len() && i2 < positions.len() {
                        let p0 = positions[i0];
                        let p1 = positions[i1];
                        let p2 = positions[i2];

                        let mut n0 = normals.get(i0).copied().unwrap_or(Vec3::Y);
                        let mut n1 = normals.get(i1).copied().unwrap_or(Vec3::Y);
                        let mut n2 = normals.get(i2).copied().unwrap_or(Vec3::Y);

                        if n0.length_squared() < 1e-4 {
                            let geo_normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
                            n0 = geo_normal;
                            n1 = geo_normal;
                            n2 = geo_normal;
                        }

                        let uv0 = uvs.get(i0).copied().unwrap_or([0.0, 0.0]);
                        let uv1 = uvs.get(i1).copied().unwrap_or([0.0, 0.0]);
                        let uv2 = uvs.get(i2).copied().unwrap_or([0.0, 0.0]);

                        *scene_min = scene_min.min(p0).min(p1).min(p2);
                        *scene_max = scene_max.max(p0).max(p1).max(p2);

                        out_triangles.push(Triangle {
                            p0,
                            p1,
                            p2,
                            n0,
                            n1,
                            n2,
                            uv0,
                            uv1,
                            uv2,
                            material_id,
                        });
                    }
                }
            }
        }
    }

    for child in node.children() {
        load_node_recursive(
            &child,
            world_transform,
            buffers,
            num_materials,
            out_triangles,
            scene_min,
            scene_max,
            directional_light,
            point_lights,
        );
    }
}

const MAX_ENV_TEX_WIDTH: u32 = 2048;
const MAX_ENV_TEX_HEIGHT: u32 = 1024;
const MAX_CDF_WIDTH: u32 = 512;
const MAX_CDF_HEIGHT: u32 = 256;

#[derive(Debug, Clone)]
pub struct EnvironmentMap {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// Equirectangular Rgba16Float pixels (4 x u16 f16 bits per pixel, row-major top-to-bottom).
    pub rgba16f_data: Vec<u16>,
    pub cdf_width: u32,
    pub cdf_height: u32,
    /// Marginal CDF over rows (length = cdf_height + 1).
    pub marginal_cdf: Vec<f32>,
    /// Conditional CDF per row (length = cdf_height * (cdf_width + 1)).
    pub conditional_cdf: Vec<f32>,
    pub has_importance_cdf: bool,
}

#[inline]
fn sanitize_hdr_channel(v: f32) -> f32 {
    if v.is_finite() && v > 0.0 {
        v.min(65504.0)
    } else {
        0.0
    }
}

#[inline]
fn rotate_dir_y(dir: Vec3, angle_rad: f32) -> Vec3 {
    if angle_rad.abs() < 1e-6 {
        return dir;
    }
    let (s, c) = angle_rad.sin_cos();
    Vec3::new(dir.x * c + dir.z * s, dir.y, -dir.x * s + dir.z * c)
}

#[inline]
pub fn dir_to_equirect_uv(world_dir: Vec3, rotation_y_rad: f32) -> (f32, f32) {
    let dir = rotate_dir_y(world_dir, -rotation_y_rad).normalize_or_zero();
    let u = dir.z.atan2(dir.x) / (2.0 * PI) + 0.5;
    let v = dir.y.clamp(-1.0, 1.0).acos() / PI;
    (u, v)
}

fn downsample_rgb_box(
    src: &[[f32; 3]],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Vec<[f32; 3]> {
    let mut dst = Vec::with_capacity((dst_w * dst_h) as usize);
    for dy in 0..dst_h {
        let y0 = ((dy as u64 * src_h as u64) / dst_h as u64) as u32;
        let y1 = ((((dy + 1) as u64 * src_h as u64) + dst_h as u64 - 1) / dst_h as u64)
            .min(src_h as u64) as u32;
        let y1 = y1.max(y0 + 1);
        for dx in 0..dst_w {
            let x0 = ((dx as u64 * src_w as u64) / dst_w as u64) as u32;
            let x1 = ((((dx + 1) as u64 * src_w as u64) + dst_w as u64 - 1) / dst_w as u64)
                .min(src_w as u64) as u32;
            let x1 = x1.max(x0 + 1);

            let mut acc = [0.0f64; 3];
            let mut count = 0.0f64;
            for sy in y0..y1 {
                let row_off = (sy * src_w) as usize;
                for sx in x0..x1 {
                    let p = src[row_off + sx as usize];
                    acc[0] += p[0] as f64;
                    acc[1] += p[1] as f64;
                    acc[2] += p[2] as f64;
                    count += 1.0;
                }
            }
            let inv = if count > 0.0 { 1.0 / count } else { 1.0 };
            dst.push([
                (acc[0] * inv) as f32,
                (acc[1] * inv) as f32,
                (acc[2] * inv) as f32,
            ]);
        }
    }
    dst
}

impl EnvironmentMap {
    pub fn load_hdr<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let reader = ImageReader::open(path)
            .with_context(|| format!("Failed to open HDRI file: {}", path.display()))?
            .with_guessed_format()
            .with_context(|| format!("Failed to detect image format: {}", path.display()))?;
        let dyn_img = reader
            .decode()
            .with_context(|| format!("Failed to decode HDRI image: {}", path.display()))?;
        let rgb32f = dyn_img.into_rgb32f();
        let src_w = rgb32f.width();
        let src_h = rgb32f.height();
        if src_w == 0 || src_h == 0 {
            anyhow::bail!("HDRI image has zero dimensions: {}", path.display());
        }

        let raw_samples = rgb32f.as_raw();
        let mut pixels = Vec::with_capacity((src_w * src_h) as usize);
        for chunk in raw_samples.chunks_exact(3) {
            pixels.push([
                sanitize_hdr_channel(chunk[0]),
                sanitize_hdr_channel(chunk[1]),
                sanitize_hdr_channel(chunk[2]),
            ]);
        }

        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "environment.hdr".to_string());

        let (final_w, final_h, final_pixels) =
            if src_w > MAX_ENV_TEX_WIDTH || src_h > MAX_ENV_TEX_HEIGHT {
                let scale = (MAX_ENV_TEX_WIDTH as f32 / src_w as f32)
                    .min(MAX_ENV_TEX_HEIGHT as f32 / src_h as f32);
                let tw = ((src_w as f32 * scale).round() as u32).clamp(1, MAX_ENV_TEX_WIDTH);
                let th = ((src_h as f32 * scale).round() as u32).clamp(1, MAX_ENV_TEX_HEIGHT);
                eprintln!(
                    "[hdri] Downsampling '{}' from {}x{} to {}x{} for GPU upload",
                    name, src_w, src_h, tw, th
                );
                let ds = downsample_rgb_box(&pixels, src_w, src_h, tw, th);
                (tw, th, ds)
            } else {
                (src_w, src_h, pixels)
            };

        Self::from_rgb_f32(name, final_w, final_h, &final_pixels)
    }

    pub fn from_rgb_f32(
        name: impl Into<String>,
        width: u32,
        height: u32,
        rgb_pixels: &[[f32; 3]],
    ) -> Result<Self> {
        let expected = (width as usize) * (height as usize);
        if width == 0 || height == 0 || rgb_pixels.len() != expected {
            anyhow::bail!(
                "Invalid EnvironmentMap dimensions {}x{} for {} pixels",
                width,
                height,
                rgb_pixels.len()
            );
        }

        let mut sanitized = Vec::with_capacity(expected);
        let mut rgba16f_data = Vec::with_capacity(expected * 4);
        let one_f16 = f16::from_f32(1.0).to_bits();
        for p in rgb_pixels {
            let r = sanitize_hdr_channel(p[0]);
            let g = sanitize_hdr_channel(p[1]);
            let b = sanitize_hdr_channel(p[2]);
            sanitized.push([r, g, b]);
            rgba16f_data.push(f16::from_f32(r).to_bits());
            rgba16f_data.push(f16::from_f32(g).to_bits());
            rgba16f_data.push(f16::from_f32(b).to_bits());
            rgba16f_data.push(one_f16);
        }

        let cdf_width = width.min(MAX_CDF_WIDTH);
        let cdf_height = height.min(MAX_CDF_HEIGHT);
        let cdf_pixels = if cdf_width == width && cdf_height == height {
            sanitized
        } else {
            downsample_rgb_box(&sanitized, width, height, cdf_width, cdf_height)
        };

        let mut row_sums = vec![0.0f64; cdf_height as usize];
        let mut weights = vec![0.0f64; (cdf_width * cdf_height) as usize];
        let mut total_weight = 0.0f64;

        for cy in 0..cdf_height {
            let theta = ((cy as f64 + 0.5) / cdf_height as f64) * std::f64::consts::PI;
            let sin_theta = theta.sin().max(1e-6);
            let mut row_sum = 0.0f64;
            for cx in 0..cdf_width {
                let idx = (cy * cdf_width + cx) as usize;
                let p = cdf_pixels[idx];
                let lum = 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64;
                let w = lum * sin_theta;
                weights[idx] = w;
                row_sum += w;
            }
            row_sums[cy as usize] = row_sum;
            total_weight += row_sum;
        }

        let mut marginal_cdf = vec![0.0f32; (cdf_height + 1) as usize];
        let mut conditional_cdf = vec![0.0f32; (cdf_height * (cdf_width + 1)) as usize];
        let has_importance_cdf = total_weight > 1e-12;

        if has_importance_cdf {
            let mut m_acc = 0.0f64;
            for cy in 0..cdf_height as usize {
                m_acc += row_sums[cy] / total_weight;
                marginal_cdf[cy + 1] = m_acc.min(1.0) as f32;

                let row_off = cy * (cdf_width as usize + 1);
                let row_w = row_sums[cy];
                if row_w > 1e-12 {
                    let mut c_acc = 0.0f64;
                    for cx in 0..cdf_width as usize {
                        c_acc += weights[cy * cdf_width as usize + cx] / row_w;
                        conditional_cdf[row_off + cx + 1] = c_acc.min(1.0) as f32;
                    }
                    conditional_cdf[row_off + cdf_width as usize] = 1.0;
                } else {
                    for cx in 0..=cdf_width as usize {
                        conditional_cdf[row_off + cx] = cx as f32 / cdf_width as f32;
                    }
                }
            }
            marginal_cdf[cdf_height as usize] = 1.0;
        } else {
            for cy in 0..=cdf_height as usize {
                marginal_cdf[cy] = cy as f32 / cdf_height as f32;
            }
            for cy in 0..cdf_height as usize {
                let row_off = cy * (cdf_width as usize + 1);
                for cx in 0..=cdf_width as usize {
                    conditional_cdf[row_off + cx] = cx as f32 / cdf_width as f32;
                }
            }
        }

        let name = name.into();
        eprintln!(
            "[hdri] Loaded '{}' ({}x{}, CDF {}x{}, importance={})",
            name, width, height, cdf_width, cdf_height, has_importance_cdf
        );

        Ok(Self {
            name,
            width,
            height,
            rgba16f_data,
            cdf_width,
            cdf_height,
            marginal_cdf,
            conditional_cdf,
            has_importance_cdf,
        })
    }

    pub fn sample_bilinear_f32(&self, u: f32, v: f32) -> [f32; 3] {
        let u = u.rem_euclid(1.0);
        let v = v.clamp(0.0, 1.0);
        let fx = u * self.width as f32 - 0.5;
        let fy = v * self.height as f32 - 0.5;

        let x0 = (fx.floor() as i32).rem_euclid(self.width as i32) as u32;
        let x1 = (x0 + 1) % self.width;
        let y0 = (fy.floor() as i32).clamp(0, self.height as i32 - 1) as u32;
        let y1 = (y0 + 1).min(self.height - 1);

        let tx = fx - fx.floor();
        let ty = (fy - fy.floor()).clamp(0.0, 1.0);

        let read_px = |x: u32, y: u32| -> [f32; 3] {
            let idx = ((y * self.width + x) * 4) as usize;
            [
                f16::from_bits(self.rgba16f_data[idx]).to_f32(),
                f16::from_bits(self.rgba16f_data[idx + 1]).to_f32(),
                f16::from_bits(self.rgba16f_data[idx + 2]).to_f32(),
            ]
        };

        let c00 = read_px(x0, y0);
        let c10 = read_px(x1, y0);
        let c01 = read_px(x0, y1);
        let c11 = read_px(x1, y1);

        let mut out = [0.0f32; 3];
        for i in 0..3 {
            let top = c00[i] + (c10[i] - c00[i]) * tx;
            let bot = c01[i] + (c11[i] - c01[i]) * tx;
            out[i] = top + (bot - top) * ty;
        }
        out
    }

    /// Convert equirectangular HDRI into 6-layer WebGPU/Bevy Rgba16Float cubemap bytes
    /// (+X, -X, +Y, -Y, +Z, -Z), with intensity and Y rotation baked in.
    pub fn to_cubemap_rgba16f(&self, face_size: u32, intensity: f32, rotation_y_rad: f32) -> Vec<u8> {
        let face_pixels = (face_size * face_size) as usize;
        let mut u16_pixels = Vec::with_capacity(6 * face_pixels * 4);
        let one_f16 = f16::from_f32(1.0).to_bits();

        for face in 0..6u32 {
            for py in 0..face_size {
                let v = ((py as f32 + 0.5) / face_size as f32) * 2.0 - 1.0;
                for px in 0..face_size {
                    let u = ((px as f32 + 0.5) / face_size as f32) * 2.0 - 1.0;
                    let dir = match face {
                        0 => Vec3::new(1.0, -v, -u),  // +X
                        1 => Vec3::new(-1.0, -v, u),  // -X
                        2 => Vec3::new(u, 1.0, v),    // +Y
                        3 => Vec3::new(u, -1.0, -v),  // -Y
                        4 => Vec3::new(u, -v, 1.0),   // +Z
                        _ => Vec3::new(-u, -v, -1.0), // -Z
                    }
                    .normalize();

                    let (eu, ev) = dir_to_equirect_uv(dir, rotation_y_rad);
                    let rgb = self.sample_bilinear_f32(eu, ev);
                    u16_pixels.push(f16::from_f32((rgb[0] * intensity).clamp(0.0, 65504.0)).to_bits());
                    u16_pixels.push(f16::from_f32((rgb[1] * intensity).clamp(0.0, 65504.0)).to_bits());
                    u16_pixels.push(f16::from_f32((rgb[2] * intensity).clamp(0.0, 65504.0)).to_bits());
                    u16_pixels.push(one_f16);
                }
            }
        }

        bytemuck::cast_slice(&u16_pixels).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_map_cdf_and_cubemap() {
        // 4x2 synthetic HDRI: right half bright, left half dark
        let mut pixels = vec![[0.0, 0.0, 0.0]; 8];
        pixels[3] = [10.0, 5.0, 2.0];
        pixels[7] = [10.0, 5.0, 2.0];
        let env = EnvironmentMap::from_rgb_f32("test.hdr", 4, 2, &pixels).unwrap();
        assert!(env.has_importance_cdf);
        assert_eq!(env.marginal_cdf.len(), 3);
        assert!((env.marginal_cdf[2] - 1.0).abs() < 1e-5);
        assert_eq!(env.conditional_cdf.len(), 2 * 5);
        assert!((env.conditional_cdf[4] - 1.0).abs() < 1e-5);
        let cubemap = env.to_cubemap_rgba16f(8, 1.0, 0.0);
        assert_eq!(cubemap.len(), 6 * 8 * 8 * 8);
    }
}
