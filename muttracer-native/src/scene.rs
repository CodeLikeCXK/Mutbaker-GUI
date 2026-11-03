use anyhow::{Context, Result};
use glam::{Mat4, Vec3, Vec4};
use std::collections::HashMap;
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

pub struct Scene {
    pub triangles: Vec<Triangle>,
    pub materials: Vec<Material>,
    pub texture_array: Vec<u8>,
    pub texture_slices: u32,
    pub texture_size: u32,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
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

        Ok(Self {
            triangles,
            materials,
            texture_array,
            texture_slices,
            texture_size,
            bounds_min: scene_min,
            bounds_max: scene_max,
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
) {
    let local_transform = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world_transform = parent_transform * local_transform;
    let normal_matrix = world_transform.inverse().transpose();

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
        );
    }
}
