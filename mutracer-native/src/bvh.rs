use crate::scene::Triangle;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuBvhNode {
    pub min: [f32; 3],
    pub left_or_first_tri: u32,
    pub max: [f32; 3],
    pub tri_count: u32, // 0 for interior, >0 for leaf
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuTriangle {
    pub p0: [f32; 3],
    pub material_idx: u32,
    pub p1: [f32; 3],
    pub flags: u32,
    pub p2: [f32; 3],
    pub _pad0: u32,
    pub n0: [f32; 3],
    pub _pad1: u32,
    pub n1: [f32; 3],
    pub _pad2: u32,
    pub n2: [f32; 3],
    pub _pad3: u32,
    pub uv0: [f32; 2],
    pub uv1: [f32; 2],
    pub uv2: [f32; 2],
    pub _pad4: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuMaterial {
    pub base_color: [f32; 4],
    pub emissive: [f32; 4],
    pub roughness_metallic: [f32; 4],
    pub tex_index: [i32; 4],
}

pub struct FlatBvh {
    pub nodes: Vec<GpuBvhNode>,
    pub triangles: Vec<GpuTriangle>,
    pub materials: Vec<GpuMaterial>,
    pub texture_array: Vec<u8>,
    pub texture_slices: u32,
    pub texture_size: u32,
}

struct BuildTriangle {
    tri: Triangle,
    centroid: Vec3,
    min: Vec3,
    max: Vec3,
}

impl FlatBvh {
    pub fn build(scene: &crate::scene::Scene) -> Self {
        let mut build_tris: Vec<BuildTriangle> = scene
            .triangles
            .iter()
            .map(|t| {
                let (min, max) = t.aabb();
                BuildTriangle {
                    tri: t.clone(),
                    centroid: t.centroid(),
                    min,
                    max,
                }
            })
            .collect();

        let mut nodes = Vec::new();
        let mut ordered_tris = Vec::with_capacity(build_tris.len());

        // Allocate root node at index 0
        nodes.push(GpuBvhNode {
            min: [0.0; 3],
            left_or_first_tri: 0,
            max: [0.0; 3],
            tri_count: 0,
        });

        build_into_node(0, &mut build_tris, &mut nodes, &mut ordered_tris);

        let materials: Vec<GpuMaterial> = scene
            .materials
            .iter()
            .map(|m| GpuMaterial {
                base_color: m.base_color.to_array(),
                emissive: m.emissive.to_array(),
                roughness_metallic: m.roughness_metallic.to_array(),
                tex_index: [m.tex_slice, 0, 0, 0],
            })
            .collect();

        eprintln!(
            "[bvh] Built Flat BVH: {} nodes, {} triangles, {} materials, {} texture slices",
            nodes.len(),
            ordered_tris.len(),
            materials.len(),
            scene.texture_slices
        );

        Self {
            nodes,
            triangles: ordered_tris,
            materials,
            texture_array: scene.texture_array.clone(),
            texture_slices: scene.texture_slices,
            texture_size: scene.texture_size,
        }
    }
}

fn build_into_node(
    node_idx: usize,
    tris: &mut [BuildTriangle],
    out_nodes: &mut Vec<GpuBvhNode>,
    out_ordered_tris: &mut Vec<GpuTriangle>,
) {
    let mut b_min = Vec3::splat(f32::INFINITY);
    let mut b_max = Vec3::splat(f32::NEG_INFINITY);
    let mut c_min = Vec3::splat(f32::INFINITY);
    let mut c_max = Vec3::splat(f32::NEG_INFINITY);

    for t in tris.iter() {
        b_min = b_min.min(t.min);
        b_max = b_max.max(t.max);
        c_min = c_min.min(t.centroid);
        c_max = c_max.max(t.centroid);
    }

    out_nodes[node_idx].min = b_min.to_array();
    out_nodes[node_idx].max = b_max.to_array();

    let tri_count = tris.len();
    const MAX_LEAF_TRIS: usize = 4;

    if tri_count <= MAX_LEAF_TRIS {
        let first_tri = out_ordered_tris.len() as u32;
        for t in tris.iter() {
            out_ordered_tris.push(GpuTriangle {
                p0: t.tri.p0.to_array(),
                material_idx: t.tri.material_id,
                p1: t.tri.p1.to_array(),
                flags: 0,
                p2: t.tri.p2.to_array(),
                _pad0: 0,
                n0: t.tri.n0.to_array(),
                _pad1: 0,
                n1: t.tri.n1.to_array(),
                _pad2: 0,
                n2: t.tri.n2.to_array(),
                _pad3: 0,
                uv0: t.tri.uv0,
                uv1: t.tri.uv1,
                uv2: t.tri.uv2,
                _pad4: [0.0, 0.0],
            });
        }
        out_nodes[node_idx].left_or_first_tri = first_tri;
        out_nodes[node_idx].tri_count = tri_count as u32;
        return;
    }

    // Split along the axis with largest centroid extent
    let extent = c_max - c_min;
    let axis = if extent.x > extent.y && extent.x > extent.z {
        0
    } else if extent.y > extent.z {
        1
    } else {
        2
    };

    let mid = tris.len() / 2;
    tris.select_nth_unstable_by(mid, |a, b| {
        let ca = match axis {
            0 => a.centroid.x,
            1 => a.centroid.y,
            _ => a.centroid.z,
        };
        let cb = match axis {
            0 => b.centroid.x,
            1 => b.centroid.y,
            _ => b.centroid.z,
        };
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let left_child_idx = out_nodes.len();
    out_nodes.push(GpuBvhNode {
        min: [0.0; 3],
        left_or_first_tri: 0,
        max: [0.0; 3],
        tri_count: 0,
    });
    out_nodes.push(GpuBvhNode {
        min: [0.0; 3],
        left_or_first_tri: 0,
        max: [0.0; 3],
        tri_count: 0,
    });

    out_nodes[node_idx].left_or_first_tri = left_child_idx as u32;
    out_nodes[node_idx].tri_count = 0; // interior

    let (left_tris, right_tris) = tris.split_at_mut(mid);
    build_into_node(left_child_idx, left_tris, out_nodes, out_ordered_tris);
    build_into_node(left_child_idx + 1, right_tris, out_nodes, out_ordered_tris);
}
