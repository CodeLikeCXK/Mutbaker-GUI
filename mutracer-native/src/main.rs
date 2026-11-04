use anyhow::{Context, Result};
use glam::{UVec3, Vec3};
use mutracer_native::baker::{bake_irradiance_volume, BakeSettings};
use mutracer_native::bvh::FlatBvh;
use mutracer_native::gpu::GpuContext;
use mutracer_native::scene::Scene;
use std::env;
use std::path::{Path, PathBuf};

fn print_usage() {
    eprintln!(
        r#"
MuTracer Native - Windows ARM64 DX12 Path Tracer & Irradiance Volume Baker
Usage:
  mutracer-native bake [options]
  mutracer-native probe-gpu
  mutracer-native verify-ktx2 <path.ktx2>

Bake Options:
  --scene <path.glb>          Path to glTF/GLB input scene (required)
  --out <path.vxgi.ktx2>      Output KTX2 3D texture file (default: irradiance.vxgi.ktx2)
  --resolution <rx,ry,rz>     Voxel resolution grid, e.g. 8,8,8 (default: 8,8,8)
  --spp <samples>             Samples per ambient cube face (default: 64)
  --bounces <count>           Max diffuse bounces (default: 2)
  --bounds <minX,minY,minZ,maxX,maxY,maxZ> Custom bounding box (default: scene AABB)
"#
    );
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    match args[1].as_str() {
        "probe-gpu" => {
            println!("Initializing DX12 on Windows ARM64...");
            let gpu = GpuContext::new_dx12()?;
            println!("Device initialized successfully: {}", gpu.adapter_info.name);
        }
        "verify-ktx2" => {
            if args.len() < 3 {
                eprintln!("Error: Missing ktx2 path");
                return Ok(());
            }
            verify_ktx2(&args[2])?;
        }
        "bake" => {
            run_bake(&args[2..])?;
        }
        _ => {
            print_usage();
        }
    }

    Ok(())
}

fn run_bake(args: &[String]) -> Result<()> {
    let mut scene_path: Option<PathBuf> = None;
    let mut out_path = PathBuf::from("irradiance.vxgi.ktx2");
    let mut resolution = UVec3::new(8, 8, 8);
    let mut spp = 64u32;
    let mut max_bounces = 2u32;
    let mut custom_bounds: Option<(Vec3, Vec3)> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--scene" => {
                i += 1;
                if i < args.len() {
                    scene_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--out" => {
                i += 1;
                if i < args.len() {
                    out_path = PathBuf::from(&args[i]);
                }
            }
            "--resolution" => {
                i += 1;
                if i < args.len() {
                    let parts: Vec<u32> = args[i]
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    if parts.len() == 3 {
                        resolution = UVec3::new(parts[0], parts[1], parts[2]);
                    }
                }
            }
            "--spp" => {
                i += 1;
                if i < args.len() {
                    if let Ok(v) = args[i].parse() {
                        spp = v;
                    }
                }
            }
            "--bounces" => {
                i += 1;
                if i < args.len() {
                    if let Ok(v) = args[i].parse() {
                        max_bounces = v;
                    }
                }
            }
            "--bounds" => {
                i += 1;
                if i < args.len() {
                    let parts: Vec<f32> = args[i]
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    if parts.len() == 6 {
                        custom_bounds = Some((
                            Vec3::new(parts[0], parts[1], parts[2]),
                            Vec3::new(parts[3], parts[4], parts[5]),
                        ));
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }

    let scene_path = scene_path.context("Missing required argument --scene <path.glb>")?;

    println!("============================================================");
    println!("MuTracer Native - Irradiance Volume Baker (DX12 ARM64)");
    println!("============================================================");

    let gpu = GpuContext::new_dx12()?;
    let scene = Scene::load_gltf(&scene_path)?;
    let bvh = FlatBvh::build(&scene);

    let (b_min, b_max) = custom_bounds.unwrap_or_else(|| {
        let size = scene.bounds_max - scene.bounds_min;
        let margin = size * 0.05; // 5% padding
        (scene.bounds_min - margin, scene.bounds_max + margin)
    });

    let settings = BakeSettings {
        resolution,
        bounds_min: b_min,
        bounds_max: b_max,
        spp,
        max_bounces,
    };

    let result = bake_irradiance_volume(&gpu, &bvh, &settings)?;

    // Write KTX2 file
    std::fs::write(&out_path, &result.ktx2_bytes)
        .with_context(|| format!("Failed to write output to {}", out_path.display()))?;

    // Calculate Bevy Transform
    let center = (b_min + b_max) * 0.5;
    let size = b_max - b_min;

    let meta = serde_json::json!({
        "voxels_texture": out_path.file_name().unwrap_or_default().to_string_lossy(),
        "voxel_resolution": [result.resolution.x, result.resolution.y, result.resolution.z],
        "texture_3d_dimensions": [result.texture_dimensions.x, result.texture_dimensions.y, result.texture_dimensions.z],
        "format": "VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 (Rgb9e5Ufloat)",
        "bounds_min": [b_min.x, b_min.y, b_min.z],
        "bounds_max": [b_max.x, b_max.y, b_max.z],
        "bevy_transform": {
            "translation": [center.x, center.y, center.z],
            "scale": [size.x, size.y, size.z]
        },
        "bevy_usage_example": format!(
            "commands.spawn((\n    Transform::from_xyz({:.4}, {:.4}, {:.4}).with_scale(Vec3::new({:.4}, {:.4}, {:.4})),\n    IrradianceVolume {{\n        voxels: asset_server.load(\"{}\"),\n        intensity: 1.0,\n        ..default()\n    }},\n));",
            center.x, center.y, center.z, size.x, size.y, size.z, out_path.display()
        )
    });

    let meta_path = out_path.with_extension("meta.json");
    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;

    println!("============================================================");
    println!("Bake completed successfully in {:.2?}!", result.duration);
    println!("Output 3D texture: {} ({} bytes)", out_path.display(), result.ktx2_bytes.len());
    println!("Output metadata:   {}", meta_path.display());
    println!("\nBevy IrradianceVolume Spawn Code:\n{}", meta["bevy_usage_example"].as_str().unwrap());
    println!("============================================================");

    Ok(())
}

fn verify_ktx2<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?;
    println!("Verifying KTX2: {} ({} bytes)", path.display(), bytes.len());

    let reader = ktx2::Reader::new(&bytes).map_err(|e| anyhow::anyhow!("KTX2 parse error: {e:?}"))?;
    let header = reader.header();
    println!("  Format: {:?}", header.format);
    println!("  Dimensions: {} x {} x {}", header.pixel_width, header.pixel_height, header.pixel_depth);
    println!("  Levels: {}, Layers: {}, Faces: {}", header.level_count, header.layer_count, header.face_count);
    println!("  Supercompression: {:?}", header.supercompression_scheme);

    let level0 = reader.levels().next().context("No level 0 found in KTX2")?;
    println!("  Level 0 data size: {} bytes", level0.data.len());
    let expected = header.pixel_width * header.pixel_height * header.pixel_depth * 4;
    assert_eq!(level0.data.len() as u32, expected, "Data length should match width*height*depth*4");
    println!("KTX2 verification PASSED!");

    Ok(())
}
