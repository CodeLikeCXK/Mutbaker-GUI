use bevy::core_pipeline::tonemapping::{DebandDither, Tonemapping};
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::light::{IrradianceVolume, LightProbe};
use bevy::picking::events::{PointerClick, PointerOut, PointerOver};
use bevy::prelude::*;
use bevy::asset::RenderAssetUsages;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use glam::UVec3;
use muttracer_native::baker::{bake_irradiance_volume, BakeResult, BakeSettings};
use muttracer_native::bvh::FlatBvh;
use muttracer_native::gpu::GpuContext;
use muttracer_native::scene::Scene as NativeScene;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Mutex;

// MuTracer Dark Theme Color Tokens
const BG_DARK: Color = Color::srgb(0.067, 0.071, 0.078); // #111214
const PANEL_DARK: Color = Color::srgb(0.094, 0.098, 0.106); // #18191b
const BORDER_DARK: Color = Color::srgb(0.204, 0.216, 0.251); // #343740
const TEXT_LIGHT: Color = Color::srgb(0.910, 0.914, 0.929); // #e8e9ed
const TEXT_MUTED: Color = Color::srgb(0.545, 0.569, 0.612); // #8b919c
const ACCENT_ORANGE: Color = Color::srgb(0.925, 0.588, 0.451); // #ec9673
const ACCENT_BUTTON: Color = Color::srgb(0.722, 0.278, 0.145); // #b84725
const BUTTON_NORMAL: Color = Color::srgb(0.165, 0.173, 0.200); // #2a2c33
const BUTTON_HOVER: Color = Color::srgb(0.240, 0.252, 0.290); // #3d404a

#[derive(Clone)]
pub struct SceneEntry {
    pub display_name: &'static str,
    pub asset_path: &'static str,
    pub visual_scale: f32,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub cam_center: Vec3,
    pub cam_dist: f32,
    pub triangle_count: usize,
}

pub const SCENES: &[SceneEntry] = &[
    SceneEntry {
        display_name: "Cornell Box",
        asset_path: "models/CornellBox/CornellBox.glb",
        visual_scale: 0.015,
        bounds_min: Vec3::new(-556.0, 0.0, -559.2),
        bounds_max: Vec3::new(0.0, 548.8, 0.0),
        cam_center: Vec3::new(-4.17, 4.11, -4.19),
        cam_dist: 16.0,
        triangle_count: 36,
    },
    SceneEntry {
        display_name: "Irradiance Volume Example",
        asset_path: "models/IrradianceVolumeExample/IrradianceVolumeExample.glb",
        visual_scale: 1.0,
        bounds_min: Vec3::new(-3.3, 0.0, -1.5),
        bounds_max: Vec3::new(3.3, 3.0, 1.5),
        cam_center: Vec3::new(0.0, 1.5, 0.0),
        cam_dist: 8.0,
        triangle_count: 24,
    },
    SceneEntry {
        display_name: "Cubes Scene",
        asset_path: "models/cubes/Cubes.glb",
        visual_scale: 1.0,
        bounds_min: Vec3::new(-5.0, 0.0, -5.0),
        bounds_max: Vec3::new(5.0, 5.0, 5.0),
        cam_center: Vec3::new(0.0, 2.5, 0.0),
        cam_dist: 15.0,
        triangle_count: 120,
    },
];

#[derive(Resource)]
pub struct AppState {
    pub scene_name: String,
    pub scene_path: String,
    pub triangle_count: usize,
    pub resolution: UVec3,
    pub spp: u32,
    pub bounces: u32,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub visual_scale: f32,
    pub show_probe_grid: bool,
    pub use_baked_raster: bool,
    pub is_baking: bool,
    pub status_text: String,
    pub baked_result: Option<BakeResult>,
    pub bake_tx: Sender<BakeTask>,
    pub bake_rx: Mutex<Receiver<BakeTaskResult>>,
}

pub struct BakeTask {
    pub scene_path: PathBuf,
    pub settings: BakeSettings,
}

pub enum BakeTaskResult {
    Success(BakeResult),
    Error(String),
}

#[derive(Resource)]
pub struct ImportChannel {
    pub tx: Sender<PathBuf>,
    pub rx: Mutex<Receiver<PathBuf>>,
}

#[derive(Resource)]
pub struct ProbeAssets {
    pub mesh: Handle<Mesh>,
    pub material: Handle<StandardMaterial>,
}

#[derive(Component)]
pub struct ProbeSphere;

#[derive(Component)]
pub struct BakedIrradianceVolume;

#[derive(Component)]
pub struct OrbitCamera {
    pub center: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
}

#[derive(Component, Clone)]
pub enum UiAction {
    ImportExternal,
    SelectScene(usize),
    SetResolution(u32, u32, u32),
    SetSpp(u32),
    SetBounces(u32),
    StartBake,
    ToggleBakedRaster,
    ToggleProbeGrid,
    ExportOutput,
}

#[derive(Component)]
pub struct AccentButton;

#[derive(Component)]
pub struct StatusLabel;

#[derive(Component)]
pub struct InfoLabel;

#[derive(Component)]
pub struct SceneModelRoot;

fn find_assets_dir() -> String {
    let candidates = [
        "assets",
        "mutbaker-gui/assets",
        "C:/tracer-experiment/mutbaker-gui/assets",
        "C:/Bevy/assets",
    ];
    for path in &candidates {
        if Path::new(path).exists() {
            return path.to_string();
        }
    }
    "assets".to_string()
}

fn main() {
    let (bake_task_tx, bake_task_rx) = channel::<BakeTask>();
    let (bake_res_tx, bake_res_rx) = channel::<BakeTaskResult>();
    let (import_tx, import_rx) = channel::<PathBuf>();

    // Background worker for GPU baking
    std::thread::spawn(move || {
        while let Ok(task) = bake_task_rx.recv() {
            match run_gpu_bake(task) {
                Ok(res) => {
                    let _ = bake_res_tx.send(BakeTaskResult::Success(res));
                }
                Err(e) => {
                    let _ = bake_res_tx.send(BakeTaskResult::Error(format!("{e:#}")));
                }
            }
        }
    });

    let default_scene = &SCENES[0];

    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Mutbaker-GUI | MuTracer Native (Windows ARM64 DX12)".into(),
                        resolution: (1280, 800).into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(AssetPlugin {
                    file_path: find_assets_dir(),
                    ..default()
                }),
        )
        .insert_resource(ClearColor(BG_DARK))
        .insert_resource(ImportChannel {
            tx: import_tx,
            rx: Mutex::new(import_rx),
        })
        .insert_resource(AppState {
            scene_name: default_scene.display_name.to_string(),
            scene_path: default_scene.asset_path.to_string(),
            triangle_count: default_scene.triangle_count,
            resolution: UVec3::new(4, 4, 4),
            spp: 32,
            bounces: 2,
            bounds_min: default_scene.bounds_min,
            bounds_max: default_scene.bounds_max,
            visual_scale: default_scene.visual_scale,
            show_probe_grid: true,
            use_baked_raster: false,
            is_baking: false,
            status_text: "Ready | Qualcomm Adreno 8cx Gen 3 (DX12)".into(),
            baked_result: None,
            bake_tx: bake_task_tx,
            bake_rx: Mutex::new(bake_res_rx),
        })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                orbit_camera_controller,
                sync_probe_spheres,
                draw_probe_gizmos,
                sync_baked_irradiance_volume,
                update_button_highlights,
                poll_bake_results,
                poll_import_dialog,
                update_labels,
            ),
        )
        .add_observer(handle_ui_click)
        .add_observer(handle_button_over)
        .add_observer(handle_button_out)
        .run();
}

fn run_gpu_bake(task: BakeTask) -> anyhow::Result<BakeResult> {
    let gpu = GpuContext::new_dx12()?;
    let scene = NativeScene::load_gltf(&task.scene_path)?;
    let bvh = FlatBvh::build(&scene);
    bake_irradiance_volume(&gpu, &bvh, &task.settings)
}

fn setup(
    mut commands: Commands,
    assets: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let sc = &SCENES[0];
    let cam = OrbitCamera {
        center: sc.cam_center,
        distance: sc.cam_dist,
        yaw: 0.8,
        pitch: 0.35,
    };
    let pos = cam.center
        + Vec3::new(
            cam.yaw.cos() * cam.pitch.cos(),
            cam.pitch.sin(),
            cam.yaw.sin() * cam.pitch.cos(),
        ) * cam.distance;

    // Use Reinhard tonemapping to bypass tonemapping LUT placeholder (prevents magenta render)
    commands.spawn((
        Camera3d::default(),
        Tonemapping::Reinhard,
        DebandDither::Disabled,
        Transform::from_translation(pos).looking_at(cam.center, Vec3::Y),
        cam,
        IsDefaultUiCamera,
    ));

    // Directional Sunlight & Ambient Light
    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: true,
            illuminance: 12000.0,
            ..default()
        },
        Transform::from_xyz(20.0, 50.0, 30.0).looking_at(sc.cam_center, Vec3::Y),
    ));

    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 800.0,
        ..default()
    });

    // Shared white probe debug mesh and material
    let probe_mesh = meshes.add(Sphere::new(1.0).mesh().uv(24, 14));
    let probe_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.95, 0.95), // White debug sphere
        perceptual_roughness: 0.85,
        metallic: 0.0,
        reflectance: 0.5,
        ..default()
    });
    commands.insert_resource(ProbeAssets {
        mesh: probe_mesh,
        material: probe_mat,
    });

    // Spawn Initial Scene Model
    spawn_scene_model(&mut commands, &assets, sc.asset_path, sc.visual_scale);

    // Build MuTracer-style UI Overlay
    build_ui(&mut commands);
}

fn spawn_scene_model(
    commands: &mut Commands,
    assets: &AssetServer,
    path: &str,
    scale: f32,
) {
    commands.spawn((
        WorldAssetRoot(assets.load(GltfAssetLabel::Scene(0).from_asset(path.to_string()))),
        Transform::from_scale(Vec3::splat(scale)),
        SceneModelRoot,
    ));
}

fn build_ui(commands: &mut Commands) {
    // Root container
    commands
        .spawn((
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                justify_content: JustifyContent::SpaceBetween,
                ..default()
            },
            Pickable::IGNORE,
        ))
        .with_children(|root| {
            // Top Bar
            root.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(42.0),
                    border: UiRect::bottom(Val::Px(1.0)),
                    padding: UiRect::axes(Val::Px(16.0), Val::Px(8.0)),
                    align_items: AlignItems::Center,
                    justify_content: JustifyContent::SpaceBetween,
                    ..default()
                },
                BackgroundColor(BG_DARK),
                BorderColor::all(BORDER_DARK),
                Pickable::default(),
            ))
            .with_children(|bar| {
                bar.spawn((
                    Text::new("MuTracer Native | Irradiance Baker"),
                    TextFont::from_font_size(14.0),
                    TextColor(ACCENT_ORANGE),
                    Pickable::IGNORE,
                ));

                bar.spawn((
                    Text::new("Ready | Qualcomm Adreno 8cx Gen 3 (DX12)"),
                    TextFont::from_font_size(12.0),
                    TextColor(TEXT_LIGHT),
                    StatusLabel,
                    Pickable::IGNORE,
                ));
            });

            // Middle: Right Inspector Panel
            root.spawn((
                Node {
                    width: Val::Percent(100.0),
                    flex_grow: 1.0,
                    flex_direction: FlexDirection::Row,
                    justify_content: JustifyContent::FlexEnd,
                    ..default()
                },
                Pickable::IGNORE,
            ))
            .with_children(|mid| {
                mid.spawn((
                    Node {
                        width: Val::Px(320.0),
                        height: Val::Percent(100.0),
                        border: UiRect::left(Val::Px(1.0)),
                        padding: UiRect::all(Val::Px(14.0)),
                        flex_direction: FlexDirection::Column,
                        overflow: Overflow::scroll_y(),
                        ..default()
                    },
                    BackgroundColor(PANEL_DARK),
                    BorderColor::all(BORDER_DARK),
                    Pickable::default(),
                ))
                .with_children(|panel| {
                    panel_section_header(panel, "SCENE SELECTION");
                    action_button(panel, "+ IMPORT GLTF/GLB...", UiAction::ImportExternal);

                    for (i, sc) in SCENES.iter().enumerate() {
                        panel_button(panel, sc.display_name, UiAction::SelectScene(i));
                    }

                    panel.spawn((
                        Text::new("Model: Cornell Box\nGrid: 4x4x4 (64 Probes)\nSPP: 32 | Bounces: 2"),
                        TextFont::from_font_size(11.0),
                        TextColor(TEXT_MUTED),
                        InfoLabel,
                        Pickable::IGNORE,
                    ));

                    panel_section_header(panel, "VOXEL RESOLUTION");
                    panel
                        .spawn((
                            Node {
                                width: Val::Percent(100.0),
                                flex_direction: FlexDirection::Row,
                                justify_content: JustifyContent::SpaceBetween,
                                ..default()
                            },
                            Pickable::IGNORE,
                        ))
                        .with_children(|row| {
                            panel_sub_button(row, "4x4x4", UiAction::SetResolution(4, 4, 4));
                            panel_sub_button(row, "8x8x8", UiAction::SetResolution(8, 8, 8));
                            panel_sub_button(row, "8x8x3", UiAction::SetResolution(8, 8, 3));
                        });

                    panel_section_header(panel, "QUALITY & BOUNCES");
                    panel
                        .spawn((
                            Node {
                                width: Val::Percent(100.0),
                                flex_direction: FlexDirection::Row,
                                justify_content: JustifyContent::SpaceBetween,
                                ..default()
                            },
                            Pickable::IGNORE,
                        ))
                        .with_children(|row| {
                            panel_sub_button(row, "16 SPP", UiAction::SetSpp(16));
                            panel_sub_button(row, "32 SPP", UiAction::SetSpp(32));
                            panel_sub_button(row, "64 SPP", UiAction::SetSpp(64));
                        });

                    panel
                        .spawn((
                            Node {
                                width: Val::Percent(100.0),
                                flex_direction: FlexDirection::Row,
                                justify_content: JustifyContent::SpaceBetween,
                                margin: UiRect::top(Val::Px(4.0)),
                                ..default()
                            },
                            Pickable::IGNORE,
                        ))
                        .with_children(|row| {
                            panel_sub_button(row, "1 Bounce", UiAction::SetBounces(1));
                            panel_sub_button(row, "2 Bounces", UiAction::SetBounces(2));
                            panel_sub_button(row, "4 Bounces", UiAction::SetBounces(4));
                        });

                    panel_section_header(panel, "BAKE ACTIONS");
                    action_button(panel, "START GPU BAKE", UiAction::StartBake);
                    panel_button(
                        panel,
                        "Raster + Baked GI: OFF",
                        UiAction::ToggleBakedRaster,
                    );
                    action_button(panel, "EXPORT KTX2", UiAction::ExportOutput);

                    panel_section_header(panel, "VISUALIZATION");
                    panel_button(
                        panel,
                        "Toggle Probe Grid",
                        UiAction::ToggleProbeGrid,
                    );
                });
            });
        });
}

fn panel_section_header(parent: &mut ChildSpawnerCommands<'_>, title: &str) {
    parent.spawn((
        Text::new(title),
        TextFont::from_font_size(11.0),
        TextColor(ACCENT_ORANGE),
        Node {
            margin: UiRect {
                top: Val::Px(14.0),
                bottom: Val::Px(6.0),
                ..default()
            },
            ..default()
        },
        Pickable::IGNORE,
    ));
}

fn panel_button(parent: &mut ChildSpawnerCommands<'_>, label: &str, action: UiAction) {
    parent
        .spawn((
            Node {
                width: Val::Percent(100.0),
                height: Val::Px(28.0),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                margin: UiRect::bottom(Val::Px(4.0)),
                ..default()
            },
            BackgroundColor(BUTTON_NORMAL),
            BorderColor::all(BORDER_DARK),
            Pickable::default(),
            action,
        ))
        .with_children(|btn| {
            btn.spawn((
                Text::new(label),
                TextFont::from_font_size(11.5),
                TextColor(TEXT_LIGHT),
                Pickable::IGNORE,
            ));
        });
}

fn panel_sub_button(parent: &mut ChildSpawnerCommands<'_>, label: &str, action: UiAction) {
    parent
        .spawn((
            Node {
                width: Val::Percent(31.0),
                height: Val::Px(26.0),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(BUTTON_NORMAL),
            BorderColor::all(BORDER_DARK),
            Pickable::default(),
            action,
        ))
        .with_children(|btn| {
            btn.spawn((
                Text::new(label),
                TextFont::from_font_size(11.0),
                TextColor(TEXT_LIGHT),
                Pickable::IGNORE,
            ));
        });
}

fn action_button(parent: &mut ChildSpawnerCommands<'_>, label: &str, action: UiAction) {
    parent
        .spawn((
            Node {
                width: Val::Percent(100.0),
                height: Val::Px(34.0),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(6.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                margin: UiRect::bottom(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(ACCENT_BUTTON),
            BorderColor::all(ACCENT_ORANGE),
            Pickable::default(),
            action,
            AccentButton,
        ))
        .with_children(|btn| {
            btn.spawn((
                Text::new(label),
                TextFont::from_font_size(12.0),
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));
        });
}

fn get_action_from_entity(
    mut target: Entity,
    actions: &Query<&UiAction>,
    parents: &Query<&ChildOf>,
) -> Option<UiAction> {
    loop {
        if let Ok(action) = actions.get(target) {
            return Some((*action).clone());
        }
        if let Ok(parent) = parents.get(target) {
            target = parent.parent();
        } else {
            return None;
        }
    }
}

fn find_action_entity(
    mut target: Entity,
    actions: &Query<(Entity, &UiAction)>,
    parents: &Query<&ChildOf>,
) -> Option<Entity> {
    loop {
        if actions.get(target).is_ok() {
            return Some(target);
        }
        if let Ok(parent) = parents.get(target) {
            target = parent.parent();
        } else {
            return None;
        }
    }
}

fn is_action_selected(action: &UiAction, state: &AppState) -> bool {
    match action {
        UiAction::SetResolution(x, y, z) => state.resolution == UVec3::new(*x, *y, *z),
        UiAction::SetSpp(s) => state.spp == *s,
        UiAction::SetBounces(b) => state.bounces == *b,
        UiAction::SelectScene(idx) => {
            *idx < SCENES.len() && state.scene_name == SCENES[*idx].display_name
        }
        UiAction::ToggleProbeGrid => state.show_probe_grid,
        UiAction::ToggleBakedRaster => state.use_baked_raster,
        _ => false,
    }
}

fn handle_ui_click(
    click: On<PointerClick>,
    query: Query<&UiAction>,
    parents: Query<&ChildOf>,
    mut commands: Commands,
    assets: Res<AssetServer>,
    import_channel: Res<ImportChannel>,
    mut state: ResMut<AppState>,
    scene_model_query: Query<Entity, With<SceneModelRoot>>,
    mut cam_query: Query<(&mut Transform, &mut OrbitCamera)>,
) {
    let Some(action) = get_action_from_entity(click.entity, &query, &parents) else {
        return;
    };

    match action {
        UiAction::ImportExternal => {
            state.status_text = "Opening File Dialog... Please select a .glb/.gltf file".into();
            let tx = import_channel.tx.clone();
            std::thread::spawn(move || {
                let ps_script = "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Application]::EnableVisualStyles(); $f = New-Object System.Windows.Forms.OpenFileDialog; $f.Filter = '3D Models (*.glb;*.gltf)|*.glb;*.gltf|All files (*.*)|*.*'; $f.Title = 'Select glTF/GLB 3D Scene'; if ($f.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $f.FileName }";
                let output = std::process::Command::new("powershell")
                    .args(["-NoProfile", "-Command", ps_script])
                    .output();
                if let Ok(out) = output {
                    let path_str = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    if !path_str.is_empty() {
                        let path = PathBuf::from(path_str);
                        if path.exists() {
                            let _ = tx.send(path);
                        }
                    }
                }
            });
        }
        UiAction::SelectScene(idx) => {
            let sc = &SCENES[idx];
            state.scene_name = sc.display_name.to_string();
            state.scene_path = sc.asset_path.to_string();
            state.triangle_count = sc.triangle_count;
            state.bounds_min = sc.bounds_min;
            state.bounds_max = sc.bounds_max;
            state.visual_scale = sc.visual_scale;
            state.baked_result = None;
            state.use_baked_raster = false;

            for e in &scene_model_query {
                commands.entity(e).despawn();
            }
            spawn_scene_model(&mut commands, &assets, sc.asset_path, sc.visual_scale);

            if let Ok((mut transform, mut cam)) = cam_query.single_mut() {
                cam.center = sc.cam_center;
                cam.distance = sc.cam_dist;
                let eye = cam.center
                    + Vec3::new(
                        cam.yaw.cos() * cam.pitch.cos(),
                        cam.pitch.sin(),
                        cam.yaw.sin() * cam.pitch.cos(),
                    ) * cam.distance;
                transform.translation = eye;
                transform.look_at(cam.center, Vec3::Y);
            }

            state.status_text = format!("Loaded: {}", sc.display_name);
        }
        UiAction::SetResolution(x, y, z) => {
            state.resolution = UVec3::new(x, y, z);
            state.status_text = format!("Grid: {}x{}x{} ({} probes)", x, y, z, x * y * z);
        }
        UiAction::SetSpp(s) => {
            state.spp = s;
            state.status_text = format!("Quality: {} SPP", s);
        }
        UiAction::SetBounces(b) => {
            state.bounces = b;
            state.status_text = format!("Bounces: {}", b);
        }
        UiAction::ToggleProbeGrid => {
            state.show_probe_grid = !state.show_probe_grid;
            state.status_text = format!(
                "Probe Grid: {}",
                if state.show_probe_grid { "ON" } else { "OFF" }
            );
        }
        UiAction::ToggleBakedRaster => {
            if state.baked_result.is_none() {
                state.status_text = "Please bake irradiance first before enabling baked GI!".into();
            } else {
                state.use_baked_raster = !state.use_baked_raster;
                if state.use_baked_raster {
                    state.is_baking = false;
                    state.status_text =
                        "Mode: Raster + Baked Irradiance Volume (Path Tracing OFF)".into();
                } else {
                    state.status_text =
                        "Mode: Standard Direct Lighting (Path Tracing Ready)".into();
                }
            }
        }
        UiAction::StartBake => {
            if !state.is_baking {
                state.is_baking = true;
                state.status_text = "Baking Irradiance on Adreno GPU (DX12)...".into();

                let full_scene_path = if Path::new(&state.scene_path).is_absolute() {
                    PathBuf::from(&state.scene_path)
                } else {
                    std::env::current_dir()
                        .unwrap_or_else(|_| PathBuf::from("."))
                        .join("assets")
                        .join(&state.scene_path)
                };

                let task = BakeTask {
                    scene_path: full_scene_path,
                    settings: BakeSettings {
                        resolution: state.resolution,
                        bounds_min: state.bounds_min,
                        bounds_max: state.bounds_max,
                        spp: state.spp,
                        max_bounces: state.bounces,
                    },
                };
                let _ = state.bake_tx.send(task);
            }
        }
        UiAction::ExportOutput => {
            if let Some(ref res) = state.baked_result {
                let out_ktx2 = PathBuf::from("baked_irradiance.vxgi.ktx2");
                let out_meta = PathBuf::from("baked_irradiance.vxgi.meta.json");
                if let Ok(_) = std::fs::write(&out_ktx2, &res.ktx2_bytes) {
                    let meta = serde_json::json!({
                        "voxels_texture": "baked_irradiance.vxgi.ktx2",
                        "format": "VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 (Rgb9e5Ufloat)",
                        "resolution": [res.resolution.x, res.resolution.y, res.resolution.z],
                        "dimensions": [res.texture_dimensions.x, res.texture_dimensions.y, res.texture_dimensions.z],
                    });
                    let _ = std::fs::write(
                        &out_meta,
                        serde_json::to_string_pretty(&meta).unwrap_or_default(),
                    );
                    state.status_text =
                        format!("Saved {} ({} bytes)!", out_ktx2.display(), res.ktx2_bytes.len());
                }
            } else {
                state.status_text = "Please bake irradiance first!".into();
            }
        }
    }
}

fn poll_import_dialog(
    mut commands: Commands,
    assets: Res<AssetServer>,
    import_channel: Res<ImportChannel>,
    mut state: ResMut<AppState>,
    scene_model_query: Query<Entity, With<SceneModelRoot>>,
    mut cam_query: Query<(&mut Transform, &mut OrbitCamera)>,
) {
    let maybe_path = {
        if let Ok(rx) = import_channel.rx.lock() {
            rx.try_recv().ok()
        } else {
            None
        }
    };

    if let Some(path) = maybe_path {
        let file_name = path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| "ImportedModel.glb".to_string());

        // Copy file to local assets/imported directory
        let imported_dir = Path::new("assets/imported");
        let _ = std::fs::create_dir_all(imported_dir);
        let dest = imported_dir.join(&file_name);
        let _ = std::fs::copy(&path, &dest);

        // Also copy to target/debug/assets/imported if running from target/debug
        let target_imported = Path::new("target/debug/assets/imported");
        let _ = std::fs::create_dir_all(target_imported);
        let _ = std::fs::copy(&path, target_imported.join(&file_name));

        // Parse with NativeScene to obtain bounds & triangle count
        if let Ok(native_scene) = NativeScene::load_gltf(&path) {
            let mut b_min = Vec3::splat(f32::INFINITY);
            let mut b_max = Vec3::splat(f32::NEG_INFINITY);
            for tri in &native_scene.triangles {
                let (t_min, t_max) = tri.aabb();
                b_min = b_min.min(t_min);
                b_max = b_max.max(t_max);
            }

            let extents = b_max - b_min;
            let max_dim = extents.x.max(extents.y).max(extents.z);
            let visual_scale = if max_dim > 100.0 {
                0.015
            } else {
                1.0
            };

            let v_center = (b_min + b_max) * 0.5 * visual_scale;
            let v_size = extents * visual_scale;
            let cam_dist = (v_size.x.max(v_size.y).max(v_size.z) * 2.2).clamp(3.0, 50.0);

            state.bounds_min = b_min;
            state.bounds_max = b_max;
            state.visual_scale = visual_scale;
            state.scene_name = file_name.clone();
            state.scene_path = path.to_string_lossy().to_string();
            state.triangle_count = native_scene.triangles.len();
            state.baked_result = None;
            state.use_baked_raster = false;

            for e in &scene_model_query {
                commands.entity(e).despawn();
            }

            let asset_rel_path = format!("imported/{}", file_name);
            spawn_scene_model(&mut commands, &assets, &asset_rel_path, visual_scale);

            if let Ok((mut transform, mut cam)) = cam_query.single_mut() {
                cam.center = v_center;
                cam.distance = cam_dist;
                let eye = cam.center
                    + Vec3::new(
                        cam.yaw.cos() * cam.pitch.cos(),
                        cam.pitch.sin(),
                        cam.yaw.sin() * cam.pitch.cos(),
                    ) * cam.distance;
                transform.translation = eye;
                transform.look_at(cam.center, Vec3::Y);
            }

            state.status_text = format!(
                "Imported: {} ({} Tris, size {:.1}x{:.1}x{:.1})",
                file_name, state.triangle_count, extents.x, extents.y, extents.z
            );
        } else {
            state.status_text = format!("Failed to parse glTF: {}", file_name);
        }
    }
}

fn update_button_highlights(
    state: Res<AppState>,
    mut buttons: Query<(
        Entity,
        &UiAction,
        &mut BackgroundColor,
        &mut BorderColor,
        &Children,
    )>,
    mut text_query: Query<&mut TextColor>,
    mut text_content_query: Query<&mut Text>,
) {
    if !state.is_changed() {
        return;
    }

    for (_, action, mut bg, mut border, children) in &mut buttons {
        let is_selected = is_action_selected(action, &state);

        // Update button text for dynamic buttons like ToggleBakedRaster
        if matches!(action, UiAction::ToggleBakedRaster) {
            for child in children.iter() {
                if let Ok(mut text) = text_content_query.get_mut(child) {
                    text.0 = if state.use_baked_raster {
                        "Raster + Baked GI: ON".into()
                    } else {
                        "Raster + Baked GI: OFF".into()
                    };
                }
            }
        }

        // Keep action buttons (START GPU BAKE, EXPORT KTX2, IMPORT GLTF) with distinct accent styling
        if matches!(
            action,
            UiAction::StartBake | UiAction::ExportOutput | UiAction::ImportExternal
        ) {
            continue;
        }

        if is_selected {
            bg.0 = Color::srgb(0.42, 0.20, 0.12);
            *border = BorderColor::all(ACCENT_ORANGE);
            for child in children.iter() {
                if let Ok(mut tc) = text_query.get_mut(child) {
                    tc.0 = Color::WHITE;
                }
            }
        } else {
            bg.0 = BUTTON_NORMAL;
            *border = BorderColor::all(BORDER_DARK);
            for child in children.iter() {
                if let Ok(mut tc) = text_query.get_mut(child) {
                    tc.0 = TEXT_LIGHT;
                }
            }
        }
    }
}

fn handle_button_over(
    over: On<PointerOver>,
    actions: Query<(Entity, &UiAction)>,
    parents: Query<&ChildOf>,
    mut bg: Query<&mut BackgroundColor>,
    state: Res<AppState>,
) {
    if let Some(target) = find_action_entity(over.entity, &actions, &parents) {
        if let Ok((_, action)) = actions.get(target) {
            if let Ok(mut c) = bg.get_mut(target) {
                if matches!(action, UiAction::StartBake | UiAction::ExportOutput) {
                    c.0 = Color::srgb(0.82, 0.35, 0.20);
                } else if is_action_selected(action, &state) {
                    c.0 = Color::srgb(0.50, 0.26, 0.16);
                } else {
                    c.0 = BUTTON_HOVER;
                }
            }
        }
    }
}

fn handle_button_out(
    out: On<PointerOut>,
    actions: Query<(Entity, &UiAction)>,
    parents: Query<&ChildOf>,
    mut bg: Query<&mut BackgroundColor>,
    state: Res<AppState>,
) {
    if let Some(target) = find_action_entity(out.entity, &actions, &parents) {
        if let Ok((_, action)) = actions.get(target) {
            if let Ok(mut c) = bg.get_mut(target) {
                if matches!(action, UiAction::StartBake | UiAction::ExportOutput) {
                    c.0 = ACCENT_BUTTON;
                } else if is_action_selected(action, &state) {
                    c.0 = Color::srgb(0.42, 0.20, 0.12);
                } else {
                    c.0 = BUTTON_NORMAL;
                }
            }
        }
    }
}

fn orbit_camera_controller(
    mut cam_query: Query<(&mut Transform, &mut OrbitCamera)>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mouse_scroll: Res<AccumulatedMouseScroll>,
) {
    let Ok((mut transform, mut cam)) = cam_query.single_mut() else {
        return;
    };

    if mouse_buttons.pressed(MouseButton::Left) {
        cam.yaw += mouse_motion.delta.x * 0.006;
        cam.pitch = (cam.pitch - mouse_motion.delta.y * 0.006).clamp(-1.5, 1.5);
    }

    if mouse_scroll.delta.y != 0.0 {
        cam.distance = (cam.distance - mouse_scroll.delta.y * 0.8).clamp(1.5, 50.0);
    }

    let eye = cam.center
        + Vec3::new(
            cam.yaw.cos() * cam.pitch.cos(),
            cam.pitch.sin(),
            cam.yaw.sin() * cam.pitch.cos(),
        ) * cam.distance;

    transform.translation = eye;
    transform.look_at(cam.center, Vec3::Y);
}

fn sync_probe_spheres(
    mut commands: Commands,
    state: Res<AppState>,
    probe_assets: Res<ProbeAssets>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    probe_spheres: Query<Entity, With<ProbeSphere>>,
) {
    if !state.is_changed() && !probe_spheres.is_empty() {
        return;
    }

    let expected_count = if state.show_probe_grid {
        (state.resolution.x * state.resolution.y * state.resolution.z) as usize
    } else {
        0
    };

    let current_count = probe_spheres.iter().count();
    if current_count == expected_count && (!state.is_changed()) {
        return;
    }

    // Despawn old probe spheres
    for entity in &probe_spheres {
        commands.entity(entity).despawn();
    }

    if !state.show_probe_grid {
        return;
    }

    let rx = state.resolution.x;
    let ry = state.resolution.y;
    let rz = state.resolution.z;
    let size = state.bounds_max - state.bounds_min;
    let v_size = size * state.visual_scale;
    let max_extent = v_size.x.max(v_size.y).max(v_size.z);
    let radius = (max_extent / 45.0).clamp(0.04, 0.35);

    let tex_w = rx;
    let tex_h = 2 * ry;
    let idx = |s: u32, t: u32, p: u32| (s + t * tex_w + p * (tex_w * tex_h)) as usize;

    for vz in 0..rz {
        for vy in 0..ry {
            for vx in 0..rx {
                let t_coord = Vec3::new(
                    (vx as f32 + 0.5) / rx as f32,
                    (vy as f32 + 0.5) / ry as f32,
                    (vz as f32 + 0.5) / rz as f32,
                );
                let p = state.bounds_min + t_coord * size;
                let vp = p * state.visual_scale;

                // Color probe sphere according to baked indirect irradiance
                let mat_handle = if let Some(ref baked) = state.baked_result {
                    let f_px = baked.raw_floats[idx(vx, vy, vz)];
                    let f_nx = baked.raw_floats[idx(vx, vy + ry, vz)];
                    let f_py = baked.raw_floats[idx(vx, vy, vz + rz)];
                    let f_ny = baked.raw_floats[idx(vx, vy + ry, vz + rz)];
                    let f_pz = baked.raw_floats[idx(vx, vy, vz + 2 * rz)];
                    let f_nz = baked.raw_floats[idx(vx, vy + ry, vz + 2 * rz)];

                    let avg_r = (f_px[0] + f_nx[0] + f_py[0] + f_ny[0] + f_pz[0] + f_nz[0]) / 6.0;
                    let avg_g = (f_px[1] + f_nx[1] + f_py[1] + f_ny[1] + f_pz[1] + f_nz[1]) / 6.0;
                    let avg_b = (f_px[2] + f_nx[2] + f_py[2] + f_ny[2] + f_pz[2] + f_nz[2]) / 6.0;
                    let max_c = avg_r.max(avg_g).max(avg_b).max(1e-4);

                    // Blend white with chromatic color bleeding so probes show true diffuse lighting
                    let cr = (avg_r / max_c).clamp(0.0, 1.0);
                    let cg = (avg_g / max_c).clamp(0.0, 1.0);
                    let cb = (avg_b / max_c).clamp(0.0, 1.0);
                    let probe_color = Color::srgb(
                        0.50 * cr + 0.45,
                        0.50 * cg + 0.45,
                        0.50 * cb + 0.45,
                    );
                    materials.add(StandardMaterial {
                        base_color: probe_color,
                        perceptual_roughness: 0.85,
                        metallic: 0.0,
                        reflectance: 0.5,
                        ..default()
                    })
                } else {
                    probe_assets.material.clone()
                };

                commands.spawn((
                    Mesh3d(probe_assets.mesh.clone()),
                    MeshMaterial3d(mat_handle),
                    Transform::from_translation(vp).with_scale(Vec3::splat(radius)),
                    ProbeSphere,
                ));
            }
        }
    }
}

fn draw_probe_gizmos(mut gizmos: Gizmos, state: Res<AppState>) {
    let center = (state.bounds_min + state.bounds_max) * 0.5;
    let size = state.bounds_max - state.bounds_min;

    let visual_scale = state.visual_scale;
    let v_center = center * visual_scale;
    let v_size = size * visual_scale;

    // Draw bounding box
    gizmos.cube(
        Transform::from_translation(v_center).with_scale(v_size),
        ACCENT_ORANGE.with_alpha(0.6),
    );

    if !state.show_probe_grid {
        return;
    }

    let rx = state.resolution.x;
    let ry = state.resolution.y;
    let rz = state.resolution.z;
    let max_extent = v_size.x.max(v_size.y).max(v_size.z);
    let radius = (max_extent / 45.0).clamp(0.04, 0.35);

    let tex_w = rx;
    let tex_h = 2 * ry;
    let idx = |s: u32, t: u32, p: u32| (s + t * tex_w + p * (tex_w * tex_h)) as usize;

    for vz in 0..rz {
        for vy in 0..ry {
            for vx in 0..rx {
                let t_coord = Vec3::new(
                    (vx as f32 + 0.5) / rx as f32,
                    (vy as f32 + 0.5) / ry as f32,
                    (vz as f32 + 0.5) / rz as f32,
                );
                let p = state.bounds_min + t_coord * size;
                let vp = p * visual_scale;

                // Calculate light direction for this probe
                let (dominant_dir, arrow_color) = if let Some(ref baked) = state.baked_result {
                    let f_px = baked.raw_floats[idx(vx, vy, vz)];
                    let f_nx = baked.raw_floats[idx(vx, vy + ry, vz)];
                    let f_py = baked.raw_floats[idx(vx, vy, vz + rz)];
                    let f_ny = baked.raw_floats[idx(vx, vy + ry, vz + rz)];
                    let f_pz = baked.raw_floats[idx(vx, vy, vz + 2 * rz)];
                    let f_nz = baked.raw_floats[idx(vx, vy + ry, vz + 2 * rz)];

                    let lum = |f: [f32; 4]| 0.2126 * f[0] + 0.7152 * f[1] + 0.0722 * f[2];
                    let dir_x = lum(f_px) - lum(f_nx);
                    let dir_y = lum(f_py) - lum(f_ny);
                    let dir_z = lum(f_pz) - lum(f_nz);
                    let light_vec = Vec3::new(dir_x, dir_y, dir_z);

                    let dir = if light_vec.length_squared() > 1e-6 {
                        light_vec.normalize()
                    } else {
                        Vec3::Y
                    };
                    (dir, Color::srgb(1.0, 0.85, 0.25)) // Warm golden arrow indicating baked light direction
                } else {
                    let dir = Vec3::new(20.0, 50.0, 30.0).normalize();
                    (dir, Color::srgb(1.0, 0.90, 0.40)) // Sunlight direction arrow
                };

                // Draw prominent directional pointer from the center of the sphere outward
                gizmos.arrow(vp, vp + dominant_dir * (radius * 2.2), arrow_color);
            }
        }
    }
}

fn sync_baked_irradiance_volume(
    mut commands: Commands,
    state: Res<AppState>,
    mut images: ResMut<Assets<Image>>,
    mut ambient_light: ResMut<GlobalAmbientLight>,
    volume_query: Query<Entity, With<BakedIrradianceVolume>>,
) {
    if !state.is_changed() {
        return;
    }

    if state.use_baked_raster && state.baked_result.is_some() {
        let res = state.baked_result.as_ref().unwrap();

        if volume_query.is_empty() {
            let mut packed_pixels: Vec<u32> = Vec::with_capacity(res.raw_floats.len());
            for p in &res.raw_floats {
                packed_pixels.push(muttracer_native::ktx2_writer::pack_rgb9e5(p[0], p[1], p[2]));
            }

            let image = Image::new(
                Extent3d {
                    width: res.texture_dimensions.x,
                    height: res.texture_dimensions.y,
                    depth_or_array_layers: res.texture_dimensions.z,
                },
                TextureDimension::D3,
                bytemuck::cast_slice(&packed_pixels).to_vec(),
                TextureFormat::Rgb9e5Ufloat,
                RenderAssetUsages::RENDER_WORLD,
            );
            let image_handle = images.add(image);

            let center = (state.bounds_min + state.bounds_max) * 0.5 * state.visual_scale;
            let size = (state.bounds_max - state.bounds_min) * state.visual_scale;

            // Expand volume bounds by 25% with gentle falloff so all walls, ceiling and floor are inside the light probe range
            let volume_size = size * 1.25;

            commands.spawn((
                Transform::from_translation(center).with_scale(volume_size),
                LightProbe {
                    falloff: Vec3::splat(0.06),
                },
                IrradianceVolume {
                    voxels: image_handle,
                    intensity: 1400.0,
                    ..default()
                },
                BakedIrradianceVolume,
            ));
        }

        // Soft ambient fill alongside the rich baked indirect global illumination
        ambient_light.brightness = 150.0;
    } else {
        for entity in &volume_query {
            commands.entity(entity).despawn();
        }
        ambient_light.brightness = 800.0;
    }
}

fn poll_bake_results(mut state: ResMut<AppState>) {
    let maybe_res = {
        if let Ok(rx) = state.bake_rx.lock() {
            rx.try_recv().ok()
        } else {
            None
        }
    };

    if let Some(res) = maybe_res {
        state.is_baking = false;
        match res {
            BakeTaskResult::Success(b) => {
                state.status_text = format!(
                    "Baked in {:.2?} | {}x{}x{} ({} bytes)",
                    b.duration,
                    b.texture_dimensions.x,
                    b.texture_dimensions.y,
                    b.texture_dimensions.z,
                    b.ktx2_bytes.len()
                );
                state.baked_result = Some(b);
            }
            BakeTaskResult::Error(err) => {
                state.status_text = format!("Bake Error: {}", err);
            }
        }
    }
}

fn update_labels(
    state: Res<AppState>,
    mut status_query: Query<&mut Text, (With<StatusLabel>, Without<InfoLabel>)>,
    mut info_query: Query<&mut Text, (With<InfoLabel>, Without<StatusLabel>)>,
) {
    for mut text in &mut status_query {
        text.0 = state.status_text.clone();
    }
    for mut text in &mut info_query {
        let total_p = state.resolution.x * state.resolution.y * state.resolution.z;
        let mode_desc = if state.is_baking {
            "GPU Path Tracing Active"
        } else if state.use_baked_raster {
            "Raster + Baked GI"
        } else {
            "Direct Raster Preview"
        };
        text.0 = format!(
            "Model: {}\nPipeline: {}\nGrid: {}x{}x{} ({} Probes)\nSPP: {} | Bounces: {}",
            state.scene_name,
            mode_desc,
            state.resolution.x,
            state.resolution.y,
            state.resolution.z,
            total_p,
            state.spp,
            state.bounces
        );
    }
}
