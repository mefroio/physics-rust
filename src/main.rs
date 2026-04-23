use bevy::render::{
    mesh::{primitives::CylinderMeshBuilder, Indices},
    render_asset::RenderAssetUsages,
    render_resource::PrimitiveTopology,
    view::RenderLayers,
};
use bevy::tasks::{
    futures_lite::future::{block_on, poll_once},
    AsyncComputeTaskPool, Task,
};
use bevy::{
    input::{
        mouse::{MouseMotion, MouseScrollUnit, MouseWheel},
        ButtonInput,
    },
    math::{primitives::Cuboid, IVec2},
    prelude::*,
    time::Virtual,
    render::camera::ClearColorConfig,
};
use bevy_rapier3d::{
    prelude::*,
    rapier::{
        math::{AngVector, Vector},
        na::UnitQuaternion,
    },
    render::{DebugRenderContext, RapierDebugRenderPlugin},
};
use std::collections::{HashMap, HashSet};
use std::f32::consts::TAU;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Component)]
struct OrbitCamera;

#[derive(Component)]
struct TerrainDebugText;

#[derive(Component)]
struct VehicleChassis;

#[derive(Component)]
struct VehicleHudText;

#[derive(Component)]
struct VehicleWheel {
    local_offset: Vec3,
    radius: f32,
    rest_length: f32,
    stiffness: f32,
    damping: f32,
    steering: bool,
    drive: bool,
    health: f32,
    detached: bool,
    compression: f32,
    time_since_grounded: f32,
    spin_angle: f32,
    spin_velocity: f32,
    mesh_handle: Handle<Mesh>,
    material_handle: Handle<StandardMaterial>,
}

#[derive(Component)]
struct WheelVisual;

#[derive(Resource)]
struct VehicleEntities {
    chassis: Entity,
    wheels: [Entity; 4],
}

#[derive(Resource, Default)]
struct VehicleControlInput {
    steering: f32,
    throttle: f32,
    brake: f32,
}

#[derive(Resource, Default)]
struct VehicleStatus {
    speed_mps: f32,
    wheels_attached: usize,
}

#[derive(Resource)]
struct ChaseCameraState {
    offset_local: Vec3,
    follow_speed: f32,
}

#[derive(Resource)]
struct TerrainAssets {
    material: Handle<StandardMaterial>,
}

#[derive(Resource, Default)]
struct TerrainManager {
    chunks: HashMap<IVec2, Entity>,
}

#[derive(Resource, Default)]
struct PendingChunkTasks {
    tasks: HashMap<IVec2, Task<Option<ChunkBuildResult>>>,
}

struct ChunkBuildResult {
    mesh: Mesh,
    collider: Collider,
}

static SEED_COUNTER: AtomicU64 = AtomicU64::new(0);

fn generate_seed() -> u64 {
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x1234_5678_9ABC_DEF0u64);
    let counter = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mixed = time ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    mixed ^ mixed.rotate_left(17)
}

#[derive(Resource)]
struct CameraZoomSettings {
    min_distance: f32,
    max_distance: f32,
    zoom_speed: f32,
}

#[derive(Resource, Clone, Copy)]
struct TerrainSeed(u64);

const CAMERA_TARGET: Vec3 = Vec3::ZERO;
const CAMERA_MIN_HEIGHT_ABOVE_TARGET: f32 = 0.6;
const TERRAIN_CHUNK_SIZE: f32 = 40.0;
const TERRAIN_CHUNK_RESOLUTION: usize = 32;
const TERRAIN_AMPLITUDE: f32 = 15.0;
const TERRAIN_LOAD_RADIUS: i32 = 3;
const TERRAIN_UNLOAD_RADIUS: i32 = TERRAIN_LOAD_RADIUS + 1;
const MAX_CHUNK_TASKS_PER_FRAME: usize = 10000000;
const MOUNTAIN_HEIGHT_SCALE: f32 = 10.0;
const ANIMATION_SPEED_MULTIPLIER: f32 = 1.45;

const WHEEL_COUNT: usize = 4;
const WHEEL_RADIUS_DEFAULT: f32 = 0.55;
const WHEEL_WIDTH_DEFAULT: f32 = 0.3;
const WHEEL_TREAD_SEGMENTS: usize = 32;
const SUSPENSION_REST_LENGTH: f32 = 0.75;
const SUSPENSION_STIFFNESS: f32 = 22000.0;
const SUSPENSION_DAMPING: f32 = 3200.0;
const ENGINE_FORCE: f32 = 29000.0;
const HANDBRAKE_FORCE: f32 = 42000.0;
const HANDBRAKE_BLEED_RATE: f32 = 9.0;
const HANDBRAKE_MIN_SPEED: f32 = 3.5;
const WHEEL_SPIN_TRACK_RATE: f32 = 18.0;
const WHEEL_SPIN_AIR_RATE: f32 = 6.0;
const LATERAL_FRICTION_FORCE: f32 = 6500.0;
const STEERING_MAX_ANGLE: f32 = 0.6;
const STEERING_SPEED: f32 = 2.8;
const WHEEL_MAX_HEALTH: f32 = 100.0;
const WHEEL_SPOKE_COUNT: usize = 6;
const CRASH_VELOCITY_THRESHOLD: f32 = 28.0;
const CRASH_COMPRESSION_THRESHOLD: f32 = 0.65;

fn camera_start_position() -> Vec3 {
    Vec3::new(-6.0, 5.5, 10.0)
}

fn chassis_base_color() -> Color {
    Color::rgb(0.18, 0.28, 0.86)
}

fn wheel_base_color() -> Color {
    Color::rgb(0.07, 0.07, 0.07)
}

fn wheel_lug_color() -> Color {
    Color::rgb(0.62, 0.64, 0.7)
}

fn wheel_axis_color() -> Color {
    Color::rgb(0.12, 0.13, 0.16)
}

fn wheel_base_rotation() -> Quat {
    Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)
}

fn apply_time_scale(mut time: ResMut<Time<Virtual>>) {
    time.set_relative_speed(ANIMATION_SPEED_MULTIPLIER);
}

fn build_car_body_mesh(chassis_half_extents: Vec3) -> Mesh {
    #[derive(Clone, Copy)]
    struct SliceProfile {
        z: f32,
        width_bottom: f32,
        width_shoulder: f32,
        width_roof: f32,
        shoulder_height: f32,
        roof_height: f32,
    }

    let half_width = chassis_half_extents.x;
    let half_height = chassis_half_extents.y;
    let half_length = chassis_half_extents.z;
    let height = half_height * 2.0;

    let slices = [
        SliceProfile {
            z: -half_length,
            width_bottom: half_width * 0.45,
            width_shoulder: half_width * 0.32,
            width_roof: half_width * 0.18,
            shoulder_height: (-half_height + height * 0.20).clamp(-half_height, half_height),
            roof_height: (-half_height + height * 0.25).clamp(-half_height, half_height),
        },
        SliceProfile {
            z: -half_length * 0.65,
            width_bottom: half_width * 0.90,
            width_shoulder: half_width * 0.65,
            width_roof: half_width * 0.32,
            shoulder_height: (-half_height + height * 0.35).clamp(-half_height, half_height),
            roof_height: (-half_height + height * 0.55).clamp(-half_height, half_height),
        },
        SliceProfile {
            z: -half_length * 0.2,
            width_bottom: half_width * 1.00,
            width_shoulder: half_width * 0.75,
            width_roof: half_width * 0.45,
            shoulder_height: (-half_height + height * 0.45).clamp(-half_height, half_height),
            roof_height: (half_height * 0.90).clamp(-half_height, half_height),
        },
        SliceProfile {
            z: half_length * 0.25,
            width_bottom: half_width * 0.98,
            width_shoulder: half_width * 0.74,
            width_roof: half_width * 0.45,
            shoulder_height: (-half_height + height * 0.48).clamp(-half_height, half_height),
            roof_height: (half_height * 0.90).clamp(-half_height, half_height),
        },
        SliceProfile {
            z: half_length * 0.65,
            width_bottom: half_width * 0.85,
            width_shoulder: half_width * 0.60,
            width_roof: half_width * 0.30,
            shoulder_height: (-half_height + height * 0.42).clamp(-half_height, half_height),
            roof_height: (-half_height + height * 0.60).clamp(-half_height, half_height),
        },
        SliceProfile {
            z: half_length,
            width_bottom: half_width * 0.60,
            width_shoulder: half_width * 0.42,
            width_roof: half_width * 0.20,
            shoulder_height: (-half_height + height * 0.28).clamp(-half_height, half_height),
            roof_height: (-half_height + height * 0.32).clamp(-half_height, half_height),
        },
    ];

    let mut positions: Vec<Vec3> = Vec::with_capacity(slices.len() * 6);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(slices.len() * 6);
    let mut slice_indices: Vec<[u32; 6]> = Vec::with_capacity(slices.len());

    for (i, slice) in slices.iter().enumerate() {
        let start = positions.len() as u32;
        let z_ratio = if slices.len() > 1 {
            i as f32 / (slices.len() - 1) as f32
        } else {
            0.0
        };

        positions.push(Vec3::new(-slice.width_bottom, -half_height, slice.z));
        uvs.push([z_ratio, 0.0]);
        positions.push(Vec3::new(slice.width_bottom, -half_height, slice.z));
        uvs.push([z_ratio, 0.0]);

        positions.push(Vec3::new(
            -slice.width_shoulder,
            slice.shoulder_height,
            slice.z,
        ));
        uvs.push([z_ratio, 0.45]);
        positions.push(Vec3::new(
            slice.width_shoulder,
            slice.shoulder_height,
            slice.z,
        ));
        uvs.push([z_ratio, 0.45]);

        positions.push(Vec3::new(-slice.width_roof, slice.roof_height, slice.z));
        uvs.push([z_ratio, 1.0]);
        positions.push(Vec3::new(slice.width_roof, slice.roof_height, slice.z));
        uvs.push([z_ratio, 1.0]);

        slice_indices.push([start, start + 1, start + 2, start + 3, start + 4, start + 5]);
    }

    fn add_quad(indices: &mut Vec<u32>, a: u32, b: u32, c: u32, d: u32) {
        indices.extend_from_slice(&[a, b, c, a, c, d]);
    }

    let mut indices: Vec<u32> = Vec::with_capacity((slices.len() - 1) * 6 * 6 + 12);

    for window in slice_indices.windows(2) {
        let current = window[0];
        let next = window[1];
        add_quad(&mut indices, current[0], current[1], next[1], next[0]); // underside
        add_quad(&mut indices, current[0], next[0], next[2], current[2]); // lower left
        add_quad(&mut indices, next[1], current[1], current[3], next[3]); // lower right
        add_quad(&mut indices, current[2], next[2], next[4], current[4]); // upper left
        add_quad(&mut indices, next[3], current[3], current[5], next[5]); // upper right
        add_quad(&mut indices, current[5], current[4], next[4], next[5]); // roof
    }

    if let Some(front) = slice_indices.first() {
        add_quad(&mut indices, front[1], front[0], front[2], front[3]);
        add_quad(&mut indices, front[3], front[2], front[4], front[5]);
    }

    if let Some(rear) = slice_indices.last() {
        add_quad(&mut indices, rear[0], rear[1], rear[3], rear[2]);
        add_quad(&mut indices, rear[2], rear[3], rear[5], rear[4]);
    }

    let position_array: Vec<[f32; 3]> =
        positions.into_iter().map(|position| position.to_array()).collect();

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, position_array);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh.duplicate_vertices();
    mesh.compute_flat_normals();

    mesh
}

fn build_tire_mesh(radius: f32, width: f32) -> Mesh {
    #[derive(Clone, Copy)]
    struct Layer {
        y: f32,
        outer_radius: f32,
        inner_radius: f32,
    }

    fn add_quad(indices: &mut Vec<u32>, a: u32, b: u32, c: u32, d: u32) {
        indices.extend_from_slice(&[a, b, c, a, c, d]);
    }

    let half_width = width * 0.5;
    let layers = [
        Layer {
            y: -half_width,
            outer_radius: radius * 0.98,
            inner_radius: radius * 0.62,
        },
        Layer {
            y: -half_width * 0.5,
            outer_radius: radius * 1.02,
            inner_radius: radius * 0.58,
        },
        Layer {
            y: 0.0,
            outer_radius: radius * 1.01,
            inner_radius: radius * 0.4,
        },
        Layer {
            y: half_width * 0.5,
            outer_radius: radius * 1.02,
            inner_radius: radius * 0.58,
        },
        Layer {
            y: half_width,
            outer_radius: radius * 0.98,
            inner_radius: radius * 0.62,
        },
    ];

    let segments = WHEEL_TREAD_SEGMENTS.max(3);
    let layer_count = layers.len();
    let layer_stride = segments * 2;

    let mut positions: Vec<Vec3> = Vec::with_capacity(layer_count * layer_stride);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(layer_count * layer_stride);

    for (layer_index, layer) in layers.iter().enumerate() {
        let v = if layer_count > 1 {
            layer_index as f32 / (layer_count - 1) as f32
        } else {
            0.0
        };

        for seg in 0..segments {
            let theta = (seg as f32 / segments as f32) * TAU;
            let (sin_theta, cos_theta) = theta.sin_cos();
            let x = layer.outer_radius * cos_theta;
            let z = layer.outer_radius * sin_theta;
            positions.push(Vec3::new(x, layer.y, z));
            uvs.push([seg as f32 / segments as f32, v]);
        }

        for seg in 0..segments {
            let theta = (seg as f32 / segments as f32) * TAU;
            let (sin_theta, cos_theta) = theta.sin_cos();
            let x = layer.inner_radius * cos_theta;
            let z = layer.inner_radius * sin_theta;
            positions.push(Vec3::new(x, layer.y * 0.92, z));
            uvs.push([seg as f32 / segments as f32, v]);
        }
    }

    let mut indices: Vec<u32> =
        Vec::with_capacity((layer_count - 1) * segments * 12 + layer_count * segments * 6);
    let layer_stride_u32 = layer_stride as u32;
    let segments_u32 = segments as u32;

    for layer in 0..(layer_count - 1) as u32 {
        let base = layer * layer_stride_u32;
        let next = base + layer_stride_u32;
        for seg in 0..segments_u32 {
            let a = base + seg;
            let b = base + ((seg + 1) % segments_u32);
            let c = next + ((seg + 1) % segments_u32);
            let d = next + seg;
            add_quad(&mut indices, a, b, c, d);
        }
    }

    for layer in 0..(layer_count - 1) as u32 {
        let base = layer * layer_stride_u32 + segments_u32;
        let next = base + layer_stride_u32;
        for seg in 0..segments_u32 {
            let a = base + seg;
            let b = base + ((seg + 1) % segments_u32);
            let c = next + ((seg + 1) % segments_u32);
            let d = next + seg;
            add_quad(&mut indices, a, c, b, d);
        }
    }

    for layer in 0..layer_count as u32 {
        let base = layer * layer_stride_u32;
        let inner_base = base + segments_u32;
        for seg in 0..segments_u32 {
            let next = (seg + 1) % segments_u32;
            let outer_a = base + seg;
            let outer_b = base + next;
            let inner_b = inner_base + next;
            let inner_a = inner_base + seg;
            add_quad(&mut indices, outer_a, outer_b, inner_b, inner_a);
        }
    }

    let positions_array: Vec<[f32; 3]> =
        positions.into_iter().map(|p| p.to_array()).collect();

    let mut mesh =
        Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions_array);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh.duplicate_vertices();
    mesh.compute_flat_normals();

    mesh
}

fn hash2d(x: i32, z: i32, seed: u64) -> f32 {
    let mut value = seed
        .wrapping_add((x as i64 * 374761393_i64 + z as i64 * 668265263_i64) as u64)
        .rotate_left(23);
    value = value
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0xBF58_476D_1CE4_E5B9);
    ((value >> 32) as u32) as f32 / u32::MAX as f32
}

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn value_noise(x: f32, z: f32, seed: u64) -> f32 {
    let x0 = x.floor() as i32;
    let z0 = z.floor() as i32;
    let xf = x - x.floor();
    let zf = z - z.floor();

    let v00 = hash2d(x0, z0, seed);
    let v10 = hash2d(x0 + 1, z0, seed);
    let v01 = hash2d(x0, z0 + 1, seed);
    let v11 = hash2d(x0 + 1, z0 + 1, seed);

    let u = smoothstep(xf);
    let v = smoothstep(zf);

    let x1 = lerp(v00, v10, u);
    let x2 = lerp(v01, v11, u);
    lerp(x1, x2, v)
}

const FRACTAL_OFFSETS: [(f32, f32); 6] = [
    (37.2, -91.7),
    (-63.5, 54.1),
    (12.3, 84.7),
    (-48.9, -23.8),
    (91.0, 11.5),
    (-27.4, 67.3),
];

fn fractal_noise(x: f32, z: f32, seed: u64) -> f32 {
    let base_coord = Vec2::new(x, z);
    let mut amplitude = 1.0;
    let mut frequency = 0.05;
    let mut total = 0.0;
    let mut normalization = 0.0;

    for octave in 0..7 {
        let angle = 0.65 * octave as f32;
        let (sin, cos) = angle.sin_cos();
        let rotated = Vec2::new(
            base_coord.x * cos - base_coord.y * sin,
            base_coord.x * sin + base_coord.y * cos,
        );

        let offset = FRACTAL_OFFSETS[octave % FRACTAL_OFFSETS.len()];
        let octave_seed = seed ^ (octave as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);

        let warp = Vec2::new(
            value_noise(
                rotated.x * frequency * 1.37 + offset.0 * 0.37,
                rotated.y * frequency * 1.37 - offset.1 * 0.41,
                octave_seed ^ 0xA511_1A93_C7F5_4A39,
            ),
            value_noise(
                rotated.x * frequency * 1.91 - offset.1 * 0.29,
                rotated.y * frequency * 1.91 + offset.0 * 0.53,
                octave_seed ^ 0xD1B5_6F1D_7E91_BA27,
            ),
        ) * 2.0
            - Vec2::splat(1.0);

        let warped = rotated + warp * (6.5 * amplitude);

        let sample = value_noise(
            warped.x * frequency + offset.0,
            warped.y * frequency + offset.1,
            octave_seed,
        );

        total += sample * amplitude;
        normalization += amplitude;
        amplitude *= 0.48;
        frequency *= 1.92;
    }

    total / normalization
}

fn terrain_height(x: f32, z: f32, seed: u64) -> f32 {
    let broad = fractal_noise(x * 0.18, z * 0.18, seed ^ 0xFFAA_0041_33DD_EF77);
    let medium = fractal_noise(x * 0.52, z * 0.52, seed ^ 0x9E37_79B9_4C6D_2211);
    let detail = fractal_noise(x * 1.42, z * 1.42, seed ^ 0x7F4A_7C15_C3A5_992B);

    let smooth = (broad * 0.5 + medium * 0.33 + detail * 0.17).clamp(0.0, 1.0);
    let smooth_mapped = smooth * 2.0 - 1.0;

    let cliff_field = fractal_noise(
        x * 0.21 + 31.0,
        z * 0.21 - 47.0,
        seed ^ 0xACED_D00D_EE77_A119,
    );
    let cliff_noise = cliff_field * 2.0 - 1.0;
    let cliff_mask = cliff_noise.abs().powf(3.5).min(1.0);
    let cliffs = cliff_noise * cliff_mask;

    let base_height = (smooth_mapped * 0.75 + cliffs * 0.3) * TERRAIN_AMPLITUDE;

    let mountain_selector = fractal_noise(
        x * 0.052 - 83.0,
        z * 0.052 + 57.0,
        seed ^ 0xBEEF_BAAD_FEED_FACE,
    );
    let mountain_noise = mountain_selector * 2.0 - 1.0;
    let mountain_mask = ((mountain_noise + 0.28).max(0.0) / 1.28).powf(2.3);
    let mountain_soft = mountain_mask.powf(0.7);

    let ridge_field = fractal_noise(
        x * 0.24 + 13.0,
        z * 0.24 - 21.0,
        seed ^ 0xC0FE_BABE_E1E1_D00D,
    );
    let ridge = 1.0 - (ridge_field * 2.0 - 1.0).abs();
    let ridge_soft = ridge.powf(1.45);
    let ridge_sharp = ridge.powf(3.2);
    let ridge_profile = lerp(ridge_soft, ridge_sharp, mountain_mask.clamp(0.0, 1.0));

    let peak_detail = fractal_noise(x * 1.95, z * 1.95, seed ^ 0xDEAD_F00D_0C0A_AAAA);
    let peak_variation = 0.8 + (peak_detail * 2.0 - 1.0) * 0.2;

    let summit_detail = fractal_noise(x * 3.2, z * 3.2, seed ^ 0xFACE_FEED_CAFE_D00D);
    let summit_variation = 0.85 + (summit_detail * 2.0 - 1.0).max(0.0).powf(1.6) * 0.25;

    let mountain_profile = (mountain_soft * ridge_profile).powf(0.85);
    let mountain_height = mountain_profile
        * peak_variation
        * summit_variation
        * TERRAIN_AMPLITUDE
        * MOUNTAIN_HEIGHT_SCALE;

    base_height + mountain_height
}

fn generate_chunk_mesh(seed: u64, chunk_coord: IVec2) -> Mesh {
    let verts_per_side = TERRAIN_CHUNK_RESOLUTION + 1;
    let step = TERRAIN_CHUNK_SIZE / TERRAIN_CHUNK_RESOLUTION as f32;
    let origin_x = chunk_coord.x as f32 * TERRAIN_CHUNK_SIZE;
    let origin_z = chunk_coord.y as f32 * TERRAIN_CHUNK_SIZE;

    let mut positions: Vec<Vec3> = Vec::with_capacity(verts_per_side * verts_per_side);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(verts_per_side * verts_per_side);

    for row in 0..verts_per_side {
        let z = origin_z + row as f32 * step;
        for col in 0..verts_per_side {
            let x = origin_x + col as f32 * step;
            let y = terrain_height(x, z, seed);
            positions.push(Vec3::new(x, y, z));
            uvs.push([
                col as f32 / TERRAIN_CHUNK_RESOLUTION as f32,
                row as f32 / TERRAIN_CHUNK_RESOLUTION as f32,
            ]);
        }
    }

    let mut indices: Vec<u32> =
        Vec::with_capacity(TERRAIN_CHUNK_RESOLUTION * TERRAIN_CHUNK_RESOLUTION * 6);
    for row in 0..TERRAIN_CHUNK_RESOLUTION {
        for col in 0..TERRAIN_CHUNK_RESOLUTION {
            let i0 = row * verts_per_side + col;
            let i1 = i0 + 1;
            let i2 = i0 + verts_per_side;
            let i3 = i2 + 1;

            indices.extend_from_slice(&[
                i0 as u32, i2 as u32, i1 as u32, i1 as u32, i2 as u32, i3 as u32,
            ]);
        }
    }

    let normal_sample = step.max(0.5);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(positions.len());
    for position in &positions {
        let x = position.x;
        let z = position.z;

        let right = terrain_height(x + normal_sample, z, seed);
        let left = terrain_height(x - normal_sample, z, seed);
        let forward = terrain_height(x, z + normal_sample, seed);
        let back = terrain_height(x, z - normal_sample, seed);

        let tangent_x = Vec3::new(2.0 * normal_sample, right - left, 0.0);
        let tangent_z = Vec3::new(0.0, forward - back, 2.0 * normal_sample);
        let normal = tangent_z.cross(tangent_x).normalize_or_zero();

        normals.push([normal.x, normal.y, normal.z]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        positions
            .iter()
            .map(|p| [p.x, p.y, p.z])
            .collect::<Vec<[f32; 3]>>(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn world_to_chunk(x: f32, z: f32) -> IVec2 {
    IVec2::new(
        (x / TERRAIN_CHUNK_SIZE).floor() as i32,
        (z / TERRAIN_CHUNK_SIZE).floor() as i32,
    )
}

fn ensure_chunk(
    terrain_manager: &mut TerrainManager,
    pending_tasks: &mut PendingChunkTasks,
    coord: IVec2,
    seed: u64,
) -> bool {
    if terrain_manager.chunks.contains_key(&coord) || pending_tasks.tasks.contains_key(&coord) {
        return false;
    }

    let pool = AsyncComputeTaskPool::get();
    let task_coord = coord;
    let task_seed = seed;
    let task = pool.spawn(async move {
        let mesh = generate_chunk_mesh(task_seed, task_coord);
        Collider::from_bevy_mesh(&mesh, &ComputedColliderShape::TriMesh)
            .map(|collider| ChunkBuildResult { mesh, collider })
    });
    pending_tasks.tasks.insert(coord, task);
    true
}

fn spawn_chunk_entity(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    material: &Handle<StandardMaterial>,
    terrain_manager: &mut TerrainManager,
    coord: IVec2,
    result: ChunkBuildResult,
) {
    let mesh_handle = meshes.add(result.mesh);
    let entity = commands
        .spawn((
            PbrBundle {
                mesh: mesh_handle,
                material: material.clone(),
                ..default()
            },
            RigidBody::Fixed,
            result.collider,
            Friction::coefficient(1.0),
            Name::new(format!("Terrain Chunk ({}, {})", coord.x, coord.y)),
        ))
        .id();
    terrain_manager.chunks.insert(coord, entity);
}

fn generate_chunks_around(
    terrain_manager: &mut TerrainManager,
    pending_tasks: &mut PendingChunkTasks,
    center: IVec2,
    radius: i32,
    seed: u64,
) {
    for dz in -radius..=radius {
        for dx in -radius..=radius {
            let coord = IVec2::new(center.x + dx, center.y + dz);
            let _ = ensure_chunk(terrain_manager, pending_tasks, coord, seed);
        }
    }
}

fn update_terrain_chunks(
    mut commands: Commands,
    mut terrain_manager: ResMut<TerrainManager>,
    mut pending_tasks: ResMut<PendingChunkTasks>,
    terrain_seed: Res<TerrainSeed>,
    camera_query: Query<&Transform, With<OrbitCamera>>,
    vehicle_entities: Option<Res<VehicleEntities>>,
    vehicle_query: Query<&Transform, With<VehicleChassis>>,
) {
    let Ok(camera_transform) = camera_query.get_single() else {
        return;
    };

    let mut centers = Vec::new();
    centers.push(world_to_chunk(
        camera_transform.translation.x,
        camera_transform.translation.z,
    ));
    if let Some(vehicle_entities) = vehicle_entities {
        if let Ok(vehicle_transform) = vehicle_query.get(vehicle_entities.chassis) {
            centers.push(world_to_chunk(
                vehicle_transform.translation.x,
                vehicle_transform.translation.z,
            ));
        }
    }

    let mut required_chunks: HashSet<IVec2> = HashSet::new();
    let mut retained_chunks: HashSet<IVec2> = HashSet::new();

    for center in &centers {
        for dz in -TERRAIN_LOAD_RADIUS..=TERRAIN_LOAD_RADIUS {
            for dx in -TERRAIN_LOAD_RADIUS..=TERRAIN_LOAD_RADIUS {
                required_chunks.insert(IVec2::new(center.x + dx, center.y + dz));
            }
        }
        for dz in -TERRAIN_UNLOAD_RADIUS..=TERRAIN_UNLOAD_RADIUS {
            for dx in -TERRAIN_UNLOAD_RADIUS..=TERRAIN_UNLOAD_RADIUS {
                retained_chunks.insert(IVec2::new(center.x + dx, center.y + dz));
            }
        }
    }

    let seed = terrain_seed.0;

    let mut required_list: Vec<IVec2> = required_chunks.into_iter().collect();
    required_list.sort_by_key(|coord| {
        centers
            .iter()
            .map(|c| (coord.x - c.x).abs() + (coord.y - c.y).abs())
            .min()
            .unwrap_or(0)
    });

    let mut scheduled = 0usize;
    for coord in required_list {
        if ensure_chunk(&mut terrain_manager, &mut pending_tasks, coord, seed) {
            scheduled += 1;
            if scheduled >= MAX_CHUNK_TASKS_PER_FRAME {
                break;
            }
        }
    }

    let mut pending_to_remove = Vec::new();
    for coord in pending_tasks.tasks.keys() {
        if !retained_chunks.contains(coord) {
            pending_to_remove.push(*coord);
        }
    }
    for coord in pending_to_remove {
        pending_tasks.tasks.remove(&coord);
    }

    let mut to_remove = Vec::new();
    for (&coord, &entity) in terrain_manager.chunks.iter() {
        if !retained_chunks.contains(&coord) {
            to_remove.push((coord, entity));
        }
    }

    for (coord, entity) in to_remove {
        terrain_manager.chunks.remove(&coord);
        commands.entity(entity).despawn_recursive();
    }
}

fn apply_completed_chunk_tasks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    terrain_assets: Res<TerrainAssets>,
    mut terrain_manager: ResMut<TerrainManager>,
    mut pending_tasks: ResMut<PendingChunkTasks>,
) {
    let mut finished = Vec::new();
    pending_tasks.tasks.retain(|coord, task| {
        if let Some(result) = block_on(poll_once(task)) {
            finished.push((*coord, result));
            false
        } else {
            true
        }
    });

    for (coord, maybe_result) in finished {
        if terrain_manager.chunks.contains_key(&coord) {
            continue;
        }
        if let Some(result) = maybe_result {
            spawn_chunk_entity(
                &mut commands,
                &mut meshes,
                &terrain_assets.material,
                &mut terrain_manager,
                coord,
                result,
            );
        }
    }
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::rgb(0.47, 0.7, 1.0)))
        .insert_resource(CameraZoomSettings {
            min_distance: 2.0,
            max_distance: 100.0,
            zoom_speed: 5.75,
        })
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Terrain Runner".into(),
                resolution: (1280., 720.).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(RapierDebugRenderPlugin::default().disabled())
        .insert_resource(RapierConfiguration {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            ..default()
        })
        .add_systems(Startup, (apply_time_scale, setup))
        .add_systems(
            Update,
            (
                update_terrain_chunks,
                apply_completed_chunk_tasks,
                reset_scene_when_requested,
                handle_camera_orbit,
                control_vehicle_movement,
                zoom_camera,
                update_vehicle_hud_text,
                update_terrain_debug_text,
                toggle_debug_render,
            ),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    zoom_settings: Res<CameraZoomSettings>,
) {
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 175.0,
    });

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            shadows_enabled: true,
            illuminance: 35_000.0,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.7, 0.0)),
        ..default()
    });

    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 3200.0,
            range: 50.0,
            ..default()
        },
        transform: Transform::from_xyz(6.0, 8.0, 6.0),
        ..default()
    });

    let terrain_seed = TerrainSeed(generate_seed());
    let seed_value = terrain_seed.0;
    commands.insert_resource(terrain_seed);

    let terrain_material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.15, 0.5, 0.18),
        perceptual_roughness: 1.0,
        metallic: 0.0,
        ..default()
    });
    commands.insert_resource(TerrainAssets {
        material: terrain_material.clone(),
    });

    let mut terrain_manager = TerrainManager::default();
    let mut pending_tasks = PendingChunkTasks::default();
    let camera_start = camera_start_position();
    let center_chunk = world_to_chunk(camera_start.x, camera_start.z);

    let center_mesh = generate_chunk_mesh(seed_value, center_chunk);
    if let Some(center_collider) =
        Collider::from_bevy_mesh(&center_mesh, &ComputedColliderShape::TriMesh)
    {
        spawn_chunk_entity(
            &mut commands,
            &mut meshes,
            &terrain_material,
            &mut terrain_manager,
            center_chunk,
            ChunkBuildResult {
                mesh: center_mesh,
                collider: center_collider,
            },
        );
    }

    generate_chunks_around(
        &mut terrain_manager,
        &mut pending_tasks,
        center_chunk,
        TERRAIN_LOAD_RADIUS,
        seed_value,
    );
    commands.insert_resource(terrain_manager);
    commands.insert_resource(pending_tasks);

    let chassis_half_extents = Vec3::new(1.15, 0.35, 2.1);
    let wheel_positions = [
        Vec3::new(0.95, -0.1, -1.45),
        Vec3::new(-0.95, -0.1, -1.45),
        Vec3::new(0.95, -0.1, 1.45),
        Vec3::new(-0.95, -0.1, 1.45),
    ];
    let steering_flags = [true, true, false, false];
    let drive_flags = [true, true, true, true];
    let chassis_mesh = meshes.add(build_car_body_mesh(chassis_half_extents));
    let tire_mesh = meshes.add(build_tire_mesh(WHEEL_RADIUS_DEFAULT, WHEEL_WIDTH_DEFAULT));
    let rim_mesh = meshes.add(
        CylinderMeshBuilder::new(WHEEL_RADIUS_DEFAULT * 0.58, WHEEL_WIDTH_DEFAULT * 0.72, 36)
            .segments(1)
            .build(),
    );
    let hub_mesh = meshes.add(
        CylinderMeshBuilder::new(WHEEL_RADIUS_DEFAULT * 0.28, WHEEL_WIDTH_DEFAULT * 0.62, 24)
            .segments(1)
            .build(),
    );
    let spoke_mesh = meshes.add(Mesh::from(Cuboid::new(
        WHEEL_RADIUS_DEFAULT * 0.8,
        WHEEL_WIDTH_DEFAULT * 0.18,
        WHEEL_RADIUS_DEFAULT * 0.1,
    )));
    let lug_mesh = meshes.add(
        CylinderMeshBuilder::new(
            WHEEL_RADIUS_DEFAULT * 0.085,
            WHEEL_WIDTH_DEFAULT * 0.14,
            20,
        )
        .segments(1)
        .build(),
    );
    let hub_axis_mesh = meshes.add(
        CylinderMeshBuilder::new(
            WHEEL_RADIUS_DEFAULT * 0.22,
            WHEEL_WIDTH_DEFAULT * 0.18,
            26,
        )
        .segments(1)
        .build(),
    );

    let chassis_material = materials.add(StandardMaterial {
        base_color: chassis_base_color(),
        metallic: 0.35,
        perceptual_roughness: 0.45,
        ..default()
    });
    let tire_material = materials.add(StandardMaterial {
        base_color: wheel_base_color(),
        metallic: 0.2,
        perceptual_roughness: 0.85,
        cull_mode: None,
        ..default()
    });
    let rim_material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.72, 0.73, 0.78),
        metallic: 0.85,
        perceptual_roughness: 0.16,
        reflectance: 0.6,
        ..default()
    });
    let spoke_material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.26, 0.28, 0.32),
        metallic: 0.75,
        perceptual_roughness: 0.22,
        ..default()
    });
    let lug_material = materials.add(StandardMaterial {
        base_color: wheel_lug_color(),
        metallic: 0.9,
        perceptual_roughness: 0.18,
        reflectance: 0.7,
        ..default()
    });
    let axis_material = materials.add(StandardMaterial {
        base_color: wheel_axis_color(),
        metallic: 0.6,
        perceptual_roughness: 0.35,
        ..default()
    });

    let car_spawn_height = terrain_height(0.0, 0.0, seed_value) + 2.5;
    let car_spawn_position = Vec3::new(0.0, car_spawn_height, 0.0);

    let chassis_entity = commands
        .spawn((
            PbrBundle {
                mesh: chassis_mesh.clone(),
                material: chassis_material.clone(),
                transform: Transform::from_translation(car_spawn_position),
                ..default()
            },
            RigidBody::Dynamic,
            Collider::cuboid(
                chassis_half_extents.x,
                chassis_half_extents.y,
                chassis_half_extents.z,
            ),
            AdditionalMassProperties::Mass(1200.0),
            Restitution::coefficient(0.05),
            Friction {
                coefficient: 1.0,
                combine_rule: CoefficientCombineRule::Average,
            },
            Damping {
                linear_damping: 0.22,
                angular_damping: 0.38,
            },
            Velocity::zero(),
            ExternalForce::default(),
            ExternalImpulse::default(),
            VehicleChassis,
            Name::new("Explorer Chassis"),
        ))
        .id();

    let wheel_rotation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
    let mut wheel_entities: Vec<Entity> = Vec::with_capacity(WHEEL_COUNT);
    commands.entity(chassis_entity).with_children(|builder| {
        for (index, offset) in wheel_positions.iter().enumerate() {
            let steering = steering_flags[index];
            let drive = drive_flags[index];
            let wheel_label = match index {
                0 => "Front-Right",
                1 => "Front-Left",
                2 => "Rear-Right",
                _ => "Rear-Left",
            };
            let wheel_entity = builder
                .spawn((
                    PbrBundle {
                        mesh: tire_mesh.clone(),
                        material: tire_material.clone(),
                        transform: Transform {
                            translation: *offset - Vec3::Y * SUSPENSION_REST_LENGTH,
                            rotation: wheel_rotation,
                            scale: Vec3::ONE,
                        },
                        ..default()
                    },
                    VehicleWheel {
                        local_offset: *offset,
                        radius: WHEEL_RADIUS_DEFAULT,
                        rest_length: SUSPENSION_REST_LENGTH,
                        stiffness: SUSPENSION_STIFFNESS,
                        damping: SUSPENSION_DAMPING,
                        steering,
                        drive,
                        health: WHEEL_MAX_HEALTH,
                        detached: false,
                        compression: 0.0,
                        time_since_grounded: 0.0,
                        spin_angle: 0.0,
                        spin_velocity: 0.0,
                        mesh_handle: tire_mesh.clone(),
                        material_handle: tire_material.clone(),
                    },
                    WheelVisual,
                    Name::new(format!("{} Wheel", wheel_label)),
                ))
                .with_children(|wheel_builder| {
                    wheel_builder.spawn((
                        PbrBundle {
                            mesh: rim_mesh.clone(),
                            material: rim_material.clone(),
                            transform: Transform::IDENTITY,
                            ..default()
                        },
                        Name::new(format!("{} Rim", wheel_label)),
                    ));
                    wheel_builder.spawn((
                        PbrBundle {
                            mesh: hub_mesh.clone(),
                            material: rim_material.clone(),
                            transform: Transform::IDENTITY,
                            ..default()
                        },
                        Name::new(format!("{} Hub", wheel_label)),
                    ));
                    for spoke_index in 0..WHEEL_SPOKE_COUNT {
                        let angle =
                            (spoke_index as f32 / WHEEL_SPOKE_COUNT as f32) * TAU;
                        let rotation = Quat::from_axis_angle(Vec3::Y, angle);
                        wheel_builder.spawn((
                            PbrBundle {
                                mesh: spoke_mesh.clone(),
                                material: spoke_material.clone(),
                                transform: Transform {
                                    translation: Vec3::ZERO,
                                    rotation,
                                    scale: Vec3::ONE,
                                },
                                ..default()
                            },
                            Name::new(format!(
                                "{} Spoke {}",
                                wheel_label,
                                spoke_index + 1,
                            )),
                        ));
                    }

                    let detail_face_offset = WHEEL_WIDTH_DEFAULT * 0.46;
                    let face_offset = if offset.x >= 0.0 {
                        -detail_face_offset
                    } else {
                        detail_face_offset
                    };
                    let center_translation = Vec3::new(0.0, face_offset * 0.6, 0.0);
                    wheel_builder.spawn((
                        PbrBundle {
                            mesh: hub_axis_mesh.clone(),
                            material: axis_material.clone(),
                            transform: Transform {
                                translation: center_translation,
                                scale: Vec3::new(0.88, 0.32, 0.88),
                                ..default()
                            },
                            ..default()
                        },
                        Name::new(format!("{} Axis Cap", wheel_label)),
                    ));

                    let lug_ring_radius = WHEEL_RADIUS_DEFAULT * 0.38;
                    for lug_index in 0..4 {
                        let angle = TAU * (lug_index as f32) / 4.0;
                        let lug_translation = Vec3::new(
                            lug_ring_radius * angle.cos(),
                            face_offset,
                            lug_ring_radius * angle.sin(),
                        );
                        wheel_builder.spawn((
                            PbrBundle {
                                mesh: lug_mesh.clone(),
                                material: lug_material.clone(),
                                transform: Transform {
                                    translation: lug_translation,
                                    scale: Vec3::new(1.0, 0.25, 1.0),
                                    ..default()
                                },
                                ..default()
                            },
                            Name::new(format!("{} Lug {}", wheel_label, lug_index + 1)),
                        ));
                    }
                })
                .id();
            wheel_entities.push(wheel_entity);
        }
    });

    let wheel_entities: [Entity; WHEEL_COUNT] =
        wheel_entities.try_into().expect("Exactly four wheels");
    commands.insert_resource(VehicleEntities {
        chassis: chassis_entity,
        wheels: wheel_entities,
    });
    commands.insert_resource(VehicleControlInput::default());
    commands.insert_resource(VehicleStatus {
        speed_mps: 0.0,
        wheels_attached: WHEEL_COUNT,
    });

    let camera_start = camera_start_position();
    let target = CAMERA_TARGET;
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(camera_start).looking_at(target, Vec3::Y),
            ..default()
        },
        OrbitCamera,
        Name::new("Main Camera"),
    ));

    let initial_offset = camera_start - car_spawn_position;
    let initial_distance = initial_offset.length();
    let clamped_distance =
        initial_distance.clamp(zoom_settings.min_distance, zoom_settings.max_distance);
    let offset_local = if initial_distance > f32::EPSILON {
        initial_offset.normalize() * clamped_distance
    } else {
        Vec3::new(0.0, zoom_settings.min_distance, zoom_settings.min_distance)
    };
    commands.insert_resource(ChaseCameraState {
        offset_local,
        follow_speed: 3.0,
    });

    let mut overlay_camera = commands.spawn(Camera2dBundle::default());
    overlay_camera
        .insert(Camera {
            order: 1,
            clear_color: ClearColorConfig::None,
            ..default()
        })
        .insert(RenderLayers::layer(1));

    commands.spawn((
        TextBundle {
            text: Text::from_section(
                "Vehicle: Speed -- m/s | Wheels 4/4 | Press R to reset",
                TextStyle {
                    font: Handle::<Font>::default(),
                    font_size: 18.0,
                    color: Color::WHITE,
                },
            ),
            style: Style {
                position_type: PositionType::Absolute,
                top: Val::Px(10.0),
                left: Val::Px(10.0),
                ..default()
            },
            background_color: BackgroundColor(Color::rgba(0.2, 0.0, 0.0, 0.7)),
            ..default()
        },
        RenderLayers::layer(1),
        VehicleHudText,
    ));

    commands.spawn((
        TextBundle {
            text: Text::from_section(
                "Terrain Debug: initializing...",
                TextStyle {
                    font: Handle::<Font>::default(),
                    font_size: 16.0,
                    color: Color::rgb(0.9, 0.95, 1.0),
                },
            ),
            style: Style {
                position_type: PositionType::Absolute,
                top: Val::Px(42.0),
                left: Val::Px(10.0),
                ..default()
            },
            background_color: BackgroundColor(Color::rgba(0.05, 0.05, 0.2, 0.65)),
            ..default()
        },
        RenderLayers::layer(1),
        TerrainDebugText,
    ));
}

fn clamp_camera_offset(offset: Vec3) -> Vec3 {
    let min_height = CAMERA_MIN_HEIGHT_ABOVE_TARGET;
    let radius = offset.length();
    if radius <= f32::EPSILON {
        return Vec3::new(0.0, min_height, 0.0);
    }

    if offset.y >= min_height {
        return offset;
    }

    if radius <= min_height {
        return offset.normalize() * min_height;
    }

    let horizontal = Vec3::new(offset.x, 0.0, offset.z);
    let horizontal_len = horizontal.length();
    let new_horizontal_mag = (radius * radius - min_height * min_height)
        .max(0.0)
        .sqrt();
    let horizontal_dir = if horizontal_len > 1e-6 {
        horizontal / horizontal_len
    } else {
        Vec3::Z
    };
    horizontal_dir * new_horizontal_mag + Vec3::Y * min_height
}

fn handle_camera_orbit(
    buttons: Res<ButtonInput<MouseButton>>,
    time: Res<Time>,
    mut motion_events: EventReader<MouseMotion>,
    vehicle_entities: Option<Res<VehicleEntities>>,
    vehicle_query: Query<&GlobalTransform, With<VehicleChassis>>,
    mut camera_query: Query<&mut Transform, With<OrbitCamera>>,
    mut chase_state: ResMut<ChaseCameraState>,
) {
    let Some(vehicle_entities) = vehicle_entities else {
        motion_events.clear();
        return;
    };
    let Ok(target_transform) = vehicle_query.get(vehicle_entities.chassis) else {
        motion_events.clear();
        return;
    };

    let mut cumulative_delta = Vec2::ZERO;
    for event in motion_events.read() {
        cumulative_delta += event.delta;
    }

    if let Ok(mut camera_transform) = camera_query.get_single_mut() {
        let car_transform = target_transform.compute_transform();
        let target = car_transform.translation;

        if chase_state.offset_local.length_squared() < 1e-6 {
            chase_state.offset_local = Vec3::new(0.0, 3.0, 6.0);
        }

        let mut desired_offset_world = car_transform.rotation * chase_state.offset_local;
        desired_offset_world = clamp_camera_offset(desired_offset_world);
        let desired_position = target + desired_offset_world;

        if cumulative_delta != Vec2::ZERO && buttons.pressed(MouseButton::Left) {
            let mut current_offset = camera_transform.translation - target;
            if current_offset.length_squared() < f32::EPSILON {
                current_offset = desired_offset_world;
            }
            let radius = current_offset.length().max(0.1);
            let yaw_sensitivity = 0.01;
            let pitch_sensitivity = 0.01;

            let yaw_rot = Quat::from_rotation_y(-cumulative_delta.x * yaw_sensitivity);
            let mut desired_offset = yaw_rot * current_offset;

            let right_axis = desired_offset.cross(Vec3::Y);
            if right_axis.length_squared() > 1e-6 {
                let pitch_rot = Quat::from_axis_angle(
                    right_axis.normalize(),
                    -cumulative_delta.y * pitch_sensitivity,
                );
                desired_offset = pitch_rot * desired_offset;
            }

            if desired_offset.length_squared() > f32::EPSILON {
                desired_offset = desired_offset.normalize() * radius;
            }

            desired_offset = clamp_camera_offset(desired_offset);
            camera_transform.translation = target + desired_offset;
            let inverse_rot = car_transform.rotation.inverse();
            chase_state.offset_local = inverse_rot * desired_offset;
        } else {
            let follow_factor =
                (1.0 - (-time.delta_seconds() * chase_state.follow_speed).exp()).clamp(0.0, 1.0);
            camera_transform.translation = camera_transform
                .translation
                .lerp(desired_position, follow_factor);
            let inverse_rot = car_transform.rotation.inverse();
            chase_state.offset_local = inverse_rot * desired_offset_world;
        }

        camera_transform.look_at(target, Vec3::Y);
    }
}
fn control_vehicle_movement(
    mut commands: Commands,
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut control: ResMut<VehicleControlInput>,
    mut status: ResMut<VehicleStatus>,
    vehicle_entities: Option<Res<VehicleEntities>>,
    mut chassis_query: Query<
        (&GlobalTransform, &mut ExternalForce, &Velocity),
        With<VehicleChassis>,
    >,
    mut wheel_query: Query<
        (Entity, &mut VehicleWheel, &mut Transform, &mut Visibility),
        With<WheelVisual>,
    >,
    rapier_context: Res<RapierContext>,
) {
    let Some(vehicle_entities) = vehicle_entities else {
        return;
    };
    let Ok((chassis_transform, mut external_force, velocity)) =
        chassis_query.get_mut(vehicle_entities.chassis)
    else {
        return;
    };

    let dt = time.delta_seconds();

    let mut steering_target: f32 = 0.0;
    if keys.pressed(KeyCode::KeyA) {
        steering_target += 1.0;
    }
    if keys.pressed(KeyCode::KeyD) {
        steering_target -= 1.0;
    }

    let mut throttle_target: f32 = 0.0;
    if keys.pressed(KeyCode::KeyW) {
        throttle_target += 1.0;
    }
    if keys.pressed(KeyCode::KeyS) {
        throttle_target -= 1.0;
    }
    let handbrake_target: f32 = if keys.pressed(KeyCode::Space) {
        1.0
    } else {
        0.0
    };

    let steering_rate = (STEERING_SPEED * dt).clamp(0.0, 1.0);
    let steering_target_clamped = steering_target.clamp(-1.0, 1.0);
    control.steering += (steering_target_clamped - control.steering) * steering_rate;
    control.throttle = throttle_target.clamp(-1.0, 1.0);
    control.brake = handbrake_target.clamp(0.0, 1.0);

    let handbrake_input = control.brake;
    let handbrake_active = handbrake_input > 0.0;
    if handbrake_input > 0.0 {
        control.throttle = 0.0;
    }

    external_force.force = Vec3::ZERO;
    external_force.torque = Vec3::ZERO;

    let chassis_translation = chassis_transform.translation();
    let (_, chassis_rotation, _) = chassis_transform.to_scale_rotation_translation();
    let up = (chassis_rotation * Vec3::Y).normalize();
    let forward = (chassis_rotation * -Vec3::Z).normalize();

    let base_wheel_rotation = wheel_base_rotation();
    let mut wheels_attached = 0usize;

    for &wheel_entity in &vehicle_entities.wheels {
        if let Ok((_, mut wheel, mut wheel_transform, mut visibility)) =
            wheel_query.get_mut(wheel_entity)
        {
            if wheel.detached {
                wheel.spin_velocity = 0.0;
                *visibility = Visibility::Hidden;
                continue;
            }

            *visibility = Visibility::Visible;
            wheels_attached += 1;

            let steer_angle = if wheel.steering {
                control.steering * STEERING_MAX_ANGLE
            } else {
                0.0
            };
            let steer_rotation = Quat::from_axis_angle(Vec3::Y, steer_angle);

            let suspension_anchor = chassis_translation + chassis_rotation * wheel.local_offset;
            let ray_dir = -up;
            let max_distance = wheel.rest_length + wheel.radius + 1.0;
            let filter = QueryFilter::default().exclude_rigid_body(vehicle_entities.chassis);

            let mut compression = wheel.compression;
            let mut suspension_point = suspension_anchor - up * (wheel.rest_length - compression);
            let mut contact_ratio = 0.0;
            let mut point_velocity = velocity.linvel;
            let mut grounded = false;

            if let Some((hit_entity, toi)) =
                rapier_context.cast_ray(suspension_anchor, ray_dir, max_distance, true, filter)
            {
                if hit_entity != vehicle_entities.chassis {
                    let distance = toi.max(0.0);
                    let contact_distance = (distance - wheel.radius).max(0.0);
                    compression = (wheel.rest_length - contact_distance)
                        .clamp(-wheel.rest_length, wheel.rest_length * 1.5);
                    let compression_positive = compression.max(0.0);
                    contact_ratio = (compression_positive / wheel.rest_length).clamp(0.0, 1.0);

                    suspension_point = suspension_anchor - up * (wheel.rest_length - compression);
                    let r = suspension_point - chassis_translation;
                    point_velocity = velocity.linvel + velocity.angvel.cross(r);
                    let vertical_speed = point_velocity.dot(up);

                    let spring_force = compression_positive * wheel.stiffness;
                    let damping_force = vertical_speed * wheel.damping;
                    let suspension_force = up * (spring_force - damping_force);

                    external_force.force += suspension_force;
                    external_force.torque +=
                        (suspension_point - chassis_translation).cross(suspension_force);

                    let previous_air_time = wheel.time_since_grounded;
                    wheel.time_since_grounded = 0.0;
                    grounded = compression_positive > 0.0;

                    if grounded && previous_air_time > 0.25 {
                        let impact_velocity = vertical_speed.abs();
                        let severity =
                            impact_velocity + compression_positive / wheel.rest_length * 22.0;
                        if severity > CRASH_VELOCITY_THRESHOLD
                            || compression_positive / wheel.rest_length
                                > CRASH_COMPRESSION_THRESHOLD
                        {
                            wheel.health -= severity * 0.8;
                        }
                    }
                }
            }

            if !grounded {
                wheel.time_since_grounded += dt;
                compression =
                    (compression - dt * 1.2).clamp(-wheel.rest_length, wheel.rest_length);
                contact_ratio = 0.0;
            }

            wheel.compression = compression;

            let suspension_extension = (wheel.rest_length - wheel.compression)
                .clamp(-wheel.rest_length, wheel.rest_length * 1.5);
            wheel_transform.translation = wheel.local_offset - Vec3::Y * suspension_extension;

            let mut wheel_forward = steer_rotation * forward;
            if let Some(normalized) = wheel_forward.try_normalize() {
                wheel_forward = normalized;
            } else {
                wheel_forward = forward;
            }
            let mut wheel_right = wheel_forward.cross(up);
            if let Some(normalized) = wheel_right.try_normalize() {
                wheel_right = normalized;
            } else {
                wheel_right = wheel_forward
                    .cross(Vec3::Y)
                    .try_normalize()
                    .unwrap_or(Vec3::X);
            }

            let long_speed = point_velocity.dot(wheel_forward);
            let lat_speed = point_velocity.dot(wheel_right);

            if contact_ratio > 0.0 {
                let drive_force = if wheel.drive && !handbrake_active {
                    wheel_forward * control.throttle * ENGINE_FORCE
                } else {
                    Vec3::ZERO
                };

                let brake_direction = if long_speed.abs() > 0.5 {
                    long_speed.signum()
                } else {
                    long_speed / 0.5
                };
                let brake_force = if handbrake_active {
                    -wheel_forward * brake_direction * handbrake_input * HANDBRAKE_FORCE
                } else {
                    Vec3::ZERO
                };

                let lateral_force = -wheel_right * lat_speed * LATERAL_FRICTION_FORCE;

                let traction_force = (drive_force + brake_force + lateral_force) * contact_ratio;

                external_force.force += traction_force;
                external_force.torque +=
                    (suspension_point - chassis_translation).cross(traction_force);
            }

            let mut smoothing_rate = if grounded {
                WHEEL_SPIN_TRACK_RATE
            } else {
                WHEEL_SPIN_AIR_RATE
            };
            if handbrake_active {
                let speed_abs = long_speed.abs();
                let extra = if speed_abs < HANDBRAKE_MIN_SPEED {
                    (HANDBRAKE_MIN_SPEED / speed_abs.max(0.2)).clamp(1.0, 6.0)
                } else {
                    1.0
                };
                smoothing_rate = HANDBRAKE_BLEED_RATE * extra.max(1.0);
            }
            let target_spin_speed = if handbrake_active { 0.0 } else { long_speed };
            let blend = (smoothing_rate * dt).clamp(0.0, 1.0);
            wheel.spin_velocity += (target_spin_speed - wheel.spin_velocity) * blend;

            let spin_delta = (wheel.spin_velocity / wheel.radius) * dt;
            wheel.spin_angle = (wheel.spin_angle + spin_delta).rem_euclid(TAU);
            let spin_rotation = Quat::from_axis_angle(Vec3::Y, wheel.spin_angle);
            wheel_transform.rotation =
                steer_rotation * (base_wheel_rotation * spin_rotation);

            if wheel.health <= 0.0 {
                wheel.detached = true;
                wheel.health = 0.0;
                wheel.time_since_grounded = 0.0;
                wheel.spin_velocity = 0.0;
                *visibility = Visibility::Hidden;

                let world_rotation = chassis_rotation * wheel_transform.rotation;
                let debris_transform = Transform {
                    translation: suspension_point,
                    rotation: world_rotation,
                    scale: Vec3::ONE,
                };

                commands.spawn((
                    PbrBundle {
                        mesh: wheel.mesh_handle.clone(),
                        material: wheel.material_handle.clone(),
                        transform: debris_transform,
                        ..default()
                    },
                    RigidBody::Dynamic,
                    Collider::ball(wheel.radius * 0.92),
                    AdditionalMassProperties::Mass(25.0),
                    Restitution::coefficient(0.25),
                    Friction {
                        coefficient: 1.1,
                        combine_rule: CoefficientCombineRule::Average,
                    },
                    Damping {
                        linear_damping: 0.25,
                        angular_damping: 0.25,
                    },
                    Velocity {
                        linvel: velocity.linvel + up * 2.5,
                        angvel: velocity.angvel,
                    },
                    ExternalImpulse::default(),
                    Name::new("Detached Wheel"),
                ));
            }
        }
    }

    let new_speed = velocity.linvel.length();
    if (status.speed_mps - new_speed).abs() > 0.01 {
        status.speed_mps = new_speed;
    }
    if status.wheels_attached != wheels_attached {
        status.wheels_attached = wheels_attached;
    }
}

fn zoom_camera(
    mut scroll_events: EventReader<MouseWheel>,
    zoom: Res<CameraZoomSettings>,
    vehicle_entities: Option<Res<VehicleEntities>>,
    vehicle_query: Query<&GlobalTransform, With<VehicleChassis>>,
    mut camera_query: Query<&mut Transform, With<OrbitCamera>>,
    mut chase_state: ResMut<ChaseCameraState>,
) {
    let mut scroll_delta = 0.0f32;

    for event in scroll_events.read() {
        let step = match event.unit {
            MouseScrollUnit::Line => event.y,
            MouseScrollUnit::Pixel => event.y * 0.1,
        };
        scroll_delta += step;
    }

    if scroll_delta.abs() <= f32::EPSILON {
        return;
    }

    let Some(vehicle_entities) = vehicle_entities else {
        return;
    };
    let Ok(target_transform) = vehicle_query.get(vehicle_entities.chassis) else {
        return;
    };

    let Ok(mut camera_transform) = camera_query.get_single_mut() else {
        return;
    };

    let car_transform = target_transform.compute_transform();
    let target = car_transform.translation;

    let current_distance = chase_state.offset_local.length().max(f32::EPSILON);
    let zoom_offset = scroll_delta * zoom.zoom_speed;
    let new_distance = (current_distance - zoom_offset).clamp(zoom.min_distance, zoom.max_distance);

    let mut dir_local = chase_state.offset_local.normalize_or_zero();
    if dir_local == Vec3::ZERO {
        dir_local = Vec3::new(0.0, 0.0, 1.0);
    }
    chase_state.offset_local = dir_local * new_distance;

    let desired_offset_world = car_transform.rotation * chase_state.offset_local;
    camera_transform.translation = target + desired_offset_world;
    camera_transform.look_at(target, Vec3::Y);
}

fn reset_scene_when_requested(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    terrain_assets: Res<TerrainAssets>,
    mut terrain_seed: ResMut<TerrainSeed>,
    mut terrain_manager: ResMut<TerrainManager>,
    mut pending_tasks: ResMut<PendingChunkTasks>,
    keys: Res<ButtonInput<KeyCode>>,
    vehicle_entities: Option<Res<VehicleEntities>>,
    mut vehicle_status: ResMut<VehicleStatus>,
    mut vehicle_control: ResMut<VehicleControlInput>,
    mut chase_state: ResMut<ChaseCameraState>,
    mut queries: ParamSet<(
        Query<
            (
                &mut Transform,
                &mut GlobalTransform,
                &mut Velocity,
                &mut ExternalForce,
                &mut ExternalImpulse,
                &RapierRigidBodyHandle,
            ),
            (With<VehicleChassis>, Without<OrbitCamera>),
        >,
        Query<(&mut VehicleWheel, &mut Transform, &mut Visibility), With<WheelVisual>>,
        Query<&Transform, (With<OrbitCamera>, Without<VehicleChassis>)>,
    )>,
    mut context: ResMut<RapierContext>,
) {
    if !keys.just_pressed(KeyCode::KeyR) {
        return;
    }

    let new_seed = generate_seed();
    terrain_seed.0 = new_seed;

    for &entity in terrain_manager.chunks.values() {
        commands.entity(entity).despawn_recursive();
    }
    terrain_manager.chunks.clear();
    pending_tasks.tasks.clear();

    let camera_start = camera_start_position();
    let center_chunk = world_to_chunk(camera_start.x, camera_start.z);
    let center_mesh = generate_chunk_mesh(new_seed, center_chunk);
    if let Some(center_collider) =
        Collider::from_bevy_mesh(&center_mesh, &ComputedColliderShape::TriMesh)
    {
        spawn_chunk_entity(
            &mut commands,
            &mut meshes,
            &terrain_assets.material,
            &mut terrain_manager,
            center_chunk,
            ChunkBuildResult {
                mesh: center_mesh,
                collider: center_collider,
            },
        );
    }

    generate_chunks_around(
        &mut terrain_manager,
        &mut pending_tasks,
        center_chunk,
        TERRAIN_LOAD_RADIUS,
        new_seed,
    );
    let scale = context.physics_scale();

    if let Some(vehicle_entities) = vehicle_entities {
        let mut spawn_position_opt = None;
        let mut car_rotation_opt = None;

        {
            let mut chassis_query = queries.p0();
            if let Ok((
                mut transform,
                mut global_transform,
                mut velocity,
                mut external_force,
                mut external_impulse,
                handle,
            )) = chassis_query.get_mut(vehicle_entities.chassis)
            {
                let car_spawn_height = terrain_height(0.0, 0.0, new_seed) + 2.5;
                let spawn_position = Vec3::new(0.0, car_spawn_height, 0.0);
                spawn_position_opt = Some(spawn_position);
                car_rotation_opt = Some(transform.rotation);

                let reset_transform = Transform::from_translation(spawn_position);
                *transform = reset_transform;
                *global_transform = GlobalTransform::from(reset_transform);
                velocity.linvel = Vec3::ZERO;
                velocity.angvel = Vec3::ZERO;
                external_force.force = Vec3::ZERO;
                external_force.torque = Vec3::ZERO;
                external_impulse.impulse = Vec3::ZERO;
                external_impulse.torque_impulse = Vec3::ZERO;

                if let Some(body) = context.bodies.get_mut(handle.0) {
                    body.set_translation(
                        Vector::new(
                            spawn_position.x / scale,
                            spawn_position.y / scale,
                            spawn_position.z / scale,
                        ),
                        true,
                    );
                    body.set_linvel(Vector::zeros(), true);
                    body.set_angvel(AngVector::zeros(), true);
                    body.set_rotation(UnitQuaternion::identity(), true);
                }
            }
        }

        if let Some(spawn_position) = spawn_position_opt {
            {
                let mut wheel_query = queries.p1();
                for &wheel_entity in &vehicle_entities.wheels {
                    if let Ok((mut wheel, mut wheel_transform, mut visibility)) =
                        wheel_query.get_mut(wheel_entity)
                    {
                        wheel.detached = false;
                        wheel.health = WHEEL_MAX_HEALTH;
                        wheel.compression = 0.0;
                        wheel.time_since_grounded = 0.0;
                        wheel.spin_angle = 0.0;
                        wheel.spin_velocity = 0.0;
                        *visibility = Visibility::Visible;
                        let base_rotation = wheel_base_rotation();
                        wheel_transform.translation =
                            wheel.local_offset - Vec3::Y * wheel.rest_length;
                        wheel_transform.rotation = base_rotation;
                    }
                }
            }

            if let Some(car_rotation) = car_rotation_opt {
                if let Ok(camera_transform) = queries.p2().get_single() {
                    let camera_pos = camera_transform.translation;
                    let relative = camera_pos - spawn_position;
                    chase_state.offset_local = car_rotation.inverse() * relative;
                }
            }
        }
    }

    vehicle_status.speed_mps = 0.0;
    vehicle_status.wheels_attached = WHEEL_COUNT;
    vehicle_control.steering = 0.0;
    vehicle_control.throttle = 0.0;
    vehicle_control.brake = 0.0;
}

fn update_vehicle_hud_text(
    status: Res<VehicleStatus>,
    mut text_query: Query<&mut Text, With<VehicleHudText>>,
) {
    if !status.is_changed() {
        return;
    }

    let Ok(mut text) = text_query.get_single_mut() else {
        return;
    };

    let speed_kmh = status.speed_mps * 3.6;
    let mut hud_line = format!(
        "Vehicle: Speed {:+05.1} km/h | Wheels {}/{} | Press R to reset",
        speed_kmh, status.wheels_attached, WHEEL_COUNT,
    );

    if status.wheels_attached < WHEEL_COUNT {
        hud_line.push_str(" | DAMAGE DETECTED");
    }

    text.sections[0].value = hud_line;
}

fn update_terrain_debug_text(
    terrain_manager: Res<TerrainManager>,
    pending_tasks: Res<PendingChunkTasks>,
    terrain_seed: Res<TerrainSeed>,
    time: Res<Time>,
    camera_query: Query<&Transform, With<OrbitCamera>>,
    debug_context: Option<Res<DebugRenderContext>>,
    mut text_query: Query<&mut Text, With<TerrainDebugText>>,
) {
    let Ok(mut text) = text_query.get_single_mut() else {
        return;
    };

    let chunk_count = terrain_manager.chunks.len();
    let pending_count = pending_tasks.tasks.len();

    let camera_height = camera_query
        .get_single()
        .map(|transform| transform.translation.y)
        .unwrap_or(0.0);

    let dt = time.delta_seconds();
    let fps = if dt > f32::EPSILON { 1.0 / dt } else { 0.0 };

    let debug_enabled = debug_context.map(|ctx| ctx.enabled).unwrap_or_default();

    text.sections[0].value = format!(
        "Seed {:016X}\nChunks {:03} | Pending {:03}\nCamera Y {:>6.2} | FPS {:>5.1}\nMountain Scale {:.2} | Debug (X) {}",
        terrain_seed.0,
        chunk_count,
        pending_count,
        camera_height,
        fps,
        MOUNTAIN_HEIGHT_SCALE,
        if debug_enabled { "ON" } else { "OFF" },
    );
}

fn toggle_debug_render(keys: Res<ButtonInput<KeyCode>>, mut context: ResMut<DebugRenderContext>) {
    if keys.just_pressed(KeyCode::KeyX) {
        context.enabled = !context.enabled;
    }
}
