# physics-rust

A 3D interactive terrain exploration simulator with vehicle dynamics, built in Rust using the Bevy game engine and Rapier3D physics.

![Rust](https://img.shields.io/badge/Rust-2021-orange) ![Bevy](https://img.shields.io/badge/Bevy-0.13-blue) ![Rapier3D](https://img.shields.io/badge/Rapier3D-0.25-green)

## Overview

Drive a physically simulated vehicle across procedurally generated terrain. The simulation features a 4-wheel independent suspension system, realistic engine forces, a wheel damage model, and infinite chunk-based terrain that loads and unloads dynamically around the vehicle.

All geometry — the car body, wheels, and terrain — is generated procedurally at runtime. No external 3D models are required.

## Features

### Vehicle Physics
- 4-wheel independent suspension with spring-damper dynamics (spring: 22,000 N/m, damping: 3,200 N·s/m)
- Per-wheel lateral friction, longitudinal drive force, and steering
- Handbrake with momentum bleed-off
- Wheel damage system: impacts above 28 m/s or severe suspension compression cause damage; wheels detach and become physics debris when destroyed
- Vehicle mass: 1,200 kg; engine force: 29,000 N

### Procedural Terrain
- Fractal noise with 7 octaves for natural-looking heightmaps
- Cliff masking, ridge fields, mountain range placement, and peak variation layers
- Chunk-based streaming (40×40 units, 32×32 vertex resolution per chunk)
- Async mesh and collider generation via `AsyncComputeTaskPool` to avoid frame stutter
- Deterministic from a seed — reproducible with the hex seed shown in the HUD

### Controls

| Key / Input | Action |
|---|---|
| W / S | Throttle / brake |
| A / D | Steer left / right |
| Space | Handbrake |
| R | Reset scene with new random terrain |
| X | Toggle physics debug rendering |
| Mouse drag (left) | Orbit camera |
| Mouse scroll | Zoom in / out |

### Camera
- Orbit camera that follows the vehicle with smooth exponential lag (3.0 s⁻¹)
- Manual orbit override with left mouse drag
- Zoom range: 2.0 – 100.0 units
- Automatic clamping to prevent clipping below terrain

### HUD
- Speed (km/h), attached wheel count, damage warning
- Terrain seed, active chunk count, pending tasks, camera height
- Real-time FPS counter, mountain scale multiplier, physics debug status

## Requirements

- Rust 1.70+ (Edition 2021)
- Cargo
- A GPU capable of rendering 3D graphics (Bevy uses wgpu)

## Building & Running

```bash
# Release build (recommended — full performance)
cargo run --release

# Debug build (faster to compile, slower at runtime)
cargo run
```

## Project Structure

```
physics-rust/
├── Cargo.toml      # Dependencies: bevy 0.13, bevy_rapier3d 0.25
├── Cargo.lock
└── src/
    └── main.rs     # All simulation code (~2,000 lines)
```

## Implementation Highlights

- **ECS architecture** — fully component-based via Bevy; resources, components, and systems are cleanly separated
- **Suspension simulation** — raycast per wheel, spring/damper force accumulation, contact ratio tracking, and impact velocity for damage calculation
- **Terrain noise** — multi-scale sampling with coordinate rotation at each octave to break up grid artifacts; cliff generation uses a separate masking pass
- **Seeded determinism** — XOR-based hash over a 64-bit seed; seed displayed in HUD as hex for reproducibility
- **Async chunk loading** — terrain chunks are spawned as background tasks and merged into the world when complete, keeping the main thread free
- **Procedural geometry** — car body built from 6 cross-sectional profiles; each wheel has tire tread, rim, hub, 6 spokes, and 4 lug nuts, all generated in code

## Physics Constants

| Parameter | Value |
|---|---|
| Gravity | 9.81 m/s² |
| Terrain base amplitude | 15.0 units |
| Mountain height multiplier | 10× |
| Spring stiffness | 22,000 N/m |
| Damping | 3,200 N·s/m |
| Engine force | 29,000 N |
| Handbrake force | 42,000 N |
| Lateral friction | 6,500 N |
| Max steering angle | ±0.6 rad |
| Crash velocity threshold | 28 m/s |
