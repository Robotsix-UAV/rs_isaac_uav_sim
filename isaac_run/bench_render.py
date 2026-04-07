#!/usr/bin/env python3
# Copyright 2026 ROBOTSIX
# Author: Damien SIX (damien@robotsix.net)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Render performance benchmark — no PX4 / MAVLink required.

Runs the sim loop with different carb/render settings in sequence,
measuring physics loop time and world.render() time independently.

Usage:
    ./python.sh bench_render.py [--headless]
"""

import statistics
import time

# ---------------------------------------------------------------------------
# SimulationApp must be first Omniverse call
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({
    'headless': False,
    'physics_device': 'cuda',
    'device': 'cuda',
    'scene_graph_instancing': True,
    'extra_args': [
        '--/app/asyncRendering=true',
        '--/app/omni.usd/asyncHandshake=true',
    ],
})

import carb  # noqa: E402, I100
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pxr import UsdLux  # noqa: E402, I100

from isaacsim.core.api.objects import GroundPlane  # noqa: E402, I100
from isaacsim.core.api.world import World  # noqa: E402
from isaacsim.core.prims import RigidPrim  # noqa: E402
import isaacsim.core.utils.prims as prim_utils  # noqa: E402

# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------
DRONE_USD = (
    '/home/robotsix-docker/LS2N/ros2_ws/src/rs_isaac_uav_sim/assets/'
    'basic_quadrotor/basic_quadrotor.usda'
)

world = World(
    stage_units_in_meters=1.0,
    backend='torch',
    device='cuda',
    physics_dt=1.0 / 500.0,
    rendering_dt=1.0 / 50.0,
)

GroundPlane(prim_path='/World/GroundPlane', z_position=0.0,
            color=np.array([0.3, 0.3, 0.3]))

stage = world.stage
dome = UsdLux.DomeLight.Define(stage, '/World/DomeLight')
dome.CreateIntensityAttr(500.0)

prim_utils.create_prim(
    prim_path='/World/drone_0/basic_quadrotor',
    prim_type='Xform',
    position=(0.0, 0.0, 0.5),
    usd_path=DRONE_USD,
)
frame_view = RigidPrim(
    prim_paths_expr='/World/drone_0/basic_quadrotor/Geometry/base_link/frame',
    name='bench_frame',
    reset_xform_properties=False,
)
world.scene.add(frame_view)
world.reset()

_s = carb.settings.get_settings()

# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------
CONFIGS = [
    {
        'name': 'A  RaytracedLighting (current baseline)',
        'settings': {
            '/rtx/rendermode': 'RaytracedLighting',
            '/rtx/post/aa/op': 0,
            '/rtx/pathtracing/spp': 1,
            '/rtx/shadows/enabled': False,
            '/rtx/ambientOcclusion/enabled': False,
            '/rtx/reflections/enabled': False,
            '/rtx/indirectDiffuse/enabled': False,
            '/rtx/post/dlss/enabled': False,
            '/rtx/post/taa/enabled': False,
            '/rtx/post/fog/enabled': False,
        },
    },
    {
        'name': 'B  + histogram / lensflare / motionblur off',
        'settings': {
            '/rtx/rendermode': 'RaytracedLighting',
            '/rtx/post/aa/op': 0,
            '/rtx/pathtracing/spp': 1,
            '/rtx/shadows/enabled': False,
            '/rtx/ambientOcclusion/enabled': False,
            '/rtx/reflections/enabled': False,
            '/rtx/indirectDiffuse/enabled': False,
            '/rtx/post/dlss/enabled': False,
            '/rtx/post/taa/enabled': False,
            '/rtx/post/fog/enabled': False,
            # Extra: disable post-processing passes
            '/rtx/post/histogram/enabled': False,
            '/rtx/post/lensFlares/enabled': False,
            '/rtx/post/motionblur/enabled': False,
            '/rtx/post/chromaticAberration/enabled': False,
            '/rtx/post/vignetteMask/enabled': False,
        },
    },
    {
        'name': 'C  PathTracing spp=1 (for comparison)',
        'settings': {
            '/rtx/rendermode': 'PathTracing',
            '/rtx/pathtracing/spp': 1,
            '/rtx/pathtracing/totalSpp': 1,
            '/rtx/post/aa/op': 0,
            '/rtx/shadows/enabled': False,
            '/rtx/ambientOcclusion/enabled': False,
            '/rtx/reflections/enabled': False,
            '/rtx/indirectDiffuse/enabled': False,
            '/rtx/post/dlss/enabled': False,
            '/rtx/post/taa/enabled': False,
            '/rtx/post/fog/enabled': False,
            '/rtx/post/histogram/enabled': False,
            '/rtx/post/lensFlares/enabled': False,
            '/rtx/post/motionblur/enabled': False,
        },
    },
    {
        'name': 'D  RaytracedLighting + GPU-CPU transfer reduction (batch)',
        'settings': {
            '/rtx/rendermode': 'RaytracedLighting',
            '/rtx/post/aa/op': 0,
            '/rtx/pathtracing/spp': 1,
            '/rtx/shadows/enabled': False,
            '/rtx/ambientOcclusion/enabled': False,
            '/rtx/reflections/enabled': False,
            '/rtx/indirectDiffuse/enabled': False,
            '/rtx/post/dlss/enabled': False,
            '/rtx/post/taa/enabled': False,
            '/rtx/post/fog/enabled': False,
            '/rtx/post/histogram/enabled': False,
            '/rtx/post/lensFlares/enabled': False,
            '/rtx/post/motionblur/enabled': False,
            '/rtx/post/chromaticAberration/enabled': False,
            '/rtx/post/vignetteMask/enabled': False,
        },
        'batch_gpu_transfer': True,  # Special flag: single batched CPU transfer
    },
]

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
PHYSICS_HZ = 500
RENDER_EVERY_N = 10          # 50 Hz render
WARMUP_STEPS = 500           # 1 s warmup
MEASURE_STEPS = 5000         # 10 s measurement


def apply_settings(cfg):
    for key, val in cfg['settings'].items():
        if isinstance(val, bool):
            _s.set_bool(key, val)
        elif isinstance(val, int):
            _s.set_int(key, val)
        elif isinstance(val, float):
            _s.set_float(key, val)
        else:
            _s.set(key, val)


def run_benchmark(cfg):
    apply_settings(cfg)
    do_batch = cfg.get('batch_gpu_transfer', False)

    # Warmup
    render_step = 0
    for _ in range(WARMUP_STEPS):
        if not simulation_app.is_running():
            return None
        world.step(render=False)
        render_step += 1
        if render_step >= RENDER_EVERY_N:
            world.render()
            render_step = 0

    # Measurement
    loop_times = []
    render_times = []
    transfer_times = []
    render_step = 0

    for _ in range(MEASURE_STEPS):
        if not simulation_app.is_running():
            return None

        t0 = time.monotonic()

        # --- Simulate GPU-CPU transfer cost (like vehicle.py step) ---
        if do_batch:
            # Batch: single cat + single transfer
            poses_t, orients_t = frame_view.get_world_poses(usd=False)
            vels_t = frame_view.get_velocities()
            batch_t = torch.cat([poses_t, orients_t, vels_t], dim=1)
            t_xfer0 = time.monotonic()
            batch_t.cpu().numpy()  # one sync
            t_xfer1 = time.monotonic()
        else:
            # Baseline: 3 separate transfers (as in vehicle.py)
            poses_t, orients_t = frame_view.get_world_poses(usd=False)
            vels_t = frame_view.get_velocities()
            t_xfer0 = time.monotonic()
            poses_t.cpu().numpy()
            orients_t.cpu().numpy()
            vels_t.cpu().numpy()
            t_xfer1 = time.monotonic()

        transfer_times.append((t_xfer1 - t_xfer0) * 1e3)

        world.step(render=False)

        render_step += 1
        t_render = 0.0
        if render_step >= RENDER_EVERY_N:
            tr0 = time.monotonic()
            world.render()
            tr1 = time.monotonic()
            t_render = (tr1 - tr0) * 1e3
            render_times.append(t_render)
            render_step = 0

        t1 = time.monotonic()
        loop_times.append((t1 - t0) * 1e3)

    return {
        'loop_avg': statistics.mean(loop_times),
        'loop_p95': sorted(loop_times)[int(len(loop_times) * 0.95)],
        'loop_max': max(loop_times),
        'render_avg': statistics.mean(render_times) if render_times else 0,
        'render_p95': sorted(render_times)[int(len(render_times) * 0.95)] if render_times else 0,
        'render_max': max(render_times) if render_times else 0,
        'xfer_avg': statistics.mean(transfer_times),
        'xfer_p95': sorted(transfer_times)[int(len(transfer_times) * 0.95)],
        'render_count': len(render_times),
    }


# ---------------------------------------------------------------------------
# Run all configs and collect results
# ---------------------------------------------------------------------------
results = []
print('\n' + '=' * 70)
print('RENDER BENCHMARK — Isaac Sim 6.0')
print(f'Physics: {PHYSICS_HZ} Hz | Render divider: every {RENDER_EVERY_N} steps '
      f'({PHYSICS_HZ // RENDER_EVERY_N} Hz)')
print(f'Warmup: {WARMUP_STEPS} steps | Measure: {MEASURE_STEPS} steps')
print('=' * 70)

for i, cfg in enumerate(CONFIGS):
    if not simulation_app.is_running():
        break
    print(f'\n[{i + 1}/{len(CONFIGS)}] Running config: {cfg["name"]} ...')
    res = run_benchmark(cfg)
    if res is None:
        print('  Aborted (window closed).')
        break
    results.append((cfg['name'], res))
    print(
        f'  Loop  : avg={res["loop_avg"]:.3f}ms  '
        f'p95={res["loop_p95"]:.3f}ms  max={res["loop_max"]:.3f}ms'
    )
    print(
        f'  Render: avg={res["render_avg"]:.3f}ms  '
        f'p95={res["render_p95"]:.3f}ms  max={res["render_max"]:.3f}ms  '
        f'({res["render_count"]} frames)'
    )
    print(
        f'  XferGPU-CPU: avg={res["xfer_avg"]:.4f}ms  '
        f'p95={res["xfer_p95"]:.4f}ms'
    )

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
if results:
    print('\n' + '=' * 70)
    print('SUMMARY')
    print(f'{"Config":<50} {"Loop avg":>9} {"Render avg":>11} {"Xfer avg":>9}')
    print('-' * 70)
    baseline_loop = results[0][1]['loop_avg']
    for name, res in results:
        gain = (1 - res['loop_avg'] / baseline_loop) * 100
        sign = '-' if gain > 0 else '+'
        print(
            f'{name:<50} '
            f'{res["loop_avg"]:>8.3f}ms '
            f'{res["render_avg"]:>10.3f}ms '
            f'{res["xfer_avg"]:>8.4f}ms '
            f'  ({sign}{abs(gain):.1f}%)'
        )
    print('=' * 70)

    # Write results to file for persistent record
    report_path = (
        '/home/robotsix-docker/LS2N/ros2_ws/'
        '.claude_memory/bench_render_results.txt'
    )
    with open(report_path, 'w') as f:
        f.write('RENDER BENCHMARK RESULTS\n')
        f.write(
            f'Physics: {PHYSICS_HZ} Hz  Render: {PHYSICS_HZ // RENDER_EVERY_N} Hz\n\n'
        )
        for name, res in results:
            f.write(f'{name}\n')
            f.write(
                f'  loop_avg={res["loop_avg"]:.3f}ms  '
                f'loop_p95={res["loop_p95"]:.3f}ms\n'
            )
            f.write(
                f'  render_avg={res["render_avg"]:.3f}ms  '
                f'render_p95={res["render_p95"]:.3f}ms\n'
            )
            f.write(f'  xfer_avg={res["xfer_avg"]:.4f}ms\n\n')
    print(f'\nResults written to: {report_path}')

simulation_app.close()
