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
Second render benchmark — isolates world.step() cost and tests async rendering.

Measures world.step(render=False) alone to identify the physics bottleneck,
and tests with/without asyncRendering to find what helps most.

Usage:
    ./python.sh bench_render2.py [--async | --no-async]
"""

import argparse
import statistics
import sys
import time

# ---------------------------------------------------------------------------
# Parse before SimulationApp
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--async-render', action='store_true', default=True,
                    help='Enable asyncRendering (default: True)')
parser.add_argument('--no-async-render', dest='async_render', action='store_false')
args, _unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + _unknown

# ---------------------------------------------------------------------------
# SimulationApp
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp  # noqa: E402

_extra = []
if args.async_render:
    _extra = [
        '--/app/asyncRendering=true',
        '--/app/omni.usd/asyncHandshake=true',
    ]
    print('asyncRendering=ON')
else:
    print('asyncRendering=OFF')

simulation_app = SimulationApp({
    'headless': False,
    'physics_device': 'cuda',
    'device': 'cuda',
    'scene_graph_instancing': True,
    'extra_args': _extra,
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
# Scene
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

# Best settings from bench_render.py (config D)
_s.set('/rtx/rendermode', 'RaytracedLighting')
_s.set_int('/rtx/post/aa/op', 0)
_s.set_int('/rtx/pathtracing/spp', 1)
_s.set_bool('/rtx/shadows/enabled', False)
_s.set_bool('/rtx/ambientOcclusion/enabled', False)
_s.set_bool('/rtx/reflections/enabled', False)
_s.set_bool('/rtx/indirectDiffuse/enabled', False)
_s.set_bool('/rtx/post/dlss/enabled', False)
_s.set_bool('/rtx/post/taa/enabled', False)
_s.set_bool('/rtx/post/fog/enabled', False)
_s.set_bool('/rtx/post/histogram/enabled', False)
_s.set_bool('/rtx/post/lensFlares/enabled', False)
_s.set_bool('/rtx/post/motionblur/enabled', False)
_s.set_bool('/rtx/post/chromaticAberration/enabled', False)
_s.set_bool('/rtx/post/vignetteMask/enabled', False)

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------
PHYSICS_HZ = 500
RENDER_EVERY_N = 10
WARMUP_STEPS = 500
MEASURE_STEPS = 5000


def run_phase(label, render_every_n):
    """Run measurement phase. Returns timing stats dict."""
    # Warmup
    rn = 0
    for _ in range(WARMUP_STEPS):
        if not simulation_app.is_running():
            return None
        world.step(render=False)
        rn += 1
        if rn >= render_every_n:
            world.render()
            rn = 0

    loop_times = []
    render_times = []
    step_times = []   # world.step(render=False) ONLY
    xfer_times = []

    rn = 0
    for _ in range(MEASURE_STEPS):
        if not simulation_app.is_running():
            return None

        t0 = time.monotonic()

        # Read GPU state (mirrors vehicle.py)
        poses_t, orients_t = frame_view.get_world_poses(usd=False)
        vels_t = frame_view.get_velocities()
        t_xfer0 = time.monotonic()
        batch_data = torch.cat([poses_t, orients_t, vels_t], dim=1)
        batch_data.cpu().numpy()
        t_xfer1 = time.monotonic()
        xfer_times.append((t_xfer1 - t_xfer0) * 1e3)

        # Physics step (isolated)
        ts0 = time.monotonic()
        world.step(render=False)
        ts1 = time.monotonic()
        step_times.append((ts1 - ts0) * 1e3)

        # Render (every N steps)
        rn += 1
        if rn >= render_every_n:
            tr0 = time.monotonic()
            world.render()
            tr1 = time.monotonic()
            render_times.append((tr1 - tr0) * 1e3)
            rn = 0

        t1 = time.monotonic()
        loop_times.append((t1 - t0) * 1e3)

    def stats(data):
        if not data:
            return {'avg': 0, 'p95': 0, 'max': 0, 'min': 0}
        s = sorted(data)
        return {
            'avg': statistics.mean(data),
            'p95': s[int(len(s) * 0.95)],
            'max': max(data),
            'min': min(data),
        }

    return {
        'label': label,
        'loop': stats(loop_times),
        'step': stats(step_times),
        'render': stats(render_times),
        'xfer': stats(xfer_times),
    }


# ---------------------------------------------------------------------------
# Phase 1: Render every 10 steps (50 Hz) — current approach
# ---------------------------------------------------------------------------
phases = []

print('\n' + '=' * 70)
print('BENCHMARK 2 — Isolating physics step vs render cost')
async_label = 'async=ON' if args.async_render else 'async=OFF'
print(f'Mode: {async_label}')

print('\n[Phase 1] Render every 10 steps (50 Hz) ...')
r1 = run_phase('render@50Hz', render_every_n=10)
if r1:
    phases.append(r1)
    print(f'  Loop  : avg={r1["loop"]["avg"]:.3f}ms  p95={r1["loop"]["p95"]:.3f}ms')
    print(f'  Step  : avg={r1["step"]["avg"]:.3f}ms  p95={r1["step"]["p95"]:.3f}ms  '
          f'(world.step only)')
    print(f'  Render: avg={r1["render"]["avg"]:.3f}ms  p95={r1["render"]["p95"]:.3f}ms')
    print(f'  Xfer  : avg={r1["xfer"]["avg"]:.4f}ms')

# ---------------------------------------------------------------------------
# Phase 2: Render every 20 steps (25 Hz) — test render amortization
# ---------------------------------------------------------------------------
if simulation_app.is_running():
    print('\n[Phase 2] Render every 20 steps (25 Hz) ...')
    r2 = run_phase('render@25Hz', render_every_n=20)
    if r2:
        phases.append(r2)
        print(f'  Loop  : avg={r2["loop"]["avg"]:.3f}ms  p95={r2["loop"]["p95"]:.3f}ms')
        print(f'  Step  : avg={r2["step"]["avg"]:.3f}ms  p95={r2["step"]["p95"]:.3f}ms')
        print(f'  Render: avg={r2["render"]["avg"]:.3f}ms  p95={r2["render"]["p95"]:.3f}ms')

# ---------------------------------------------------------------------------
# Phase 3: No render at all (physics only baseline)
# ---------------------------------------------------------------------------
if simulation_app.is_running():
    print('\n[Phase 3] No render (physics only baseline) ...')

    # Warmup
    for _ in range(WARMUP_STEPS):
        if not simulation_app.is_running():
            break
        world.step(render=False)
        world.render()   # still need periodic render to keep the app alive

    step_only = []
    loop_only = []
    xfer_only = []
    rn = 0
    for _ in range(MEASURE_STEPS):
        if not simulation_app.is_running():
            break
        t0 = time.monotonic()
        poses_t, orients_t = frame_view.get_world_poses(usd=False)
        vels_t = frame_view.get_velocities()
        batch_data = torch.cat([poses_t, orients_t, vels_t], dim=1)
        tx0 = time.monotonic()
        batch_data.cpu().numpy()
        tx1 = time.monotonic()
        xfer_only.append((tx1 - tx0) * 1e3)

        ts0 = time.monotonic()
        world.step(render=False)
        ts1 = time.monotonic()
        step_only.append((ts1 - ts0) * 1e3)

        # Render every 100 steps just to keep app alive (not measured)
        rn += 1
        if rn >= 100:
            world.render()
            rn = 0

        t1 = time.monotonic()
        loop_only.append((t1 - t0) * 1e3)

    if step_only:
        s_sorted = sorted(step_only)
        l_sorted = sorted(loop_only)
        x_sorted = sorted(xfer_only)
        phases.append({
            'label': 'physics_only (render@5Hz)',
            'loop': {'avg': statistics.mean(loop_only),
                     'p95': l_sorted[int(len(l_sorted) * 0.95)]},
            'step': {'avg': statistics.mean(step_only),
                     'p95': s_sorted[int(len(s_sorted) * 0.95)]},
            'render': {'avg': 0, 'p95': 0},
            'xfer': {'avg': statistics.mean(xfer_only),
                     'p95': x_sorted[int(len(x_sorted) * 0.95)]},
        })
        print(f'  Loop  : avg={statistics.mean(loop_only):.3f}ms  '
              f'p95={l_sorted[int(len(l_sorted)*0.95)]:.3f}ms')
        print(f'  Step  : avg={statistics.mean(step_only):.3f}ms  '
              f'p95={s_sorted[int(len(s_sorted)*0.95)]:.3f}ms')
        print(f'  Xfer  : avg={statistics.mean(xfer_only):.4f}ms')

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print(f'SUMMARY ({async_label})')
print(f'{"Phase":<35} {"Loop avg":>9} {"Step avg":>9} {"Render avg":>11}')
print('-' * 70)
for p in phases:
    print(
        f'{p["label"]:<35} '
        f'{p["loop"]["avg"]:>8.3f}ms '
        f'{p["step"]["avg"]:>8.3f}ms '
        f'{p["render"]["avg"]:>10.3f}ms'
    )
print('=' * 70)

# Write results
report_path = (
    '/home/robotsix-docker/LS2N/ros2_ws/'
    '.claude_memory/bench_render2_results.txt'
)
with open(report_path, 'a') as f:
    f.write(f'\n=== {async_label} ===\n')
    for p in phases:
        f.write(f'{p["label"]}: loop_avg={p["loop"]["avg"]:.3f}ms  '
                f'step_avg={p["step"]["avg"]:.3f}ms  '
                f'render_avg={p["render"]["avg"]:.3f}ms  '
                f'xfer_avg={p["xfer"]["avg"]:.4f}ms\n')

print(f'\nResults appended to: {report_path}')

simulation_app.close()
