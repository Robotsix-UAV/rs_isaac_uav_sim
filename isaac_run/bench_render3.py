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
Third render benchmark — tests resolution impact and solver iterations.

Usage:
    ./python.sh bench_render3.py [--width W] [--height H] [--solver-iters N]
"""

import argparse
import statistics
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--width', type=int, default=640,
                    help='Viewport width (default: 640)')
parser.add_argument('--height', type=int, default=360,
                    help='Viewport height (default: 360)')
parser.add_argument('--solver-iters', type=int, default=4,
                    help='PhysX solver position iterations (default: 4)')
args, _unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + _unknown

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
        f'--/app/renderer/resolution/width={args.width}',
        f'--/app/renderer/resolution/height={args.height}',
        f'--/app/window/width={args.width + 160}',
        f'--/app/window/height={args.height + 140}',
    ],
})

import carb  # noqa: E402, I100
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pxr import Sdf, UsdLux  # noqa: E402, I100

from isaacsim.core.api.objects import GroundPlane  # noqa: E402, I100
from isaacsim.core.api.world import World  # noqa: E402
from isaacsim.core.prims import RigidPrim  # noqa: E402
import isaacsim.core.utils.prims as prim_utils  # noqa: E402

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

# Configure physics solver iterations (affects physics step cost)
physics_context = world.get_physics_context()
physics_context.enable_gpu_dynamics(True)
physics_context.set_broadphase_type('GPU')

scene_prim = stage.GetPrimAtPath(physics_context.prim_path)
scene_prim.CreateAttribute(
    'physxScene:enableDirectGpuAccess', Sdf.ValueTypeNames.Bool).Set(True)
scene_prim.CreateAttribute(
    'physxScene:enableGpuArticulations', Sdf.ValueTypeNames.Bool).Set(True)

# Set solver iteration count
try:
    physics_context.set_solver_type('TGS')  # Temporal Gauss-Seidel (faster for articulations)
    print('[BENCH3] Solver: TGS')
except Exception as e:
    print(f'[BENCH3] set_solver_type failed: {e}')

# Set position/velocity iterations
try:
    scene_prim.CreateAttribute(
        'physxScene:solvePositionIterationCount', Sdf.ValueTypeNames.UInt
    ).Set(args.solver_iters)
    scene_prim.CreateAttribute(
        'physxScene:solveVelocityIterationCount', Sdf.ValueTypeNames.UInt
    ).Set(1)
    print(f'[BENCH3] Solver iterations: pos={args.solver_iters} vel=1')
except Exception as e:
    print(f'[BENCH3] solver iterations warning: {e}')

world.reset()

_s = carb.settings.get_settings()

# Best render settings from bench_render (config D)
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

print(f'[BENCH3] Resolution: {args.width}x{args.height} | asyncRendering: ON')

# ---------------------------------------------------------------------------
PHYSICS_HZ = 500
WARMUP_STEPS = 500
MEASURE_STEPS = 5000
RENDER_EVERY_N = 10


def run_phase(label, render_every_n):
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
    step_times = []
    xfer_times = []
    rn = 0

    for _ in range(MEASURE_STEPS):
        if not simulation_app.is_running():
            return None
        t0 = time.monotonic()

        poses_t, orients_t = frame_view.get_world_poses(usd=False)
        vels_t = frame_view.get_velocities()
        batch_t = torch.cat([poses_t, orients_t, vels_t], dim=1)
        tx0 = time.monotonic()
        batch_t.cpu().numpy()
        tx1 = time.monotonic()
        xfer_times.append((tx1 - tx0) * 1e3)

        ts0 = time.monotonic()
        world.step(render=False)
        ts1 = time.monotonic()
        step_times.append((ts1 - ts0) * 1e3)

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
            return {'avg': 0, 'p95': 0}
        s = sorted(data)
        return {'avg': statistics.mean(data), 'p95': s[int(len(s) * 0.95)]}

    return {
        'label': label,
        'loop': stats(loop_times),
        'step': stats(step_times),
        'render': stats(render_times),
        'xfer': stats(xfer_times),
    }


# ---------------------------------------------------------------------------
phases = []

print('\n' + '=' * 70)
print(f'BENCHMARK 3 — Resolution {args.width}x{args.height} | solver_iters={args.solver_iters}')
print('=' * 70)

for label, every_n in [('render@50Hz', 10), ('render@25Hz', 20)]:
    if not simulation_app.is_running():
        break
    print(f'\n[Phase] {label} ...')
    r = run_phase(label, every_n)
    if r:
        phases.append(r)
        print(f'  Loop  : avg={r["loop"]["avg"]:.3f}ms  p95={r["loop"]["p95"]:.3f}ms')
        print(f'  Step  : avg={r["step"]["avg"]:.3f}ms  p95={r["step"]["p95"]:.3f}ms')
        print(f'  Render: avg={r["render"]["avg"]:.3f}ms  p95={r["render"]["p95"]:.3f}ms')
        print(f'  Xfer  : avg={r["xfer"]["avg"]:.4f}ms')

# Physics-only phase
if simulation_app.is_running():
    print('\n[Phase] physics_only (render@5Hz) ...')
    step_only, loop_only, xfer_only = [], [], []
    rn = 0
    # Warmup
    for _ in range(WARMUP_STEPS):
        if not simulation_app.is_running():
            break
        world.step(render=False)
        rn += 1
        if rn >= 100:
            world.render()
            rn = 0
    # Measure
    rn = 0
    for _ in range(MEASURE_STEPS):
        if not simulation_app.is_running():
            break
        t0 = time.monotonic()
        batch_t = torch.cat([frame_view.get_world_poses(usd=False)[0],
                             frame_view.get_world_poses(usd=False)[1],
                             frame_view.get_velocities()], dim=1)
        tx0 = time.monotonic()
        batch_t.cpu().numpy()
        tx1 = time.monotonic()
        xfer_only.append((tx1 - tx0) * 1e3)
        ts0 = time.monotonic()
        world.step(render=False)
        ts1 = time.monotonic()
        step_only.append((ts1 - ts0) * 1e3)
        rn += 1
        if rn >= 100:
            world.render()
            rn = 0
        t1 = time.monotonic()
        loop_only.append((t1 - t0) * 1e3)
    if step_only:
        s = sorted(step_only)
        loop_sorted = sorted(loop_only)
        x = sorted(xfer_only)
        p = {
            'label': 'physics_only',
            'loop': {'avg': statistics.mean(loop_only),
                     'p95': loop_sorted[int(len(loop_sorted) * 0.95)]},
            'step': {'avg': statistics.mean(step_only),
                     'p95': s[int(len(s) * 0.95)]},
            'render': {'avg': 0, 'p95': 0},
            'xfer': {'avg': statistics.mean(xfer_only),
                     'p95': x[int(len(x) * 0.95)]},
        }
        phases.append(p)
        print(f'  Loop  : avg={p["loop"]["avg"]:.3f}ms  p95={p["loop"]["p95"]:.3f}ms')
        print(f'  Step  : avg={p["step"]["avg"]:.3f}ms  p95={p["step"]["p95"]:.3f}ms')
        print(f'  Xfer  : avg={p["xfer"]["avg"]:.4f}ms')

# ---------------------------------------------------------------------------
report_path = (
    '/home/robotsix-docker/LS2N/ros2_ws/'
    '.claude_memory/bench_render3_results.txt'
)
tag = f'{args.width}x{args.height}_solv{args.solver_iters}'
with open(report_path, 'a') as f:
    f.write(f'\n=== {tag} ===\n')
    for p in phases:
        f.write(
            f'{p["label"]}: loop_avg={p["loop"]["avg"]:.3f}ms  '
            f'step_avg={p["step"]["avg"]:.3f}ms  '
            f'render_avg={p["render"]["avg"]:.3f}ms  '
            f'xfer_avg={p["xfer"]["avg"]:.4f}ms\n'
        )

print(f'\nResults written to: {report_path}')

simulation_app.close()
