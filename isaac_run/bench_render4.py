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
Benchmark 4 — Test USD fabric sync bypass and SimulationContext.step() overhead.

The hypothesis: world.step() triggers a USD stage sync (writing physics results
back to USD layer) that is expensive. If we can skip or defer this and read
directly from GPU Fabric buffers (already possible via get_world_poses(usd=False)),
we save significant CPU time.

Configs tested:
  A) World.step(render=False) baseline
  B) SimulationContext.step(render=False) — skips World-level overhead
  C) World.step + skip USD write (/physics/updateFabricUsdTransforms = False)
  D) World.step + two combined settings to minimize USD sync
"""

import argparse
import statistics
import sys
import time

parser = argparse.ArgumentParser()
args, _unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + _unknown

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
from pxr import Sdf, UsdLux  # noqa: E402, I100

from isaacsim.core.api import SimulationContext  # noqa: E402, I100
from isaacsim.core.api.objects import GroundPlane  # noqa: E402
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

# Physics setup
physics_context = world.get_physics_context()
physics_context.enable_gpu_dynamics(True)
physics_context.set_broadphase_type('GPU')
scene_prim = stage.GetPrimAtPath(physics_context.prim_path)
scene_prim.CreateAttribute(
    'physxScene:enableDirectGpuAccess', Sdf.ValueTypeNames.Bool).Set(True)
scene_prim.CreateAttribute(
    'physxScene:enableGpuArticulations', Sdf.ValueTypeNames.Bool).Set(True)

world.reset()

_s = carb.settings.get_settings()
_sim = SimulationContext.instance()

# Best render settings
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
WARMUP = 500
MEASURE = 5000
RENDER_EVERY = 10


def _do_render_if_due(rn):
    if rn >= RENDER_EVERY:
        world.render()
        return 0
    return rn


def measure_step_fn(step_fn, label):
    """Measure a physics step function over MEASURE iterations."""
    rn = 0
    for _ in range(WARMUP):
        if not simulation_app.is_running():
            return None
        step_fn()
        rn += 1
        rn = _do_render_if_due(rn)

    step_ms = []
    loop_ms = []
    render_ms = []
    rn = 0
    for _ in range(MEASURE):
        if not simulation_app.is_running():
            return None
        t0 = time.monotonic()
        ts = time.monotonic()
        step_fn()
        te = time.monotonic()
        step_ms.append((te - ts) * 1e3)
        rn += 1
        if rn >= RENDER_EVERY:
            tr0 = time.monotonic()
            world.render()
            tr1 = time.monotonic()
            render_ms.append((tr1 - tr0) * 1e3)
            rn = 0
        t1 = time.monotonic()
        loop_ms.append((t1 - t0) * 1e3)

    def pct(d, p):
        return sorted(d)[int(len(d) * p)]

    return {
        'label': label,
        'loop_avg': statistics.mean(loop_ms),
        'loop_p95': pct(loop_ms, 0.95),
        'step_avg': statistics.mean(step_ms),
        'step_p95': pct(step_ms, 0.95),
        'render_avg': statistics.mean(render_ms) if render_ms else 0,
    }


# ---------------------------------------------------------------------------
# Config A: baseline World.step(render=False)
# ---------------------------------------------------------------------------
results = []

if simulation_app.is_running():
    print('\n[A] World.step(render=False) baseline ...')
    r = measure_step_fn(lambda: world.step(render=False), 'A:World.step')
    if r:
        results.append(r)
        print(f'  Loop : avg={r["loop_avg"]:.3f}ms  p95={r["loop_p95"]:.3f}ms')
        print(f'  Step : avg={r["step_avg"]:.3f}ms  p95={r["step_p95"]:.3f}ms')
        print(f'  Render: avg={r["render_avg"]:.3f}ms')

# ---------------------------------------------------------------------------
# Config B: SimulationContext.step(render=False) — avoids World Python overhead
# ---------------------------------------------------------------------------
if simulation_app.is_running():
    print('\n[B] SimulationContext.step(render=False) ...')
    r = measure_step_fn(lambda: _sim.step(render=False), 'B:SimCtx.step')
    if r:
        results.append(r)
        print(f'  Loop : avg={r["loop_avg"]:.3f}ms  p95={r["loop_p95"]:.3f}ms')
        print(f'  Step : avg={r["step_avg"]:.3f}ms  p95={r["step_p95"]:.3f}ms')
        print(f'  Render: avg={r["render_avg"]:.3f}ms')

# ---------------------------------------------------------------------------
# Config C: World.step + disable USD Fabric transform writes
# Hypothesis: USD write-back is expensive; direct GPU read doesn't need it.
# ---------------------------------------------------------------------------
if simulation_app.is_running():
    print('\n[C] World.step + disable USD transform write-back ...')
    # Try to disable USD write-back from physics
    try:
        _s.set_bool('/physics/updateFabricUsdTransforms', False)
        print('  Set /physics/updateFabricUsdTransforms = False')
    except Exception as e:
        print(f'  /physics/updateFabricUsdTransforms: {e}')
    try:
        _s.set_bool('/physics/updateToUSD', False)
        print('  Set /physics/updateToUSD = False')
    except Exception as e:
        print(f'  /physics/updateToUSD: {e}')

    r = measure_step_fn(lambda: world.step(render=False),
                        'C:World.step+noUSDsync')
    if r:
        results.append(r)
        print(f'  Loop : avg={r["loop_avg"]:.3f}ms  p95={r["loop_p95"]:.3f}ms')
        print(f'  Step : avg={r["step_avg"]:.3f}ms  p95={r["step_p95"]:.3f}ms')
        print(f'  Render: avg={r["render_avg"]:.3f}ms')

    # Restore
    try:
        _s.set_bool('/physics/updateFabricUsdTransforms', True)
        _s.set_bool('/physics/updateToUSD', True)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Config D: Combine SimCtx.step + disabled USD write-back
# ---------------------------------------------------------------------------
if simulation_app.is_running():
    print('\n[D] SimCtx.step + disable USD write-back ...')
    try:
        _s.set_bool('/physics/updateFabricUsdTransforms', False)
        _s.set_bool('/physics/updateToUSD', False)
    except Exception:
        pass

    r = measure_step_fn(lambda: _sim.step(render=False),
                        'D:SimCtx.step+noUSDsync')
    if r:
        results.append(r)
        print(f'  Loop : avg={r["loop_avg"]:.3f}ms  p95={r["loop_p95"]:.3f}ms')
        print(f'  Step : avg={r["step_avg"]:.3f}ms  p95={r["step_p95"]:.3f}ms')
        print(f'  Render: avg={r["render_avg"]:.3f}ms')

    try:
        _s.set_bool('/physics/updateFabricUsdTransforms', True)
        _s.set_bool('/physics/updateToUSD', True)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('SUMMARY (realtime target = 2.0ms per step)')
print(f'{"Config":<30} {"Loop avg":>9} {"Step avg":>9} {"RT%":>7} {"Render avg":>11}')
print('-' * 70)
for r in results:
    rt = 2.0 / r['loop_avg'] * 100
    print(f'{r["label"]:<30} {r["loop_avg"]:>8.3f}ms {r["step_avg"]:>8.3f}ms '
          f'{rt:>6.1f}%  {r["render_avg"]:>10.3f}ms')
print('=' * 70)

report_path = (
    '/home/robotsix-docker/LS2N/ros2_ws/'
    '.claude_memory/bench_render4_results.txt'
)
with open(report_path, 'w') as f:
    f.write('BENCHMARK 4 — USD sync bypass test\n\n')
    for r in results:
        rt = 2.0 / r['loop_avg'] * 100
        f.write(f'{r["label"]}: loop_avg={r["loop_avg"]:.3f}ms  '
                f'step_avg={r["step_avg"]:.3f}ms  '
                f'render_avg={r["render_avg"]:.3f}ms  '
                f'realtime={rt:.1f}%\n')

print(f'\nResults written to: {report_path}')

simulation_app.close()
