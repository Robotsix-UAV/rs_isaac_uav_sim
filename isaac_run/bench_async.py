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
Async physics benchmark — tests physx simulation interface split and overlap.

Tests whether physx_si.simulate() returns before GPU work completes,
enabling CPU sensor work to run during physics computation.

Results written to .claude_memory/bench_async_results.txt
"""

import statistics
import time

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
UsdLux.DomeLight.Define(stage, '/World/DomeLight').CreateIntensityAttr(500.0)
prim_utils.create_prim(
    prim_path='/World/drone_0/basic_quadrotor', prim_type='Xform',
    position=(0.0, 0.0, 0.5), usd_path=DRONE_USD,
)
frame_view = RigidPrim(
    prim_paths_expr='/World/drone_0/basic_quadrotor/Geometry/base_link/frame',
    name='bench_frame', reset_xform_properties=False,
)
world.scene.add(frame_view)

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
_s.set_bool('/physics/updateToUSD', False)

# ---------------------------------------------------------------------------
WARMUP, MEASURE, RENDER_N = 500, 5000, 10
physics_dt = 1.0 / 500.0


def _pct(data, p):
    return sorted(data)[int(len(data) * p)]


results = {}
report_lines = ['ASYNC BENCHMARK RESULTS\n']

# ---------------------------------------------------------------------------
# Phase A — world.step() baseline (reference)
# ---------------------------------------------------------------------------
if simulation_app.is_running():
    for _ in range(WARMUP):
        world.step(render=False)
    rn = 0
    for _ in range(WARMUP):
        world.render()

    t_loop, t_step = [], []
    rn = 0
    for _ in range(MEASURE):
        if not simulation_app.is_running():
            break
        t0 = time.monotonic()
        world.step(render=False)
        t1 = time.monotonic()
        t_step.append((t1 - t0) * 1e3)
        rn += 1
        if rn >= RENDER_N:
            world.render()
            rn = 0
        t2 = time.monotonic()
        t_loop.append((t2 - t0) * 1e3)

    if t_step:
        results['A'] = {
            'loop': statistics.mean(t_loop),
            'step': statistics.mean(t_step),
        }
        report_lines.append(
            f'A world.step baseline: '
            f'loop={results["A"]["loop"]:.3f}ms  '
            f'step={results["A"]["step"]:.3f}ms\n'
        )

# ---------------------------------------------------------------------------
# Phase B — physx direct: measure simulate() vs fetch_results() split
# ---------------------------------------------------------------------------
if simulation_app.is_running():
    try:
        from omni.physx import get_physx_simulation_interface  # noqa: E402, I100
        physx_si = get_physx_simulation_interface()

        for _ in range(WARMUP):
            world.step(render=False)

        t_sim, t_fetch, t_loop = [], [], []
        rn = 0
        cur_t = WARMUP * physics_dt * 2
        for _ in range(MEASURE):
            if not simulation_app.is_running():
                break
            t0 = time.monotonic()
            physx_si.simulate(physics_dt, cur_t)
            t1 = time.monotonic()
            t_sim.append((t1 - t0) * 1e3)
            physx_si.fetch_results(cur_t, True)
            t2 = time.monotonic()
            t_fetch.append((t2 - t1) * 1e3)
            cur_t += physics_dt
            rn += 1
            if rn >= RENDER_N:
                world.render()
                rn = 0
            t3 = time.monotonic()
            t_loop.append((t3 - t0) * 1e3)

        if t_sim:
            sim_avg = statistics.mean(t_sim)
            fetch_avg = statistics.mean(t_fetch)
            loop_avg = statistics.mean(t_loop)
            results['B'] = {'sim': sim_avg, 'fetch': fetch_avg, 'loop': loop_avg}
            report_lines.append(
                f'B physx direct: '
                f'sim={sim_avg:.4f}ms  fetch={fetch_avg:.4f}ms  '
                f'loop={loop_avg:.3f}ms\n'
            )
            # Is simulate() truly async? If sim_avg << (sim_avg+fetch_avg),
            # then GPU work is deferred to fetch_results().
            async_ratio = sim_avg / (sim_avg + fetch_avg) * 100
            report_lines.append(
                f'  sim/(sim+fetch) ratio = {async_ratio:.1f}% '
                f'(low = more async, high = sim blocks)\n'
            )
    except Exception as exc:
        report_lines.append(f'B physx direct FAILED: {exc}\n')

# ---------------------------------------------------------------------------
# Phase C — overlap: simulate → CPU work → fetch
# ---------------------------------------------------------------------------
if simulation_app.is_running() and 'B' in results:
    try:
        from omni.physx import get_physx_simulation_interface  # noqa: F811, E402
        physx_si = get_physx_simulation_interface()

        for _ in range(WARMUP):
            world.step(render=False)

        CPU_WORK_US = 300  # µs of simulated sensor/MAVLink work

        t_sim, t_cpu, t_fetch, t_loop = [], [], [], []
        rn = 0
        cur_t = WARMUP * physics_dt * 3
        for _ in range(MEASURE):
            if not simulation_app.is_running():
                break
            t0 = time.monotonic()
            physx_si.simulate(physics_dt, cur_t)
            t1 = time.monotonic()
            t_sim.append((t1 - t0) * 1e3)
            # Simulated CPU sensor work (busy-wait)
            deadline = t1 + CPU_WORK_US * 1e-6
            while time.monotonic() < deadline:
                pass
            t2 = time.monotonic()
            t_cpu.append((t2 - t1) * 1e3)
            physx_si.fetch_results(cur_t, True)
            t3 = time.monotonic()
            t_fetch.append((t3 - t2) * 1e3)
            cur_t += physics_dt
            rn += 1
            if rn >= RENDER_N:
                world.render()
                rn = 0
            t4 = time.monotonic()
            t_loop.append((t4 - t0) * 1e3)

        if t_loop:
            loop_avg = statistics.mean(t_loop)
            fetch_avg = statistics.mean(t_fetch)
            results['C'] = {'loop': loop_avg, 'fetch': fetch_avg}
            saved = max(0.0, CPU_WORK_US / 1000.0 - fetch_avg)
            report_lines.append(
                f'C async overlap ({CPU_WORK_US}µs CPU): '
                f'loop={loop_avg:.3f}ms  fetch_remaining={fetch_avg:.4f}ms  '
                f'saved≈{saved:.3f}ms\n'
            )
    except Exception as exc:
        report_lines.append(f'C async overlap FAILED: {exc}\n')

# ---------------------------------------------------------------------------
report_path = (
    '/home/robotsix-docker/LS2N/ros2_ws/'
    '.claude_memory/bench_async_results.txt'
)
with open(report_path, 'w') as f:
    f.writelines(report_lines)

simulation_app.close()
