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
Benchmark 5 — Profile time breakdown and test async physics overlap.

Goal: understand what's spending the 2.58ms inside world.step() and test
whether we can overlap physics GPU computation with CPU sensor work.

Phases:
  A: Time breakdown: get_world_poses() vs world.step() vs force application
  B: Direct physx interface simulate() + fetch_results() pattern
  C: Async overlap: submit physics, do CPU work (simulated), then fetch
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
dome_light = UsdLux.DomeLight.Define(stage, '/World/DomeLight')
dome_light.CreateIntensityAttr(500.0)

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

# Best render settings (Config D from bench1)
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

# Disable USD write-back to free GPU pipeline for rendering
_s.set_bool('/physics/updateToUSD', False)

WARMUP = 500
MEASURE = 5000
RENDER_N = 10

results = {}

# ---------------------------------------------------------------------------
# Phase A: Detailed time breakdown per world.step() sub-operation
# ---------------------------------------------------------------------------
if simulation_app.is_running():
    print('\n[A] Detailed sub-operation timing ...')
    # Warmup
    for _ in range(WARMUP):
        if not simulation_app.is_running():
            break
        world.step(render=False)
    rn = 0
    for _ in range(WARMUP):
        world.render()

    t_get_poses, t_get_vels, t_cpu_xfer, t_step, t_render_calls = [], [], [], [], []
    rn = 0
    for _ in range(MEASURE):
        if not simulation_app.is_running():
            break

        # (1) get_world_poses
        tp0 = time.monotonic()
        poses_t, orients_t = frame_view.get_world_poses(usd=False)
        tp1 = time.monotonic()
        t_get_poses.append((tp1 - tp0) * 1e3)

        # (2) get_velocities
        tv0 = time.monotonic()
        vels_t = frame_view.get_velocities()
        tv1 = time.monotonic()
        t_get_vels.append((tv1 - tv0) * 1e3)

        # (3) CPU transfer (batched)
        tx0 = time.monotonic()
        batch = torch.cat([poses_t, orients_t, vels_t], dim=1)
        _np = batch.cpu().numpy()
        tx1 = time.monotonic()
        t_cpu_xfer.append((tx1 - tx0) * 1e3)

        # (4) world.step
        ts0 = time.monotonic()
        world.step(render=False)
        ts1 = time.monotonic()
        t_step.append((ts1 - ts0) * 1e3)

        rn += 1
        if rn >= RENDER_N:
            tr0 = time.monotonic()
            world.render()
            tr1 = time.monotonic()
            t_render_calls.append((tr1 - tr0) * 1e3)
            rn = 0

    def s(d):
        if not d:
            return (0, 0, 0)
        sd = sorted(d)
        return (statistics.mean(d), sd[int(len(sd) * 0.95)], max(d))

    poses_s = s(t_get_poses)
    vels_s = s(t_get_vels)
    xfer_s = s(t_cpu_xfer)
    step_s = s(t_step)
    render_s = s(t_render_calls)

    total_avg = poses_s[0] + vels_s[0] + xfer_s[0] + step_s[0]
    print(f'  get_world_poses : avg={poses_s[0]:.4f}ms  p95={poses_s[1]:.4f}ms')
    print(f'  get_velocities  : avg={vels_s[0]:.4f}ms  p95={vels_s[1]:.4f}ms')
    print(f'  .cpu().numpy()  : avg={xfer_s[0]:.4f}ms  p95={xfer_s[1]:.4f}ms')
    print(f'  world.step()    : avg={step_s[0]:.4f}ms  p95={step_s[1]:.4f}ms')
    print(f'  world.render()  : avg={render_s[0]:.3f}ms  p95={render_s[1]:.3f}ms')
    print(f'  total (A+B+C+D) : avg={total_avg:.3f}ms')

    results['A_breakdown'] = {
        'get_poses': poses_s[0], 'get_vels': vels_s[0],
        'xfer': xfer_s[0], 'step': step_s[0], 'render': render_s[0],
        'total': total_avg,
    }

# ---------------------------------------------------------------------------
# Phase B: Direct physx interface — simulate() + fetch_results()
# ---------------------------------------------------------------------------
if simulation_app.is_running():
    print('\n[B] Direct physx interface: simulate() + fetch_results() ...')
    try:
        from omni.physx import get_physx_simulation_interface
        physx_si = get_physx_simulation_interface()
        physics_dt = 1.0 / 500.0

        for _ in range(WARMUP):
            if not simulation_app.is_running():
                break
            world.step(render=False)

        t_sim, t_fetch, t_total = [], [], []
        rn = 0
        current_time = 0.0
        for _ in range(MEASURE):
            if not simulation_app.is_running():
                break
            t0 = time.monotonic()

            ts = time.monotonic()
            physx_si.simulate(physics_dt, current_time)
            te = time.monotonic()
            t_sim.append((te - ts) * 1e3)

            tf0 = time.monotonic()
            physx_si.fetch_results(current_time, True)
            tf1 = time.monotonic()
            t_fetch.append((tf1 - tf0) * 1e3)

            current_time += physics_dt

            rn += 1
            if rn >= RENDER_N:
                world.render()
                rn = 0

            t1 = time.monotonic()
            t_total.append((t1 - t0) * 1e3)

        sim_s = s(t_sim)
        fetch_s = s(t_fetch)
        total_s = s(t_total)
        print(f'  simulate()      : avg={sim_s[0]:.4f}ms  p95={sim_s[1]:.4f}ms')
        print(f'  fetch_results() : avg={fetch_s[0]:.4f}ms  p95={fetch_s[1]:.4f}ms')
        print(f'  total loop      : avg={total_s[0]:.3f}ms')
        results['B_physx_direct'] = {
            'sim': sim_s[0], 'fetch': fetch_s[0], 'total': total_s[0]
        }
    except Exception as e:
        print(f'  physx direct interface failed: {e}')

# ---------------------------------------------------------------------------
# Phase C: Async overlap test — submit physics, do CPU work, then fetch
# The hypothesis: GPU physics takes ~2ms. If we submit async and do 0.5ms
# of CPU work before fetching, we "hide" 0.5ms of GPU wait time.
# ---------------------------------------------------------------------------
if simulation_app.is_running():
    print('\n[C] Async overlap: submit physics → CPU work → fetch ...')
    try:
        from omni.physx import get_physx_simulation_interface
        physx_si = get_physx_simulation_interface()
        physics_dt = 1.0 / 500.0

        for _ in range(WARMUP):
            if not simulation_app.is_running():
                break
            world.step(render=False)

        # Simulated CPU work duration (microseconds of busy-waiting)
        CPU_WORK_US = 300  # simulate ~0.3ms of sensor/MAVLink CPU work

        t_sim, t_cpu, t_fetch, t_loop = [], [], [], []
        rn = 0
        current_time = WARMUP * physics_dt
        for _ in range(MEASURE):
            if not simulation_app.is_running():
                break
            t0 = time.monotonic()

            # Submit GPU physics (async if possible)
            ts = time.monotonic()
            physx_si.simulate(physics_dt, current_time)
            te = time.monotonic()
            t_sim.append((te - ts) * 1e3)

            # Simulate overlapped CPU work (sensor computation, MAVLink, etc.)
            tc0 = time.monotonic()
            # Busy-wait for CPU_WORK_US microseconds (simulates CPU-side work)
            deadline = tc0 + CPU_WORK_US * 1e-6
            while time.monotonic() < deadline:
                pass
            tc1 = time.monotonic()
            t_cpu.append((tc1 - tc0) * 1e3)

            # Fetch physics results (waits only for remaining GPU time)
            tf0 = time.monotonic()
            physx_si.fetch_results(current_time, True)
            tf1 = time.monotonic()
            t_fetch.append((tf1 - tf0) * 1e3)

            current_time += physics_dt

            rn += 1
            if rn >= RENDER_N:
                world.render()
                rn = 0

            t1 = time.monotonic()
            t_loop.append((t1 - t0) * 1e3)

        sim_s = s(t_sim)
        cpu_s = s(t_cpu)
        fetch_s = s(t_fetch)
        loop_s = s(t_loop)
        print(f'  simulate()      : avg={sim_s[0]:.4f}ms  (async submission)')
        print(f'  CPU work        : avg={cpu_s[0]:.4f}ms  ({CPU_WORK_US}µs busy)')
        print(f'  fetch_results() : avg={fetch_s[0]:.4f}ms  (remaining GPU wait)')
        print(f'  total loop      : avg={loop_s[0]:.3f}ms')
        overlap_save = max(0, CPU_WORK_US / 1000.0 - fetch_s[0])
        print(f'  overlap saved   : ~{overlap_save:.3f}ms')
        results['C_async_overlap'] = {
            'sim': sim_s[0], 'cpu': cpu_s[0],
            'fetch': fetch_s[0], 'total': loop_s[0],
        }
    except Exception as e:
        print(f'  async overlap failed: {e}')

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('SUMMARY (realtime=100% at 2.000ms loop)')
if 'A_breakdown' in results:
    r = results['A_breakdown']
    rt = 2.0 / (r['total'] + r['render'] / 10) * 100
    print(f'A breakdown total   : {r["total"]:.3f}ms  RT%≈{rt:.1f}%')
    print(f'  get_poses={r["get_poses"]:.4f}ms  get_vels={r["get_vels"]:.4f}ms  '
          f'xfer={r["xfer"]:.4f}ms  step={r["step"]:.4f}ms')
if 'B_physx_direct' in results:
    r = results['B_physx_direct']
    rt = 2.0 / r['total'] * 100
    print(f'B physx direct total: {r["total"]:.3f}ms  RT%≈{rt:.1f}%')
    print(f'  simulate={r["sim"]:.4f}ms  fetch={r["fetch"]:.4f}ms')
if 'C_async_overlap' in results:
    r = results['C_async_overlap']
    rt = 2.0 / r['total'] * 100
    print(f'C async overlap     : {r["total"]:.3f}ms  RT%≈{rt:.1f}%')
    print(f'  simulate={r["sim"]:.4f}ms  cpu={r["cpu"]:.4f}ms  fetch={r["fetch"]:.4f}ms')
print('=' * 70)

report_path = (
    '/home/robotsix-docker/LS2N/ros2_ws/'
    '.claude_memory/bench_render5_results.txt'
)
with open(report_path, 'w') as f:
    f.write('BENCHMARK 5 — Time breakdown + async physics\n\n')
    if 'A_breakdown' in results:
        r = results['A_breakdown']
        f.write(f'A_breakdown: step={r["step"]:.4f}ms  get_poses={r["get_poses"]:.4f}ms  '
                f'get_vels={r["get_vels"]:.4f}ms  xfer={r["xfer"]:.4f}ms  '
                f'render={r["render"]:.3f}ms  total={r["total"]:.3f}ms\n')
    if 'B_physx_direct' in results:
        r = results['B_physx_direct']
        f.write(f'B_physx_direct: sim={r["sim"]:.4f}ms  fetch={r["fetch"]:.4f}ms  '
                f'total={r["total"]:.3f}ms\n')
    if 'C_async_overlap' in results:
        r = results['C_async_overlap']
        f.write(f'C_async_overlap: sim={r["sim"]:.4f}ms  cpu={r["cpu"]:.4f}ms  '
                f'fetch={r["fetch"]:.4f}ms  total={r["total"]:.3f}ms\n')

print(f'\nResults written to: {report_path}')
simulation_app.close()
