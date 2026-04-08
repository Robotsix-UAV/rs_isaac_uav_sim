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

import argparse
import sys
import time

# ---------------------------------------------------------------------------
# Parse arguments BEFORE SimulationApp initialisation
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='MAVLink multi-drone simulator (Isaac Sim 6.0)')
parser.add_argument('--num_drones', type=int, default=1,
                    help='Number of drones (default: 1)')
parser.add_argument('--headless', action='store_true', help='Run without GUI')
parser.add_argument('--config', type=str, default='',
                    help='Path to drone YAML configuration file (optional)')
parser.add_argument('--base_port', type=int, default=4560,
                    help='Base MAVLink TCP port; drone i uses base_port+i+1 (default: 4560)')
parser.add_argument('--connection_ip', type=str, default='localhost',
                    help='IP address of the PX4 SITL instance (default: localhost)')
parser.add_argument('--verbose', action='store_true',
                    help='Enable verbose MAVLink warnings')
args, _unknown = parser.parse_known_args()
# Remove our custom flags from sys.argv so SimulationApp does not misinterpret them
sys.argv = [sys.argv[0]] + _unknown

# ---------------------------------------------------------------------------
# 1. Initialise SimulationApp — MUST be first Omniverse call
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp  # noqa: E402
import carb  # noqa: E402

_extra_args = (
    [] if args.headless else [
        '--/app/asyncRendering=true',
        '--/app/omni.usd/asyncHandshake=true',
    ]
)

simulation_app = SimulationApp({
    'headless': args.headless,
    'physics_device': 'cuda',
    'device': 'cuda',
    'scene_graph_instancing': True,
    'extra_args': _extra_args,
})
_s = carb.settings.get_settings()

# Lower render quality to improve frame-rate performance
_s.set('/rtx/rendermode', 'RaytracedLighting')
_s.set('/rtx/post/aa/op', 0)                        # disable TAA/DLSS (0 = off)
_s.set('/rtx/pathtracing/spp', 1)                   # samples-per-pixel
_s.set('/rtx/shadows/enabled', False)               # disable real-time shadows
_s.set('/rtx/ambientOcclusion/enabled', False)      # disable ambient occlusion
_s.set('/rtx/reflections/enabled', False)           # disable reflections
_s.set('/rtx/indirectDiffuse/enabled', False)       # disable indirect diffuse GI
_s.set('/rtx/post/dlss/enabled', False)             # disable DLSS upscaling
_s.set('/rtx/post/taa/enabled', False)              # disable TAA
_s.set('/rtx/post/fog/enabled', False)              # disable atmospheric fog
# Additional post-processing passes — measured -0.75 ms per render call (bench1 B vs A)
_s.set_bool('/rtx/post/histogram/enabled', False)       # disable auto-exposure histogram
_s.set_bool('/rtx/post/lensFlares/enabled', False)      # disable lens flares
_s.set_bool('/rtx/post/motionblur/enabled', False)      # disable motion blur
_s.set_bool('/rtx/post/chromaticAberration/enabled', False)  # disable chromatic aberration
_s.set_bool('/rtx/post/vignetteMask/enabled', False)    # disable vignette
# Skip writing physics results back to USD layer — renderer uses Fabric directly.
# Measured: saves ~0.34 ms on loop average (bench4 A vs C), allows GPU pipeline
# to run more freely. Safe because we read state via get_world_poses(usd=False).
_s.set_bool('/physics/updateToUSD', False)
print('[RENDER] Low-quality render mode applied (RaytracedLighting, all post-processing off).')

if not args.headless:
    print(f'[RENDER] asyncRendering={_s.get("/app/asyncRendering")}  '
          f'asyncHandshake={_s.get("/app/omni.usd/asyncHandshake")}')

# ---------------------------------------------------------------------------
# 2. Imports (all Omniverse/USD imports AFTER SimulationApp)
# ---------------------------------------------------------------------------
from isaacsim.core.api.objects import GroundPlane  # noqa: E402
from isaacsim.core.api.world import World  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux  # noqa: E402

from rs_isaac_uav_sim.sim.config import (  # noqa: E402
    GPSOrigin,
    load_config_from_yaml,
    MAVLINK_UPDATE_RATE,
    MavlinkParams,
    QuadrotorParams,
    SensorParams,
)
from rs_isaac_uav_sim.sim.vehicle import DroneSimManager  # noqa: E402


# ---------------------------------------------------------------------------
# 3. Physics scene configuration (GPU pipeline)
# ---------------------------------------------------------------------------
def setup_physics_scene(world):
    physics_context = world.get_physics_context()
    physics_context.enable_gpu_dynamics(True)
    physics_context.set_broadphase_type('GPU')

    scene_prim = world.stage.GetPrimAtPath(physics_context.prim_path)
    scene_prim.CreateAttribute(
        'physxScene:enableDirectGpuAccess', Sdf.ValueTypeNames.Bool
    ).Set(True)
    scene_prim.CreateAttribute(
        'physxScene:enableGpuArticulations', Sdf.ValueTypeNames.Bool
    ).Set(True)
    print('[PHYSICS] GPU scene configured.')


# ---------------------------------------------------------------------------
# 4. Environment (ground + lighting)
# ---------------------------------------------------------------------------
def setup_environment(world):
    GroundPlane(prim_path='/World/GroundPlane', z_position=0.0)

    stage = world.stage
    dome = UsdLux.DomeLight.Define(stage, '/World/DomeLight')
    dome.CreateIntensityAttr(500.0)

    distant = UsdLux.DistantLight.Define(stage, '/World/DistantLight')
    distant.CreateIntensityAttr(1000.0)
    distant.CreateAngleAttr(0.53)
    xformable = UsdGeom.Xformable(stage.GetPrimAtPath('/World/DistantLight'))
    xformable.ClearXformOpOrder()
    xformable.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 0.0))

    print('[ENV] Ground plane and lighting added.')


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    # Load drone config from YAML if provided, otherwise use dataclass defaults
    if args.config:
        cfg = load_config_from_yaml(args.config)
        quad_params = cfg['quadrotor']
        sensor_params = cfg['sensor']
        mavlink_params = cfg['mavlink']
        gps_origin = cfg['gps_origin']
    else:
        quad_params = QuadrotorParams()
        sensor_params = SensorParams()
        mavlink_params = MavlinkParams()
        gps_origin = GPSOrigin()

    # Override connection params from CLI and derive num_rotors from config
    mavlink_params.base_port = args.base_port
    mavlink_params.connection_ip = args.connection_ip
    mavlink_params.num_rotors = len(quad_params.rotor_positions)

    physics_dt = 1.0 / MAVLINK_UPDATE_RATE  # 0.002 s @ 500 Hz

    print(f'\n{"="*70}')
    print('ISAAC SIM 6.0  —  MAVLink Multi-Drone Simulator')
    print(f'{"="*70}')
    print(f'  Drones     : {args.num_drones}')
    print(f'  Headless   : {args.headless}')
    print(f'  Config     : {args.config if args.config else "(defaults)"}')
    print(f'  Physics dt : {physics_dt*1000:.1f} ms  '
          f'({MAVLINK_UPDATE_RATE:.0f} Hz)')
    print(
        f'  MAVLink    : TCP ports '
        f'{mavlink_params.base_port + 1} – '
        f'{mavlink_params.base_port + args.num_drones}'
    )
    print(f'{"="*70}\n')

    world = World(
        stage_units_in_meters=1.0,
        backend='torch',
        device='cuda',
        physics_dt=physics_dt,
        rendering_dt=1.0 / 50.0,
    )

    setup_physics_scene(world)
    setup_environment(world)

    # Build namespace list
    namespaces = [f'drone_{i}' for i in range(args.num_drones)]

    # Instantiate manager with explicit config objects
    manager = DroneSimManager(
        namespaces=namespaces,
        quad_params=quad_params,
        sensor_params=sensor_params,
        mavlink_params=mavlink_params,
        gps_origin=gps_origin,
        verbose=args.verbose,
    )

    # Spawn USD drone assets into the stage (computes default grid positions)
    manager.spawn_drones(world)

    # Reset world — initialises physics context and USD stage
    world.reset()

    # Initialise GPU tensor views and connect MAVLink
    manager.initialize()

    print('\n[READY] Simulation running. Start PX4 SITL instances now.')
    print(
        f'[INFO]  Connect PX4 to TCP ports '
        f'{mavlink_params.base_port + 1} – '
        f'{mavlink_params.base_port + args.num_drones}\n'
    )

    # Render divider: world.render() is expensive — limit to 50 Hz even though
    # physics runs at 500 Hz. This reduces GPU stall cost 10×.
    _render_every_n = max(1, round(MAVLINK_UPDATE_RATE / 50.0))
    _render_step = 0

    while simulation_app.is_running():
        t_loop_begin = time.monotonic()
        manager.step(physics_dt)
        world.step(render=False)
        if not args.headless:
            _render_step += 1
            if _render_step >= _render_every_n:
                t_render_begin = time.monotonic()
                world.render()
                t_render_end = time.monotonic()
                manager.record_render_timing(t_render_begin, t_render_end)
                _render_step = 0
        t_loop_end = time.monotonic()
        manager.record_loop_timing(t_loop_begin, t_loop_end)

    print('\n[SHUTDOWN] Closing simulation...')
    manager.close()
    simulation_app.close()


if __name__ == '__main__':
    main()
