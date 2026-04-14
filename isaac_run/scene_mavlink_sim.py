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
parser.add_argument('--ros2_namespaces', type=str, default='',
                    help=('Comma-separated list of ROS 2 namespaces, one '
                          'per drone (e.g. "drone_0,drone_1"). When '
                          'set, Isaac Sim enables the isaacsim.ros2.bridge '
                          'extension and publishes /clock plus '
                          '<ns>/isaac_odom (nav_msgs/Odometry) from an '
                          'OmniGraph, replacing the UDP ground-truth path. '
                          'Empty disables the ROS 2 bridge path.'))
args, _unknown = parser.parse_known_args()
# Remove our custom flags from sys.argv so SimulationApp does not misinterpret them
sys.argv = [sys.argv[0]] + _unknown

# ---------------------------------------------------------------------------
# 1. Initialise SimulationApp — MUST be first Omniverse call
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp  # noqa: E402

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

import carb  # noqa: E402, I100

_s = carb.settings.get_settings()

# Balanced render config: keep RaytracedLighting (fastest RTX path) and the
# cheap visual quality wins (shadows, TAA, AO, auto-exposure) which give
# the biggest "looks like a real scene" boost per millisecond. Skip the
# truly expensive features (indirect diffuse GI, real-time reflections,
# path tracing) and the post-processing junk that adds visual noise
# without telling you anything (motion blur, chromatic aberration,
# vignette, lens flares).
_s.set('/rtx/rendermode', 'RaytracedLighting')
_s.set('/rtx/pathtracing/spp', 1)                # PT samples (only matters in PT mode)
_s.set('/rtx/shadows/enabled', True)             # real-time shadows: biggest visual win
_s.set('/rtx/ambientOcclusion/enabled', True)    # AO: contact darkening
_s.set('/rtx/reflections/enabled', True)         # specular reflections on shiny surfaces
_s.set('/rtx/indirectDiffuse/enabled', True)     # one-bounce indirect diffuse GI
_s.set('/rtx/post/aa/op', 3)                     # TAA mode (1=FXAA, 2=SMAA, 3=TAA)
_s.set('/rtx/post/taa/enabled', True)            # TAA pass for edge anti-alias
_s.set('/rtx/post/dlss/enabled', False)          # DLSS off
_s.set('/rtx/post/fog/enabled', False)           # no atmospheric fog
# Auto-exposure: ON, otherwise the scene looks washed out / dark
_s.set_bool('/rtx/post/histogram/enabled', True)
# Junk post-processing: still off (no info value)
_s.set_bool('/rtx/post/lensFlares/enabled', False)
_s.set_bool('/rtx/post/motionblur/enabled', False)
_s.set_bool('/rtx/post/chromaticAberration/enabled', False)
_s.set_bool('/rtx/post/vignetteMask/enabled', False)
# Skip writing physics results back to USD layer — renderer uses Fabric directly.
# Measured: saves ~0.34 ms on loop average (bench4 A vs C), allows GPU pipeline
# to run more freely. Safe because we read state via get_world_poses(usd=False).
_s.set_bool('/physics/updateToUSD', False)
print('[RENDER] Balanced render mode applied '
      '(RaytracedLighting + shadows + AO + TAA + auto-exposure).')

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
# 4b. ROS 2 bridge OmniGraph
# ---------------------------------------------------------------------------
def _setup_ros2_bridge_graph(
    isaac_namespaces: list, ros2_namespaces: list
) -> str:
    """
    Build an OmniGraph that publishes /clock + per-drone Odometry.

    :param isaac_namespaces: Prim-side namespace per drone (e.g.
        ``drone_0``). Used to build the chassis prim path
        ``/World/{ns}/basic_quadrotor``.
    :param ros2_namespaces: ROS 2 topic namespace per drone (e.g.
        ``drone_0``). Used for the published topic
        ``/<ns>/isaac_odom`` so downstream consumers, which already
        subscribe on per-drone namespaces, get their odometry without
        remapping.

    The graph contains one ``OnPlaybackTick`` + ``IsaacReadSimulationTime``
    pair driving every publisher, which keeps node count low even for
    multi-drone scenes. ``ROS2PublishOdometry`` emits Isaac's native ENU
    world frame with body-frame twist (``publishRawVelocities=False``
    rotates world velocities into the chassis frame); any conversion to
    a different world convention (NWU, NED, etc.) is the responsibility
    of the downstream consumer.
    """
    # Enable the bridge extension lazily so users who don't ask for
    # ROS 2 pay nothing at startup. We must do this BEFORE creating
    # OmniGraph nodes that reference isaacsim.ros2.bridge.*.
    # The correct path in Kit 106+/Isaac Sim 6 is via omni.kit.app.
    import omni.kit.app
    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled('isaacsim.ros2.bridge'):
        manager.set_extension_enabled_immediate(
            'isaacsim.ros2.bridge', True,
        )

    import omni.graph.core as og
    import usdrt.Sdf

    keys = og.Controller.Keys
    graph_path = '/World/RS_IsaacROS2Bridge'

    nodes = [
        ('OnPlaybackTick', 'isaacsim.core.nodes.OnPhysicsStep'),
        ('ReadSimTime', 'isaacsim.core.nodes.IsaacReadSimulationTime'),
        ('PublishClock', 'isaacsim.ros2.bridge.ROS2PublishClock'),
    ]
    set_values = [
        ('PublishClock.inputs:topicName', 'clock'),
    ]
    connects = [
        ('OnPlaybackTick.outputs:step', 'PublishClock.inputs:execIn'),
        ('ReadSimTime.outputs:simulationTime', 'PublishClock.inputs:timeStamp'),
    ]

    for isaac_ns, ros_ns in zip(isaac_namespaces, ros2_namespaces):
        # Point at the rigid-body base_link prim (not the Xform
        # container) — ComputeOdometry reads physics state and needs
        # a prim that has a RigidBodyAPI applied, which is where
        # vehicle.py attaches the drone mass/inertia.
        prim_path = f'/World/{isaac_ns}/basic_quadrotor/Geometry/base_link'
        node_compute = f'ComputeOdometry_{ros_ns}'
        node_publish = f'PublishOdometry_{ros_ns}'
        nodes.extend([
            (node_compute, 'isaacsim.core.nodes.IsaacComputeOdometry'),
            (node_publish, 'isaacsim.ros2.bridge.ROS2PublishOdometry'),
        ])
        set_values.extend([
            (
                f'{node_compute}.inputs:chassisPrim',
                [usdrt.Sdf.Path(prim_path)],
            ),
            # Publish per-drone on /<ros_ns>/isaac_odom. Downstream ROS
            # bridges (e.g. an ENU→NED converter) are expected to pick
            # this up and forward to PX4's vehicle_visual_odometry topic
            # under /<ros_ns>/fmu/in/.
            (f'{node_publish}.inputs:topicName', 'isaac_odom'),
            (f'{node_publish}.inputs:nodeNamespace', f'/{ros_ns}'),
            (f'{node_publish}.inputs:odomFrameId', 'world_enu'),
            (f'{node_publish}.inputs:chassisFrameId', ros_ns),
            # Rotate world-frame velocities into the chassis frame, as
            # nav_msgs/Odometry specifies twist in child_frame_id. This
            # matches what mocap_from_isaac's UDP path was doing
            # manually with _rotate_vec_by_quat_inv.
            (f'{node_publish}.inputs:publishRawVelocities', False),
        ])
        connects.extend([
            (
                'OnPlaybackTick.outputs:step',
                f'{node_compute}.inputs:execIn',
            ),
            (
                f'{node_compute}.outputs:execOut',
                f'{node_publish}.inputs:execIn',
            ),
            (
                f'{node_compute}.outputs:position',
                f'{node_publish}.inputs:position',
            ),
            (
                f'{node_compute}.outputs:orientation',
                f'{node_publish}.inputs:orientation',
            ),
            (
                f'{node_compute}.outputs:linearVelocity',
                f'{node_publish}.inputs:linearVelocity',
            ),
            (
                f'{node_compute}.outputs:angularVelocity',
                f'{node_publish}.inputs:angularVelocity',
            ),
            (
                'ReadSimTime.outputs:simulationTime',
                f'{node_publish}.inputs:timeStamp',
            ),
        ])

    # Use GRAPH_PIPELINE_STAGE_ONDEMAND to match the pattern used by
    # the stock Isaac Sim OnPhysicsStep tests — with that pipeline
    # stage the graph is evaluated every time the application ticks,
    # and the OnPhysicsStep node fires on every physics step that
    # occurred since the last app tick. The default "execution"
    # evaluator + ONDEMAND pipeline is what the upstream tests
    # (test_physics_step.py) exercise; we follow that recipe.
    og.Controller.edit(
        {
            'graph_path': graph_path,
            'pipeline_stage': og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        },
        {
            keys.CREATE_NODES: nodes,
            keys.SET_VALUES: set_values,
            keys.CONNECT: connects,
        },
    )
    ns_pairs = ', '.join(
        f'{i}→/{r}/isaac_odom'
        for i, r in zip(isaac_namespaces, ros2_namespaces)
    )
    print(f'[ROS2 BRIDGE] OmniGraph at {graph_path} publishing /clock + '
          f'{ns_pairs}')
    return graph_path


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

    # Optional ROS 2 bridge OmniGraph: native /clock + per-drone
    # nav_msgs/Odometry publishing via the Isaac Sim ROS 2 Bridge
    # extension. Replaces the clock_from_px4 + UDP-ground-truth path
    # with a single OmniGraph driven by the simulation playback tick,
    # which is both simpler and cache-warmer than the two-process
    # shim (no rclpy context living inside Carb).
    if args.ros2_namespaces:
        ros2_ns_list = [
            ns.strip() for ns in args.ros2_namespaces.split(',') if ns.strip()
        ]
        if len(ros2_ns_list) != args.num_drones:
            raise SystemExit(
                f'[ROS2 BRIDGE] --ros2_namespaces has '
                f'{len(ros2_ns_list)} entries but --num_drones={args.num_drones}; '
                f'these must match.'
            )
        bridge_graph_path = _setup_ros2_bridge_graph(namespaces, ros2_ns_list)
        # Keep a global reference so the main loop can push-evaluate
        # the graph on every world.step(). The ONDEMAND pipeline stage
        # means Isaac will only evaluate the graph when we tell it to,
        # which is exactly what we want to lock mocap publishing to
        # the physics rate (500 Hz) without paying for a full
        # simulation_app.update() every tick.
        import omni.timeline
        omni.timeline.get_timeline_interface().play()
        print('[ROS2 BRIDGE] Timeline started '
              '(required for OnPhysicsStep to fire).')
    else:
        bridge_graph_path = None

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

    # When the ROS 2 bridge OmniGraph is active we push-evaluate the
    # graph directly after every world.step() via og.Controller.
    # evaluate_sync. This locks mocap publishing to the physics rate
    # (500 Hz) — equivalent to the UDP broadcaster's cadence — and
    # avoids the per-tick overhead of simulation_app.update() (which
    # also processes render/event queues we don't care about in
    # headless MAVLink HIL).
    if bridge_graph_path is not None:
        import omni.graph.core as _og

        def _evaluate_bridge(_p=bridge_graph_path):
            _og.Controller.evaluate_sync(_p)
    else:
        _evaluate_bridge = None

    while simulation_app.is_running():
        t_loop_begin = time.monotonic()
        manager.step(physics_dt)
        world.step(render=False)
        if _evaluate_bridge is not None:
            _evaluate_bridge()
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
