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

import carb

from isaacsim import SimulationApp
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.api.world import World
import isaacsim.core.utils.prims as prim_utils
import numpy as np
from omni.physx import get_physx_visualization_interface
from pxr import Gf, UsdGeom, UsdLux

DRONE_USD = (
    '/home/robotsix-docker/LS2N/ros2_ws/src/rs_isaac_uav_sim/assets/'
    'basic_quadrotor/basic_quadrotor.usda'
)
DRONE_PRIM_PATH = '/World/basic_quadrotor'

simulation_app = SimulationApp({'headless': False})

world = World(stage_units_in_meters=1.0)

# Ground plane with physics
GroundPlane(prim_path='/World/GroundPlane', z_position=0.0,
            color=np.array([0.0, 0.0, 1.0]))

# Dome light for uniform ambient illumination
stage = world.stage
dome = UsdLux.DomeLight.Define(stage, '/World/DomeLight')
dome.CreateIntensityAttr(500.0)

# Distant (sun) light for directional shading
distant = UsdLux.DistantLight.Define(stage, '/World/DistantLight')
distant.CreateIntensityAttr(1000.0)
distant.CreateAngleAttr(0.53)
xformable = UsdGeom.Xformable(stage.GetPrimAtPath('/World/DistantLight'))
xformable.ClearXformOpOrder()
xformable.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 0.0))

# Load basic_quadrotor USD at (0, 0, 0.5) — slightly above ground
prim_utils.create_prim(
    prim_path=DRONE_PRIM_PATH,
    prim_type='Xform',
    position=(0.0, 0.0, 0.5),
    usd_path=DRONE_USD,
)

world.reset()

# Enable collision mesh visualization (USD overlay and PhysX wireframes)
carb.settings.get_settings().set_int(
    '/persistent/physics/visualizationDisplayColliders', 2)
vis = get_physx_visualization_interface()
vis.enable_visualization(True)
vis.set_visualization_parameter('CollisionShapes', True)
vis.set_visualization_scale(1.0)

print('[SCENE] basic_quadrotor scene ready. Close the Isaac Sim window to exit.')

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
