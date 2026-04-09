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
"""Launch file for the Isaac Sim MAVLink scene."""
import os

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _launch_setup(context, *args, **kwargs):
    isaac_sim_python = os.environ.get('ISAAC_SIM_PYTHON')
    if not isaac_sim_python:
        raise EnvironmentError(
            'ISAAC_SIM_PYTHON environment variable is not set. '
            'Source Isaac Sim setup before launching.'
        )

    script_path = os.path.join(
        get_package_prefix('rs_isaac_uav_sim'),
        'lib', 'rs_isaac_uav_sim', 'scene_mavlink_sim.py'
    )

    num_drones = LaunchConfiguration('num_drones').perform(context)
    headless = LaunchConfiguration('headless').perform(context)
    config_file = LaunchConfiguration('config_file').perform(context)
    base_port = LaunchConfiguration('base_port').perform(context)
    connection_ip = LaunchConfiguration('connection_ip').perform(context)

    cmd = [isaac_sim_python, script_path, '--num_drones', num_drones,
           '--base_port', base_port, '--connection_ip', connection_ip]
    if headless == 'true':
        cmd.append('--headless')
    if config_file:
        cmd.extend(['--config', config_file])

    return [ExecuteProcess(cmd=cmd, output='screen')]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'num_drones',
            default_value='1',
            description='Number of drones to spawn'
        ),
        DeclareLaunchArgument(
            'headless',
            default_value='false',
            description='Run Isaac Sim without GUI'
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value='',
            description='Path to drone YAML configuration file (optional)'
        ),
        DeclareLaunchArgument(
            'base_port',
            default_value='4560',
            description='Base MAVLink TCP port; drone i uses base_port+i+1'
        ),
        DeclareLaunchArgument(
            'connection_ip',
            default_value='localhost',
            description='IP address of the PX4 SITL instance'
        ),
        OpaqueFunction(function=_launch_setup),
    ])
