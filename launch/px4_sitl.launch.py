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
"""Launch file for Isaac Sim MAVLink scene with PX4 SITL instances."""
import os

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler,
)
from launch.event_handlers import OnShutdown
from launch.substitutions import LaunchConfiguration


def _clean(_launch_context=None) -> None:
    os.popen('pkill -x px4')


def _launch_setup(context, *args, **kwargs):  # noqa: ARG001
    isaac_sim_python = os.environ.get('ISAAC_SIM_PYTHON')
    if not isaac_sim_python:
        raise EnvironmentError(
            'ISAAC_SIM_PYTHON environment variable is not set. '
            'Source Isaac Sim setup before launching.'
        )

    px4_sitl_build_dir = LaunchConfiguration('px4_sitl_build_dir').perform(context)
    if not px4_sitl_build_dir:
        raise EnvironmentError(
            'PX4_SITL_BUILD_DIR environment variable is not set and '
            'px4_sitl_build_dir launch argument was not provided.'
        )

    num_drones = int(LaunchConfiguration('num_drones').perform(context))
    headless = LaunchConfiguration('headless').perform(context)
    config_file = LaunchConfiguration('config_file').perform(context)
    display_terminal = LaunchConfiguration('display_terminal').perform(context)
    base_port = LaunchConfiguration('base_port').perform(context)
    connection_ip = LaunchConfiguration('connection_ip').perform(context)
    verbose = LaunchConfiguration('verbose').perform(context)

    script_path = os.path.join(
        get_package_prefix('rs_isaac_uav_sim'),
        'lib', 'rs_isaac_uav_sim', 'scene_mavlink_sim.py'
    )

    isaac_cmd = [isaac_sim_python, script_path, '--num_drones', str(num_drones),
                 '--base_port', base_port, '--connection_ip', connection_ip]
    if headless == 'true':
        isaac_cmd.append('--headless')
    if verbose == 'true':
        isaac_cmd.append('--verbose')
    if config_file:
        isaac_cmd.extend(['--config', config_file])

    actions = [ExecuteProcess(cmd=isaac_cmd, output='screen')]

    px4_binary = os.path.join(px4_sitl_build_dir, 'bin', 'px4')
    px4_rootfs = os.path.join(px4_sitl_build_dir, 'etc')

    for i in range(num_drones):
        # PX4_INSTANCE = i+1 so that TCP port = 4560 + (i+1) = base_port + drone_index + 1,
        # matching the port assigned by scene_mavlink_sim (base_port + vehicle_id, vehicle_id=i+1).
        px4_instance = i + 1
        work_dir = f'/tmp/px4_sitl_{px4_instance}'
        os.makedirs(work_dir, exist_ok=True)

        env = os.environ.copy()
        env['PX4_SYS_AUTOSTART'] = '10016'
        env['HEADLESS'] = '1'
        env['PX4_INSTANCE'] = str(px4_instance)

        px4_cmd = [px4_binary, '-i', str(px4_instance), px4_rootfs, '-w', work_dir]

        if display_terminal == 'tmux':
            env_exports = (
                f'export PX4_SYS_AUTOSTART=10016 HEADLESS=1 PX4_INSTANCE={px4_instance}'
            )
            px4_cmd_str = ' '.join(px4_cmd)
            bash_cmd = f'cd {work_dir} && {env_exports} && {px4_cmd_str}'
            window_name = f'PX4_{px4_instance}'
            actions.append(
                ExecuteProcess(
                    cmd=['tmux', 'new-window', '-n', window_name,
                         '--', 'bash', '-c', bash_cmd],
                    output='screen',
                )
            )
        else:
            actions.append(
                ExecuteProcess(
                    cmd=px4_cmd,
                    additional_env={
                        'PX4_SYS_AUTOSTART': '10016',
                        'HEADLESS': '1',
                        'PX4_INSTANCE': str(px4_instance),
                    },
                    cwd=work_dir,
                    output='screen',
                )
            )

    if display_terminal == 'tmux':
        actions.append(
            RegisterEventHandler(
                OnShutdown(on_shutdown=[OpaqueFunction(function=_clean)])
            )
        )

    return actions


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
            default_value=os.path.join(
                get_package_prefix('rs_isaac_uav_sim'),
                'share', 'rs_isaac_uav_sim', 'config', 'px4_iris.yaml'
            ),
            description='Path to drone YAML configuration file'
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
        DeclareLaunchArgument(
            'px4_sitl_build_dir',
            default_value=os.environ.get('PX4_SITL_BUILD_DIR', ''),
            description='Path to PX4 SITL build directory (default: PX4_SITL_BUILD_DIR env var)'
        ),
        DeclareLaunchArgument(
            'verbose',
            default_value='false',
            description='Enable verbose MAVLink warnings (rate/timeout)'
        ),
        DeclareLaunchArgument(
            'display_terminal',
            default_value='none',
            choices=['tmux', 'none'],
            description='Launch each PX4 instance in a tmux window (tmux) or inline (none)'
        ),
        OpaqueFunction(function=_launch_setup),
    ])
