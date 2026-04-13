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
import re
import shutil
import tempfile

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnShutdown
from launch.substitutions import LaunchConfiguration


def _clean(_launch_context=None) -> None:
    os.popen('pkill -x px4')


_AIRFRAME_FILENAME_RE = re.compile(r'^(\d+)_')


def _prepare_rootfs_overlay(default_etc: str, airframe_file: str) -> tuple[str, int]:
    """
    Build a temporary PX4 rootfs that injects a custom airframe file.

    PX4's startup walks ``${R}etc/init.d-posix/airframes/`` looking for a
    file whose name starts with ``${SYS_AUTOSTART}_``. We create a temp
    directory that mirrors the stock PX4 ``etc/`` directory with symlinks,
    except for ``init.d-posix/airframes/``, which is a real directory
    populated with symlinks to each stock airframe *plus* a real copy of
    the caller-supplied airframe file. The returned path can be passed as
    PX4's positional ``rootfs`` argument.

    The airframe file name MUST start with ``<SYS_AUTOSTART>_``. The
    integer prefix is returned so the caller can set
    ``PX4_SYS_AUTOSTART`` to match.
    """
    if not os.path.isfile(airframe_file):
        raise FileNotFoundError(f'airframe_file not found: {airframe_file}')
    airframe_basename = os.path.basename(airframe_file)
    match = _AIRFRAME_FILENAME_RE.match(airframe_basename)
    if not match:
        raise ValueError(
            f'airframe_file name must start with "<id>_" (got '
            f'"{airframe_basename}")'
        )
    sys_autostart = int(match.group(1))

    # Build: <tmp>/init.d-posix/airframes/{stock symlinks + real custom file}
    # Everything else under default_etc is symlinked at the top level.
    overlay_root = tempfile.mkdtemp(prefix='rs_isaac_px4_rootfs_')

    for entry in os.listdir(default_etc):
        src = os.path.join(default_etc, entry)
        dst = os.path.join(overlay_root, entry)
        if entry != 'init.d-posix':
            os.symlink(src, dst)
            continue

        # init.d-posix: real dir so we can substitute airframes/ underneath.
        os.makedirs(dst)
        for sub in os.listdir(src):
            sub_src = os.path.join(src, sub)
            sub_dst = os.path.join(dst, sub)
            if sub != 'airframes':
                os.symlink(sub_src, sub_dst)
                continue
            os.makedirs(sub_dst)
            # Symlink all stock airframes
            for af in os.listdir(sub_src):
                os.symlink(os.path.join(sub_src, af), os.path.join(sub_dst, af))
            # Drop our custom airframe in as a real file (overwriting any
            # stock file that happened to share the same basename).
            dst_airframe = os.path.join(sub_dst, airframe_basename)
            if os.path.lexists(dst_airframe):
                os.unlink(dst_airframe)
            shutil.copy(airframe_file, dst_airframe)

    return overlay_root, sys_autostart


def _cleanup_rootfs_overlay(overlay_root: str) -> None:
    """Remove a temporary rootfs overlay created by ``_prepare_rootfs_overlay``."""
    try:
        shutil.rmtree(overlay_root)
    except OSError:
        pass


def _parse_extra_px4_env(extra_px4_env_str: str) -> tuple[dict[str, str], list[str]]:
    """
    Parse the ``extra_px4_env`` launch arg and strip PX4_PARAM_* keys.

    ``extra_px4_env`` is intended for PX4 SITL *runtime* env vars
    (``PX4_UXRCE_DDS_NS``, ``PX4_UXRCE_DDS_PORT``, …) read directly by
    the startup script. PX4 parameter overrides (``PX4_PARAM_*``) are
    rejected because they're silently no-op'd when the target value
    matches the firmware default — configure PX4 via an ``airframe_file``
    instead. Returns ``(accepted_env, rejected_keys)``.
    """
    accepted: dict[str, str] = {}
    rejected: list[str] = []
    for item in extra_px4_env_str.split():
        if '=' not in item:
            continue
        key, value = item.split('=', 1)
        if key.startswith('PX4_PARAM_'):
            rejected.append(key)
            continue
        accepted[key] = value
    return accepted, rejected


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
    airframe_file = LaunchConfiguration('airframe_file').perform(context)
    extra_px4_env_str = LaunchConfiguration('extra_px4_env').perform(context)
    ros2_namespaces = LaunchConfiguration('ros2_namespaces').perform(context)

    extra_px4_env, rejected_param_keys = _parse_extra_px4_env(extra_px4_env_str)

    actions: list = []

    if rejected_param_keys:
        actions.append(LogInfo(
            msg=('[rs_isaac_uav_sim] extra_px4_env contains PX4_PARAM_* keys '
                 f'which are ignored: {", ".join(sorted(rejected_param_keys))}.'
                 ' Configure PX4 parameters via the airframe_file argument '
                 'instead — PX4_PARAM_* env vars are silently no-op\'d when '
                 'the target value matches the firmware default.')
        ))

    # Default PX4 rootfs (the ``etc/`` dir inside PX4_SITL_BUILD_DIR). The
    # default value of SYS_AUTOSTART is the iris one (10016).
    default_px4_rootfs = os.path.join(px4_sitl_build_dir, 'etc')
    px4_rootfs = default_px4_rootfs
    sys_autostart = 10016
    overlay_root: str | None = None

    if airframe_file:
        overlay_root, sys_autostart = _prepare_rootfs_overlay(
            default_px4_rootfs, airframe_file
        )
        px4_rootfs = overlay_root
        actions.append(LogInfo(
            msg=f'[rs_isaac_uav_sim] using custom airframe '
                f'{os.path.basename(airframe_file)} (SYS_AUTOSTART={sys_autostart}) '
                f'via rootfs overlay {overlay_root}'
        ))

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
    if ros2_namespaces:
        isaac_cmd.extend(['--ros2_namespaces', ros2_namespaces])

    actions.append(ExecuteProcess(cmd=isaac_cmd, output='screen'))

    px4_binary = os.path.join(px4_sitl_build_dir, 'bin', 'px4')

    for i in range(num_drones):
        # PX4_INSTANCE = i+1 so that TCP port = 4560 + (i+1) = base_port + drone_index + 1,
        # matching the port assigned by scene_mavlink_sim (base_port + vehicle_id, vehicle_id=i+1).
        px4_instance = i + 1
        work_dir = f'/tmp/px4_sitl_{px4_instance}'
        os.makedirs(work_dir, exist_ok=True)

        px4_env = {
            'PX4_SYS_AUTOSTART': str(sys_autostart),
            'HEADLESS': '1',
            'PX4_INSTANCE': str(px4_instance),
        }
        # Per-instance runtime env vars (PX4_UXRCE_DDS_NS, PX4_UXRCE_DDS_PORT,
        # ...). Caller-supplied values win over the defaults above so a
        # downstream package can override e.g. the autostart id.
        px4_env.update(extra_px4_env)

        px4_cmd = [px4_binary, '-i', str(px4_instance), px4_rootfs, '-w', work_dir]

        if display_terminal == 'tmux':
            env_exports = 'export ' + ' '.join(
                f'{k}={v}' for k, v in px4_env.items()
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
                    additional_env=px4_env,
                    cwd=work_dir,
                    output='screen',
                )
            )

    # Always clean up the overlay on shutdown, even when display_terminal!=tmux.
    shutdown_handlers = [_clean] if display_terminal == 'tmux' else []
    if overlay_root is not None:
        def _drop_overlay(_ctx, _root=overlay_root):
            _cleanup_rootfs_overlay(_root)
        shutdown_handlers.append(_drop_overlay)

    if shutdown_handlers:
        actions.append(
            RegisterEventHandler(
                OnShutdown(on_shutdown=[
                    OpaqueFunction(function=fn) for fn in shutdown_handlers
                ])
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
            description=(
                'Path to the airframe YAML that describes the simulator-side '
                'physics and sensor parameters. Matching PX4 parameter '
                'overrides are NOT read from this file — pass them as an '
                'airframe shell script via airframe_file instead.'
            ),
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
        DeclareLaunchArgument(
            'airframe_file',
            default_value='',
            description=(
                'Optional path to a PX4 airframe shell script (e.g. '
                '"9011_crazy2fly"). The filename MUST start with the '
                '<SYS_AUTOSTART>_ prefix. When provided, the launch file '
                'builds a temporary rootfs overlay that injects this file '
                'into PX4\'s airframes directory, sets PX4_SYS_AUTOSTART to '
                'the parsed id, and passes the overlay to PX4. This is the '
                'ONLY supported way to override PX4 parameters — the '
                'PX4_PARAM_* env var mechanism is silently no-op\'d when '
                'the target value matches the firmware default, so it is '
                'not used here.'
            ),
        ),
        DeclareLaunchArgument(
            'ros2_namespaces',
            default_value='',
            description=(
                'Comma-separated per-drone ROS 2 namespaces (e.g. '
                '"crazy2fly1" for 1 drone). When set, Isaac Sim enables '
                'the isaacsim.ros2.bridge extension and publishes /clock '
                'plus /<ns>/isaac_odom via an OmniGraph. Empty disables '
                'the ROS 2 bridge path.'
            ),
        ),
        DeclareLaunchArgument(
            'extra_px4_env',
            default_value='',
            description=(
                'Extra env vars forwarded to every PX4 SITL instance, for '
                'non-parameter runtime configuration. Format: '
                'whitespace-separated KEY=VALUE pairs (e.g. '
                '"PX4_UXRCE_DDS_NS=crazy2fly1 PX4_UXRCE_DDS_PORT=8889"). '
                'Any PX4_PARAM_* key in this string is rejected with a '
                'warning — put parameter overrides in the airframe_file.'
            ),
        ),
        OpaqueFunction(function=_launch_setup),
    ])
