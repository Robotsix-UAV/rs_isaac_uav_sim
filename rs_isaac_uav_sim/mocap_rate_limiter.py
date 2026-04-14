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
MOCAP rate-limiter for Isaac Sim ground-truth odometry.

Isaac Sim's ROS 2 bridge OmniGraph publishes each drone's ground-truth
odometry on ``/<drone>/isaac_odom`` at the physics-step rate (typically
1000 Hz). That is faster than any real motion-capture system ships data
to a flight controller — PX4's EKF2 visual-odometry fusion buffer and
the ``EKF2_EV_DELAY`` compensation are tuned for real Vicon / Optitrack
rates (100-250 Hz). Feeding the raw 1000 Hz stream into PX4 overwhelms
the fusion buffer and the fused estimate drifts persistently off ground
truth, even when the physics drone is perfectly still.

This node subscribes to the raw ``isaac_odom`` topic and republishes
exactly the same message on a second topic at a configurable, bounded
rate (default 100 Hz). It deliberately does **no** frame conversion —
downstream ENU→NWU / mocap-publisher code stays unchanged.

Pair with the ``--mocap-input-topic`` arg on ``odom_enu_to_nwu_bridge``
(ls2n_drone_isaac_sim) so the LS2N mocap chain consumes the throttled
stream while any tools that need the full rate (plotters, logging) can
still subscribe to ``isaac_odom`` directly.
"""
from __future__ import annotations

import argparse
import sys

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
    QoSReliabilityPolicy,
)


class MocapRateLimiter(Node):
    """Republish ``isaac_odom`` at a bounded rate."""

    def __init__(self, input_topic: str, output_topic: str,
                 publish_rate_hz: float) -> None:
        super().__init__(
            'mocap_rate_limiter',
            parameter_overrides=[Parameter('use_sim_time', value=True)],
        )

        self._publish_period_s = 1.0 / max(publish_rate_hz, 1e-3)
        # Initialised to a large negative value so the very first
        # sample always passes — otherwise the startup transient
        # drops mocap until the subscriber's clock ticks past the
        # period boundary, which can be seconds in sim time.
        self._last_publish_time_s = -1.0e9

        # Match Isaac's ROS 2 bridge output QoS: sensor_data, best
        # effort. Republish on the same profile so downstream
        # consumers don't need to switch QoS.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self._pub = self.create_publisher(Odometry, output_topic, qos)
        self.create_subscription(
            Odometry, input_topic, self._on_isaac_odom, qos,
        )

        self.get_logger().info(
            f'Rate-limiting {input_topic} → {output_topic} at '
            f'{publish_rate_hz:.1f} Hz (period {self._publish_period_s*1e3:.1f} ms).'
        )

    def _on_isaac_odom(self, msg: Odometry) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if now_s - self._last_publish_time_s < self._publish_period_s:
            return
        self._last_publish_time_s = now_s
        # Pass-through: no frame conversion, no stamp rewrite. The
        # OmniGraph already stamps with IsaacReadSimulationTime (the
        # authoritative sim clock that PX4 uses for lockstep), so
        # propagating the original msg keeps timestamp_sample aligned
        # end-to-end with PX4's clock.
        self._pub.publish(msg)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--drone', default='crazy2fly1',
        help='Drone namespace. Used to build default input/output '
             'topics when they are not given explicitly.',
    )
    parser.add_argument(
        '--input-topic', default='',
        help='Raw Isaac odometry topic (default: /<drone>/isaac_odom).',
    )
    parser.add_argument(
        '--output-topic', default='',
        help='Rate-limited output topic (default: '
             '/<drone>/isaac_odom_mocap).',
    )
    parser.add_argument(
        '--publish-rate-hz', type=float, default=100.0,
        help='Target output rate in Hz. Real mocap systems run '
             '100-250 Hz; PX4 EKF2 default tuning assumes that range. '
             'Default 100 Hz.',
    )
    args, ros_argv = parser.parse_known_args(argv)

    input_topic = args.input_topic or f'/{args.drone}/isaac_odom'
    output_topic = args.output_topic or f'/{args.drone}/isaac_odom_mocap'

    rclpy.init(args=ros_argv)
    node = MocapRateLimiter(
        input_topic=input_topic,
        output_topic=output_topic,
        publish_rate_hz=args.publish_rate_hz,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
