- GPS and no-GPS configurations
- Multirotors with extra motors (not only 4)
- Multidrone control
- See if we limit GPU->CPU transfer to play closer to RT


## Architectural Issues (Isaac Sim 6.0 audit)

### Major

- [ ] **[M1] Hardcoded absolute asset paths**
  - `vehicle.py:16–18`, `scene_basic_quadrotor.py:22`
  - Paths are pinned to `/home/robotsix-docker/...` — breaks on any other machine or container.
  - Fix: derive path dynamically via `pathlib.Path(__file__).resolve()` relative to package root.

- [ ] **[M2] Invalid `SimulationApp` config keys silently discarded**
  - `scene_mavlink_sim.py:27–32`
  - Keys `"physics_device"`, `"device"`, `"scene_graph_instancing"` are not valid launcher keys and are silently ignored — GPU config at the launcher level is illusory.
  - Fix: remove invalid keys; use `"physics_gpu": 0` for GPU selection; move tensor device to `World(device="cuda")`.

- [ ] **[M3] State reads on `Articulation` instead of `RigidPrim`**
  - `vehicle.py:222–223`
  - `get_velocities()` and `get_world_poses()` are called on the `Articulation` view (`_prims`) while forces are applied via `_rigid_view`, creating a frame inconsistency risk.
  - Fix: consolidate all state reads on `self._rigid_view`.

- [ ] **[M4] Manual step loop instead of `world.add_physics_callback`**
  - `scene_mavlink_sim.py:152–156`
  - Physics callback receives configured dt, not the actual PhysX dt — mismatch if substepping behaviour changes.
  - Fix: register `manager.step` via `world.add_physics_callback("drone_step", manager.step)`.

### Minor

- [ ] **[m1] Wildcard prim path captures non-drone prims**
  - `vehicle.py:159` — `/World/*/basic_quadrotor...` will match any prim, not only drones.
  - Fix: use a prefixed expression e.g. `/World/drone_*/basic_quadrotor...`.

- [ ] **[m2] No ROS2 launch parameter bridge for physical/sensor config**
  - `config.py`, `scene_mavlink_sim.py` — `QuadrotorParams`, `SensorParams`, etc. are hardcoded dataclass defaults with no external override mechanism.
  - Fix: accept a YAML/JSON config file via argparse or ROS2 launch argument.

- [ ] **[m3] Non-canonical `World` import path**
  - `scene_basic_quadrotor.py:17`, `scene_mavlink_sim.py:37`
  - `from isaacsim.core.api.world import World` should be `from isaacsim.core.api import World`.

- [ ] **[m4] `_gt_state` not initialized in `__init__`**
  - `mavlink_backend.py:135` — `AttributeError` if `send_ground_truth` is called before `update_state`.
  - Fix: add `self._gt_state = None` in `__init__` and guard `send_ground_truth`.

- [ ] **[m5] Diagnostic unconditionally accesses `backends[0]`**
  - `vehicle.py:291` — crashes if `num_drones == 0`.
  - Fix: guard with `if self.num_drones == 0: return`.

  - check code quelity with expert dev