ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO}

# ── System deps ─────────────────────────────────────────────────────
# Isaac Sim 6.0 needs a working Vulkan loader (libvulkan1) plus the
# OpenGL / EGL / GLU stack for the Omniverse renderer and Iray. The
# X11 libs are pulled in because the kit windowing layer dlopens them
# even in headless mode (they only need to load, not connect to a
# display). Without these, omni.gpu_foundation fails to bring up
# Vulkan and physx then reports "no suitable CUDA GPU was found".
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-yaml \
    git \
    ca-certificates \
    libgl1 \
    libglu1-mesa \
    libglvnd0 \
    libegl1 \
    libgles2 \
    libvulkan1 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libsm6 \
    libice6 \
    libxt6 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/isaacvenv/bin:$PATH"
ENV ISAAC_SIM_PYTHON=/opt/isaacvenv/bin/python
ENV OMNI_KIT_ACCEPT_EULA=YES

# ── PX4-Autopilot v1.16 SITL ────────────────────────────────────────
# scene_mavlink_sim.py speaks MAVLink to a PX4 SITL instance over TCP.
# Build the autopilot once at image-build time so the simulation is
# usable out of the box. We use Tools/setup/ubuntu.sh with --no-nuttx
# (we don't need the embedded toolchain) and --no-sim-tools (Isaac Sim
# replaces Gazebo / jMAVSim). DONT_RUN=1 stops `make` after the px4
# binary is produced. The .git directories of all submodules are
# pruned afterwards to shave roughly half a gigabyte off the image.
ARG PX4_VERSION=v1.16.0
ENV PX4_SITL_BUILD_DIR=/opt/PX4-Autopilot/build/px4_sitl_default
RUN git clone --depth 1 --branch ${PX4_VERSION} --recursive \
        https://github.com/PX4/PX4-Autopilot.git /opt/PX4-Autopilot \
    && bash /opt/PX4-Autopilot/Tools/setup/ubuntu.sh --no-nuttx --no-sim-tools \
    && DONT_RUN=1 make -C /opt/PX4-Autopilot px4_sitl_default \
    && find /opt/PX4-Autopilot -type d -name '.git' -prune -exec rm -rf {} + \
    && find /opt/PX4-Autopilot -type d -name '__pycache__' -prune -exec rm -rf {} + \
    && rm -rf /var/lib/apt/lists/*

# ── Initialize rosdep ───────────────────────────────────────────────
RUN rosdep init || true

# ── User setup (reuse the existing 'ubuntu' user from the base image) ─
# /opt/isaacvenv is created here as ubuntu-owned so the Isaac Sim pip
# install (run as ubuntu below) lands in a tree the runtime user can
# write to — Isaac Sim creates shader / extension caches under
# isaacsim/kit/{cache,data,logs} at runtime and would otherwise fail
# with "Permission denied" errors.
RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu \
    && mkdir -p /home/ubuntu /opt/isaacvenv \
    && chown -R ubuntu:ubuntu /home/ubuntu /opt/isaacvenv

USER ubuntu

# ── Isaac Sim 6.0 (pip install as 'ubuntu' into /opt/isaacvenv) ──────
# Requires Python 3.12 (provided by ros:jazzy) and NVIDIA GPU drivers
# on the host. The [all,extscache] extras pull in core API, physics,
# kit extensions, and pre-cached Omniverse extensions (omni.physx,
# pxr, etc.).
RUN python3 -m venv --system-site-packages /opt/isaacvenv \
    && /opt/isaacvenv/bin/pip install --no-cache-dir \
        "isaacsim[all,extscache]==6.0.0" \
        --extra-index-url https://pypi.nvidia.com \
    && /opt/isaacvenv/bin/pip install --no-cache-dir pymavlink

RUN rosdep update

WORKDIR /workspace

# Persist the chosen distro so scripts can discover it at runtime
ENV ROS_DISTRO=${ROS_DISTRO}

# Source ROS + the workspace install in every interactive shell.
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/ubuntu/.bashrc \
    && echo '[ -f /workspace/install/setup.bash ] && source /workspace/install/setup.bash' \
       >> /home/ubuntu/.bashrc

# Default to an interactive shell so `docker compose run` / `exec`
# drop the user straight into a usable environment.
CMD ["/bin/bash"]
