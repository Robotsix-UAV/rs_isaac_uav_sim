from setuptools import find_packages, setup

package_name = "rs_isaac_uav_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Damien SIX",
    maintainer_email="damien@robotsix.net",
    description="UAV simulation with NVIDIA Isaac Sim 6.0 and ROS2",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
