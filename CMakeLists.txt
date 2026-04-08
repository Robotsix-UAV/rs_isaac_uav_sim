cmake_minimum_required(VERSION 3.8)
project(rs_isaac_uav_sim)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  isaac_run/scene_mavlink_sim.py
  isaac_run/scene_basic_quadrotor.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY assets/
  DESTINATION share/${PROJECT_NAME}/assets
)

if(DEFINED ENV{ISAAC_SIM_PYTHON})
  message(STATUS "[rs_isaac_uav_sim] Installing into Isaac Sim Python: $ENV{ISAAC_SIM_PYTHON}")
  execute_process(
    COMMAND $ENV{ISAAC_SIM_PYTHON} -m pip install -e ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE _pip_result
  )
  if(NOT _pip_result EQUAL 0)
    message(WARNING "[rs_isaac_uav_sim] Isaac Sim pip install failed (exit code: ${_pip_result})")
  endif()
else()
  message(STATUS "[rs_isaac_uav_sim] ISAAC_SIM_PYTHON not set — skipping Isaac Sim install")
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # This is a Python-only package; pretend the C/C++ linters are
  # already handled so ament_lint_auto doesn't add cppcheck/cpplint/
  # uncrustify tests that fail with "No files found".
  set(ament_cmake_cppcheck_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_uncrustify_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(pyright test/lint/test_pyright.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )

  ament_add_pytest_test(test_state_rotations test/unit/test_state_rotations.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )

  ament_add_pytest_test(test_mavlink_scenarios test/integration/test_mavlink_scenarios.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    TIMEOUT 300
    ENV ISAAC_SIM_PYTHON=$ENV{ISAAC_SIM_PYTHON}
  )

  ament_add_pytest_test(test_px4_sitl_takeoff test/integration/test_px4_sitl_takeoff.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    TIMEOUT 600
    ENV ISAAC_SIM_PYTHON=$ENV{ISAAC_SIM_PYTHON}
        PX4_SITL_BUILD_DIR=$ENV{PX4_SITL_BUILD_DIR}
  )
endif()

ament_package()
