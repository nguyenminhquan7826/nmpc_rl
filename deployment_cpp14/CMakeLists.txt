cmake_minimum_required(VERSION 3.10)
project(nmpc_deployment_rx LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_library(telemetry src/telemetry.cpp src/yaw_kalman.cpp)
target_include_directories(telemetry PUBLIC include)
target_compile_options(telemetry PRIVATE -Wall -Wextra -Wpedantic)
add_executable(telemetry_rx src/telemetry_rx.cpp)
target_link_libraries(telemetry_rx PRIVATE telemetry)
target_compile_options(telemetry_rx PRIVATE -Wall -Wextra -Wpedantic)
enable_testing()
add_executable(test_telemetry tests/test_telemetry.cpp)
target_link_libraries(test_telemetry PRIVATE telemetry)
add_test(NAME telemetry_contract COMMAND test_telemetry)
add_executable(test_yaw_kalman tests/test_yaw_kalman.cpp)
target_link_libraries(test_yaw_kalman PRIVATE telemetry)
add_test(NAME yaw_kalman_contract COMMAND test_yaw_kalman)

# Receiver remains dependency-free; enable the native solver explicitly.
option(BUILD_FIXED_NMPC "Build offline Fixed NMPC parity probe (CasADi required)" OFF)
if(BUILD_FIXED_NMPC)
  find_package(casadi 3.7.2 CONFIG QUIET)
  if(NOT casadi_FOUND)
    # Optional configure-time discovery of a Python wheel's native SDK.
    # An explicit -Dcasadi_DIR or CMAKE_PREFIX_PATH is tried first above.
    find_program(NMPC_CASADI_PYTHON NAMES python3 python)
    if(NMPC_CASADI_PYTHON)
      execute_process(
        COMMAND "${NMPC_CASADI_PYTHON}" -c
          "import pathlib, casadi; print(pathlib.Path(casadi.__file__).parent / 'cmake')"
        RESULT_VARIABLE wheel_result
        OUTPUT_VARIABLE wheel_cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
      if(wheel_result EQUAL 0 AND EXISTS "${wheel_cmake_dir}/casadi-config.cmake")
        find_package(casadi 3.7.2 CONFIG QUIET PATHS "${wheel_cmake_dir}" NO_DEFAULT_PATH)
      endif()
    endif()
  endif()
  if(NOT casadi_FOUND)
    message(FATAL_ERROR
      "CasADi 3.7.2 C++ SDK not found. Set -Dcasadi_DIR=/path/to/casadi/cmake, "
      "or install casadi==3.7.2 for Python and select that interpreter with "
      "-DNMPC_CASADI_PYTHON=/path/to/python. "
      "Use -DBUILD_FIXED_NMPC=OFF for telemetry/online-learning core only.")
  endif()
  add_library(fixed_nmpc src/fixed_nmpc.cpp)
  target_include_directories(fixed_nmpc PUBLIC include)
  target_link_libraries(fixed_nmpc PUBLIC casadi::casadi)
  target_compile_options(fixed_nmpc PRIVATE -Wall -Wextra -Wpedantic)
  add_executable(fixed_nmpc_probe src/fixed_nmpc_probe.cpp)
  target_link_libraries(fixed_nmpc_probe PRIVATE fixed_nmpc)
endif()

# Standard-library online learning core can also be built without CasADi.
add_library(online_ac src/online_ac.cpp src/online_controller.cpp)
target_include_directories(online_ac PUBLIC include)
target_compile_options(online_ac PRIVATE -Wall -Wextra -Wpedantic)
add_executable(actor_critic_probe src/actor_critic_probe.cpp)
target_link_libraries(actor_critic_probe PRIVATE online_ac)
add_executable(online_controller_probe src/online_controller_probe.cpp)
target_link_libraries(online_controller_probe PRIVATE online_ac)
if(BUILD_FIXED_NMPC)
  # CasADi's GNU ABI selection must agree across linked C++ interfaces.
  target_link_libraries(online_ac PUBLIC casadi::casadi)
  add_executable(four_mode_probe src/four_mode_probe.cpp)
  target_link_libraries(four_mode_probe PRIVATE fixed_nmpc online_ac)
endif()
add_executable(test_online_contract tests/test_online_contract.cpp)
target_link_libraries(test_online_contract PRIVATE online_ac)
# Assertion-based contract checks must remain enabled in Release builds.
target_compile_options(test_online_contract PRIVATE -UNDEBUG)

add_test(NAME online_contract COMMAND test_online_contract)

add_library(reference_path src/reference_path.cpp)
target_include_directories(reference_path PUBLIC include)
target_compile_options(reference_path PRIVATE -Wall -Wextra -Wpedantic)
add_executable(reference_probe src/reference_probe.cpp)
target_link_libraries(reference_probe PRIVATE reference_path)
if(BUILD_FIXED_NMPC)
  # Reference vectors crossing the solver boundary must use CasADi's C++ ABI.
  target_link_libraries(reference_path PUBLIC casadi::casadi)
  add_executable(route_controller src/route_controller.cpp)
  target_link_libraries(route_controller PRIVATE fixed_nmpc online_ac reference_path)
endif()

# Optional host/Pi PTY regression checks; receiver runtime remains Python-free.
find_program(NMPC_RX_TEST_PYTHON NAMES python3)
if(UNIX AND NMPC_RX_TEST_PYTHON)
  add_test(NAME telemetry_serial_replay
    COMMAND "${NMPC_RX_TEST_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rx.py" $<TARGET_FILE:telemetry_rx>)
  add_test(NAME telemetry_serial_startup
    COMMAND "${NMPC_RX_TEST_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rx_startup.py" $<TARGET_FILE:telemetry_rx>)
  add_test(NAME telemetry_gyro_sign
    COMMAND "${NMPC_RX_TEST_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rx_gyro_sign.py" $<TARGET_FILE:telemetry_rx>)
  add_test(NAME telemetry_yaw_kalman
    COMMAND "${NMPC_RX_TEST_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rx_kalman.py" $<TARGET_FILE:telemetry_rx>)
  set_tests_properties(telemetry_yaw_kalman telemetry_gyro_sign telemetry_serial_replay telemetry_serial_startup PROPERTIES TIMEOUT 15)
endif()
