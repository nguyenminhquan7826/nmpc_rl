cmake_minimum_required(VERSION 3.16)
project(RL_NMPC LANGUAGES CXX)

# All executable targets, including subdirectories/multi-config generators,
# are placed directly in the requested binary directory (normally build/).
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}")
endforeach()

enable_testing()
option(BUILD_FIXED_NMPC "Build native NMPC and four-mode runner (CasADi required)" ON)
add_subdirectory(deployment_cpp14)
