cmake_minimum_required(VERSION 3.10)
project(DecisionMaking)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")


add_library(matplotlib_cpp INTERFACE)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
target_link_libraries(matplotlib_cpp INTERFACE
  Python3::Python
  Python3::Module
)

find_package(Python3 COMPONENTS NumPy)
if(Python3_NumPy_FOUND)
  target_link_libraries(matplotlib_cpp INTERFACE Python3::NumPy)
else()
  target_compile_definitions(matplotlib_cpp INTERFACE WITHOUT_NUMPY)
endif()

find_package(fmt REQUIRED)
include_directories( "/usr/include/eigen3")
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(decision_making
  ${PROJECT_SOURCE_DIR}/src/decision_making.cpp
  ${PROJECT_SOURCE_DIR}/src/env.cpp)
target_link_libraries(decision_making matplotlib_cpp fmt::fmt)
