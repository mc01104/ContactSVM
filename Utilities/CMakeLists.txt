cmake_minimum_required(VERSION 2.8)

project (Utilities)

set(CXX g++)
add_definitions(-std=c++11) 

file(GLOB SOURCES *.cpp)


include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../LieGroup/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Lib/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Lib/Eigen/)


add_library(${PROJECT_NAME} STATIC ${SOURCES})
