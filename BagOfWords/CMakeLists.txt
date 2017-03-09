cmake_minimum_required(VERSION 2.8)

project (BagOfWords)

set(CXX g++)
add_definitions(-std=c++11) 

set (OpenCV_DIR "/usr/local/")
find_package( OpenCV REQUIRED )

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Utilities/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../LieGroup/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Lib/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Lib/Eigen/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Classifier/)
include_directories(${OpenCV_INCLUDE_DIRS})


file(GLOB CPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)


add_executable(${PROJECT_NAME} ${CPP_SOURCES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} Utilities LieGroup Classifier ${OpenCV_LIBS})
