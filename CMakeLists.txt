cmake_minimum_required(VERSION 2.8)

project (ContactSVM CXX)

set(CMAKE_BUILD_TYPE Debug)
add_definitions(-DLINUX)

add_subdirectory(Utilities)
add_subdirectory(LieGroup)
add_subdirectory(BagOfWords)
add_subdirectory(Classifier)
