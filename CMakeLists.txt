cmake_minimum_required(VERSION 3.3)
project(miroz)

set(OpenCV_DIR C:/Users/Anna/Documents/opencv3.4.3/build)
#set(OPENCV_EXTRA_MODULES_PATH C:/Users/Anna/Documents/opencv_contrib-3.4.3/modules)
set(OPENCV_ENABLE_NONFREE 1)
set(OpenCV_VERSION 3)
FIND_PACKAGE(OpenCV REQUIRED)

include_directories( ${MY_SOURCE_DIR}/classes/)
include_directories( ${MY_SOURCE_DIR}/../test_5 )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++17)


set(SOURCE_FILES train.cpp)
add_executable(miroz ${SOURCE_FILES})

TARGET_LINK_LIBRARIES(miroz ${OpenCV_LIBS})