cmake_minimum_required(VERSION 2.8)
project(risiko)
find_package( OpenCV REQUIRED )

set(CMAKE_CXX_STANDARD 14)


set(SOURCE_FILES src/main.cpp )
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(SOURCE_FILES src/thresh.cpp )
add_executable(thresh ${SOURCE_FILES})


include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries( thresh ${OpenCV_LIBS})
