cmake_minimum_required(VERSION 3.14)
project(VMVS)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(VTK REQUIRED)

include_directories(include
                    ${OpenCV_INCLUDE_DIRS}
                    ${VTK_USE_FILE})
message(${VTK_INCLUDE_DIRS})
#message("test")
aux_source_directory(src SOURCE_FILES)

add_executable(VMVS ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
#message(${VTK_LIBS})
