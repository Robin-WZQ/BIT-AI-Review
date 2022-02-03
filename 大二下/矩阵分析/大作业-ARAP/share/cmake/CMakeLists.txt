# This file is shared for all assignments. It is meant to be included rather
# than executed directly.
#
option(LIBIGL_USE_STATIC_LIBRARY "Use LibIGL as static library" OFF)
# libigl options: choose your dependencies (by default everything is OFF except opengl) 
option(LIBIGL_WITH_OPENGL      "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW "Use OpenGL GLFW"    ON)

find_package(LIBIGL REQUIRED QUIET)
option(USE_SOLUTION "Use solution"  OFF)

# Add your project files
include_directories("include/")
if(USE_SOLUTION)
  file(GLOB SRCFILES solution/*.cpp)
else()
  file(GLOB SRCFILES src/*.cpp)
endif()
if(BUILD_TEST)
  list(APPEND SRCFILES test/main.cpp)
else()
  list(APPEND SRCFILES main.cpp)
endif()

add_executable(${PROJECT_NAME} ${SRCFILES})
target_link_libraries(${PROJECT_NAME} igl::core igl::opengl_glfw)

