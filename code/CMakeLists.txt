project(lns)

cmake_minimum_required (VERSION 3.16)
set (CMAKE_CXX_STANDARD 14)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "RELEASE")
ENDIF()

include_directories(".")
file(GLOB SOURCES "*.cpp")
add_executable(main ${SOURCES})

# Find Boost
find_package(Boost REQUIRED COMPONENTS program_options filesystem graph iostreams system random regex serialization timer)
# ignore BOOST deprecated headers
add_definitions("-DBOOST_ALLOW_DEPRECATED_HEADERS")
add_definitions("-DBOOST_TIMER_ENABLE_DEPRECATED")
add_definitions("-DBOOST_BIND_GLOBAL_PLACEHOLDERS")
include_directories( ${Boost_INCLUDE_DIRS} )
