cmake_minimum_required(VERSION 2.8.3)
project(convert_map)

find_package(catkin_simple REQUIRED)
find_package(Boost 1.54.0 REQUIRED COMPONENTS filesystem) 
catkin_simple(ALL_DEPS_REQUIRED)

#######
# APP #
#######
cs_add_executable(convert_map
  src/main.cc
)

target_link_libraries(${PROJECT_NAME}
${Boost_LIBRARIES}
)

############
## EXPORT ##
############
cs_install()
cs_export()
