cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 20)

project(example13)

#Set oneapi install location
if (NOT ONEAPI_INSTALL_LOCATION)
    set(ONEAPI_INSTALL_LOCATION /opt/intel/oneapi)
    message("Set ONEAPI_INSTALL_LOCATION to ${ONEAPI_INSTALL_LOCATION}")
  endif()

set(HOME /home/lfield)
set(VecGeom_DIR ${HOME}/workspace/lib/cmake/VecGeom)
set(VecCore_DIR ${HOME}/workspace/lib/cmake/VecCore)
#set(G4HepEm_DIR /home/lfield/repos/AdePT/external/g4hepem/build/InstallTree)
#set(G4HepEm_INCLUDE_DIR /home/lfield/repos/AdePT/external/g4hepem/G4HepEm/include)
find_package(Geant4 REQUIRED)
find_package(VecGeom REQUIRED)
find_package(VecCore REQUIRED)
#find_package(G4HepEm REQUIRED)

include(${Geant4_USE_FILE})
include_directories(
   ${HOME}/AdePT/external/g4hepem/G4HepEm/G4HepEmData/include
   ${HOME}/AdePT/external/g4hepem/G4HepEm/G4HepEmInit/include
   ${HOME}/AdePT/external/g4hepem/G4HepEm/G4HepEmRun/include
   ${HOME}/workspace/include
   ${HOME}/AdePT/base/inc/CopCore/include
   ${HOME}/AdePT/base/inc/
   ${HOME}/AdePT/magneticfield/inc/
  )

set (SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/example13
  ${CMAKE_CURRENT_LIST_DIR}/example13.dp
)
add_compile_definitions(MGONGPU_COMMONRAND_ONHOST)

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread ")

link_directories(
  ${HOME}/AdePT/external/g4hepem/build/lib
  ${HOME}/workspace/lib/
  )


add_executable(example13.exe ${SOURCES})

target_compile_options(example13.exe PRIVATE -g -fsycl -ferror-limit=1 -fsycl-unnamed-lambda -Wall -pthread)
target_link_libraries(example13.exe PRIVATE sycl -fsycl ${Geant4_LIBRARIES} vgdml vecgeom g4HepEmData g4HepEmInit) 




