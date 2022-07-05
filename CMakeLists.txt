cmake_minimum_required (VERSION 3.0.0)

##################
# Project settings

project (BayesGPc LANGUAGES CXX)

######################
# Compilation settings

set(IS_RELEASE false)

if (IS_RELEASE)
  set(CMAKE_BUILD_TYPE Release)
  set(BUILD_TYPE "WIN32")
else (IS_RELEASE)
  set(CMAKE_BUILD_TYPE Debug)
  set(BUILD_TYPE "")
endif (IS_RELEASE)

set(CMAKE_VERBOSE_MAKEFILE TRUE)

set (CMAKE_INCLUDE_CURRENT_DIR ON)
set (CMAKE_AUTOMOC ON)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY         ${CMAKE_CURRENT_SOURCE_DIR}/dist)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/dist)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/dist)

set (CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-error=unused -Wno-error=unused-parameter")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-variable")
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-non-c-typedef-for-linkage")
else()
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-but-set-variable")
endif()
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-zero-as-null-pointer-constant")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnull-dereference -Werror")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DRC_HAVE_QT")

enable_language(Fortran)

if(APPLE)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Dunix=1 -mmacosx-version-min=11.0")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

  # CBayes dependencies
  set (BLAS_LIB "-lblas -lc")
  set (LAPACK_LIB "-llapack")

  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=overloaded-virtual -Wno-error=reorder-ctor -Wno-error=sign-compare -Wno-error=ignored-qualifiers -Wno-error=deprecated-copy")
endif(APPLE)

if(WIN32)
  # CBayes dependencies
  # FORTRAN
  set(CMAKE_FC_COMPILER "${CMAKE_COMPILER_DIR}/gfortran.exe")
  
  set_source_files_properties( include/BayesGPc/ndlfortran_timer.f PROPERTIES COMPILE_FLAGS -fdefault-integer-8 )
  set_source_files_properties( include/BayesGPc/ndlfortran_linpack.f PROPERTIES COMPILE_FLAGS -fdefault-integer-8 )
  set_source_files_properties( include/BayesGPc/ndlfortran_lbfgsb.f PROPERTIES COMPILE_FLAGS -fdefault-integer-8 )

  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_WIN -Wno-error=overloaded-virtual -Wno-error=sign-compare -Wno-error=ignored-qualifiers -Wno-error=reorder -Wno-error=parentheses -Wno-error=non-template-friend -Wno-error=unused-value")
endif(WIN32)

set (BAYES_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set (BOOST_DIR "include/boost_math/include/boost")

include_directories(${BAYES_DIR}
                    ${BOOST_DIR}
                    )
if (APPLE AND ${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1")
else (APPLE AND ${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif (APPLE AND ${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffp-contract=fast -DBOOST_MATH_STANDALONE -DDBG")
set(CMAKE_Fortran_FLAGS "-funroll-all-loops -O${OPT_LEVEL} -fPIC")

set (BAYES_FILES
  ${BAYES_DIR}/CSearchComparison.cpp
  ${BAYES_DIR}/CSearchComparison.h
  ${BAYES_DIR}/CBayesianSearch.cpp
  ${BAYES_DIR}/CBayesianSearch.h
  ${BAYES_DIR}/GridSearch.cpp
  ${BAYES_DIR}/GridSearch.h
  ${BAYES_DIR}/CGp.cpp
  ${BAYES_DIR}/CGp.h
  ${BAYES_DIR}/CMltools.cpp
  ${BAYES_DIR}/CMltools.h
  ${BAYES_DIR}/ndlassert.cpp
  ${BAYES_DIR}/ndlassert.h
  ${BAYES_DIR}/ndlexceptions.h
  ${BAYES_DIR}/ndlstrutil.cpp
  ${BAYES_DIR}/ndlstrutil.h
  ${BAYES_DIR}/COptimisable.cpp
  ${BAYES_DIR}/COptimisable.h
  ${BAYES_DIR}/CMatrix.cpp
  ${BAYES_DIR}/CMatrix.h
  ${BAYES_DIR}/CNdlInterfaces.h
  ${BAYES_DIR}/ndlutil.cpp
  ${BAYES_DIR}/ndlutil.h
  ${BAYES_DIR}/ndlfortran.h
  ${BAYES_DIR}/ndlfortran_lbfgsb.h
  ${BAYES_DIR}/ndlfortran.f
  ${BAYES_DIR}/ndlfortran_linpack.f
  ${BAYES_DIR}/ndlfortran_lbfgsb.f
  ${BAYES_DIR}/ndlfortran_timer.f
  ${BAYES_DIR}/ndlfortran_linpack.h
  ${BAYES_DIR}/ndlfortran_timer.h
  ${BAYES_DIR}/lapack.h
  ${BAYES_DIR}/CKern.cpp
  ${BAYES_DIR}/CKern.h
  ${BAYES_DIR}/CTransform.cpp
  ${BAYES_DIR}/CTransform.h
  ${BAYES_DIR}/CDataModel.h
  ${BAYES_DIR}/CDist.cpp
  ${BAYES_DIR}/CDist.h
  ${BAYES_DIR}/CNoise.cpp
  ${BAYES_DIR}/CNoise.h
  ${BAYES_DIR}/version.h
  )

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set (CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fno-omit-frame-pointer")
set (CMAKE_CXX_FLAGS_RELEASE "-O3 -g -ffast-math -fno-finite-math-only")

##############
# Source files


add_library (${PROJECT_NAME} SHARED
  ${BAYES_FILES}
)

# add_executable (${PROJECT_NAME} "${BUILD_TYPE}" 
#   ${BAYES_FILES}
# )

####################
# External libraries

find_package (Qt5 REQUIRED COMPONENTS Core)
qt5_use_modules(${PROJECT_NAME} Core)
# find_package (Qt5 REQUIRED COMPONENTS Core Gui Widgets Concurrent Svg PrintSupport Network)
# qt5_use_modules(${PROJECT_NAME} Core Gui Widgets Concurrent Svg PrintSupport Network)

# for multi-threading on Apple
if (APPLE AND ${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
  set(CMAKE_THREAD_LIBS_INIT "-lpthread")
  set(CMAKE_HAVE_THREADS_LIBRARY 1)
  set(CMAKE_USE_WIN32_THREADS_INIT 0)
  set(CMAKE_USE_PTHREADS_INIT 1)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
endif (APPLE AND ${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")

# Download submodules
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  option(GIT_SUBMODULE "Check submodules during build" ON)
  if(GIT_SUBMODULE)
    message(STATUS "Submodule update")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}")
    endif()
  endif()
endif()


############################
# Set includes and libraries

target_include_directories (${PROJECT_NAME} PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/boost_math/include
)

if(WIN32)
  # build BLAS/LAPACK
  enable_language(C)
  execute_process(
    COMMAND mkdir -p BUILD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/lapack
    RESULT_VARIABLE LAPACK_SUBMOD_RESULT)
  if(NOT LAPACK_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "Failed to make lapack BUILD directory with ${LAPACK_SUBMOD_RESULT}")
  endif()
  execute_process(
    COMMAND cmake -G "MinGW Makefiles" -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DBUILD_SHARED_LIBS=ON -DGNUtoMS=ON -DCMAKE_Fortran_FLAGS=-O0 -DBUILD_INDEX64=ON ..
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/lapack/BUILD
    RESULT_VARIABLE LAPACK_SUBMOD_RESULT)
  if(NOT LAPACK_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "LAPACK cmake failed with ${LAPACK_SUBMOD_RESULT}")
  endif()
  execute_process(
    COMMAND make -j -s
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/lapack/BUILD
    RESULT_VARIABLE LAPACK_SUBMOD_RESULT)
  if(NOT LAPACK_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "LAPACK make failed with ${LAPACK_SUBMOD_RESULT}")
  endif()
  # copy into dll, lib64, and dist
  execute_process(
    COMMAND cp bin/libblas64.dll ${CMAKE_CURRENT_SOURCE_DIR}/dll/libblas64.dll
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/lapack/BUILD
    RESULT_VARIABLE LAPACK_SUBMOD_RESULT)
  execute_process(
    COMMAND cp bin/libblas64.dll ${CMAKE_CURRENT_SOURCE_DIR}/dist/libblas64.dll
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/lapack/BUILD
    RESULT_VARIABLE LAPACK_SUBMOD_RESULT)
  execute_process(
    COMMAND cp lib/libblas64.dll.a ${CMAKE_CURRENT_SOURCE_DIR}/lib64/libblas64.dll.a
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/lapack/BUILD
    RESULT_VARIABLE LAPACK_SUBMOD_RESULT)
  execute_process(
    COMMAND cp bin/liblapack64.dll ${CMAKE_CURRENT_SOURCE_DIR}/dll/liblapack64.dll
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/lapack/BUILD
    RESULT_VARIABLE LAPACK_SUBMOD_RESULT)
  execute_process(
    COMMAND cp bin/liblapack64.dll ${CMAKE_CURRENT_SOURCE_DIR}/dist/liblapack64.dll
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/lapack/BUILD
    RESULT_VARIABLE LAPACK_SUBMOD_RESULT)
  execute_process(
    COMMAND cp lib/liblapack64.dll.a ${CMAKE_CURRENT_SOURCE_DIR}/lib64/liblapack64.dll.a
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/lapack/BUILD
    RESULT_VARIABLE LAPACK_SUBMOD_RESULT)

  # CBayes dependencies
  find_library (BLAS_LIB "blas64" PATHS ${PROJECT_SOURCE_DIR}/lib64)
  if (NOT BLAS_LIB)
    message (FATAL_ERROR
      "BLAS library missing from ${PROJECT_SOURCE_DIR}/lib64"
    )
  endif ()
  find_library (LAPACK_LIB "lapack64" PATHS ${PROJECT_SOURCE_DIR}/lib64)
  if (NOT LAPACK_LIB)
    message (FATAL_ERROR
      "LAPACK library missing from ${PROJECT_SOURCE_DIR}/lib64"
    )
  endif ()
endif(WIN32)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_link_libraries (${PROJECT_NAME}
  ${F2CLIB}
  ${BLAS_LIB}
  ${LAPACK_LIB}
)

# target_compile_definitions (${PROJECT_NAME} PUBLIC
#   ${ELEMEM_DEFINES}
# )


########################################################
# Update build date-stamp in About window on each build.

# add_custom_command (TARGET ${PROJECT_NAME} PRE_BUILD
#   COMMAND ${CMAKE_COMMAND} -E touch ${PROJECT_SOURCE_DIR}/src/About.cpp
# )

##################################################
# Add shared library dependencies to dist/ folder.

# TODO set up shared library dependency walk for libraries differently...
#if (WIN32)
#  get_target_property (QT5_QMAKE_EXECUTABLE Qt5::qmake IMPORTED_LOCATION)
#  get_filename_component (QT5_WINDEPLOYQT_EXECUTABLE
#    ${QT5_QMAKE_EXECUTABLE} PATH
#  )
#  set (QT5_WINDEPLOYQT_EXECUTABLE
#    "${QT5_WINDEPLOYQT_EXECUTABLE}/windeployqt.exe"
#  )
#  if (IS_RELEASE)
#    add_custom_command (TARGET ${PROJECT_NAME} POST_BUILD
#      COMMAND ${QT5_WINDEPLOYQT_EXECUTABLE} --release $<TARGET_FILE:${PROJECT_NAME}>
#    )
#  else (IS_RELEASE)
#    add_custom_command (TARGET ${PROJECT_NAME} POST_BUILD
#      COMMAND ${QT5_WINDEPLOYQT_EXECUTABLE} --debug $<TARGET_FILE:${PROJECT_NAME}>
#    )
#  endif (IS_RELEASE)
#endif (WIN32)

