PROJECT( CMatrix )

OPTION( USE_HDF5 "Build with HDF5 support." )

SET( CMAKE_CXX_FLAGS    "-fPIC" )
SET( CMAKE_C_FLAGS      "-fPIC" )

SET( MLPROJECTS_BASE_DIR /Users/rdehaan/projects/elemem/src )
SET( GPC_SUBDIR GPc )

if(USE_HDF5)
    ADD_DEFINITIONS(-D_HDF5)
    FIND_PACKAGE( HDF5 )
endif(USE_HDF5)

INCLUDE_DIRECTORIES( . ${HDF5_INCLUDE_DIR}
                       ${MLPROJECTS_BASE_DIR}/ndlutil/src
                       ${MLPROJECTS_BASE_DIR}/matrix/src
                       ${MLPROJECTS_BASE_DIR}/branches/helper/src
                       /usr/include
                       /usr/local/include        #for boost
                       /usr/local/include/eigen3 #for eigen3
                    )

LINK_DIRECTORIES( /usr/lib /usr/local/lib  /usr/local/lib/gcc/7/ ${HDF5_LIBRARY_DIRS} )

SET( CMatrix_sources
    ${MLPROJECTS_BASE_DIR}/${GPC_SUBDIR}/CNdlInterfaces.h
    ${MLPROJECTS_BASE_DIR}/${GPC_SUBDIR}/CMatrix.h                  ${MLPROJECTS_BASE_DIR}/${GPC_SUBDIR}/CMatrix.cpp

    ${MLPROJECTS_BASE_DIR}/${GPC_SUBDIR}/ndlassert.h               ${MLPROJECTS_BASE_DIR}/${GPC_SUBDIR}/ndlassert.cpp
    ${MLPROJECTS_BASE_DIR}/${GPC_SUBDIR}/ndlexceptions.h

    ${MLPROJECTS_BASE_DIR}/${GPC_SUBDIR}/lapack.h
    #${MLPROJECTS_BASE_DIR}/${GPC_SUBDIR}/f2c.h
    ${MLPROJECTS_BASE_DIR}/${GPC_SUBDIR}/ndlfortran.h              ${MLPROJECTS_BASE_DIR}/${GPC_SUBDIR}/ndlfortran.c
    ${MLPROJECTS_BASE_DIR}/${GPC_SUBDIR}/ndlstrutil.h              ${MLPROJECTS_BASE_DIR}/${GPC_SUBDIR}/ndlstrutil.cpp
    ${MLPROJECTS_BASE_DIR}/${GPC_SUBDIR}/ndlutil.h                 ${MLPROJECTS_BASE_DIR}/${GPC_SUBDIR}/ndlutil.cpp
    ${MLPROJECTS_BASE_DIR}/${GPC_SUBDIR}/ndlstrutil.h              ${MLPROJECTS_BASE_DIR}/${GPC_SUBDIR}/ndlstrutil.cpp
)
 
ADD_DEFINITIONS( -D_FORTRAN_MAIN_FIX )
ADD_LIBRARY( cmatrix STATIC ${CMatrix_sources} )
TARGET_LINK_LIBRARIES( cmatrix hdf5_hl hdf5 f2c m lapack blas )

#SUBDIRS( test )
