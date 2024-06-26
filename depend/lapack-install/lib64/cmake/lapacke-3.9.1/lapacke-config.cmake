# Compute locations from <prefix>/@{LIBRARY_DIR@/cmake/lapacke-<v>/<self>.cmake
get_filename_component(_LAPACKE_SELF_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(_LAPACKE_PREFIX "${_LAPACKE_SELF_DIR}" PATH)
get_filename_component(_LAPACKE_PREFIX "${_LAPACKE_PREFIX}" PATH)
get_filename_component(_LAPACKE_PREFIX "${_LAPACKE_PREFIX}" PATH)

# Load the LAPACK package with which we were built.
set(LAPACK_DIR "${_LAPACKE_PREFIX}/lib64/cmake/-3.9.1")
find_package(LAPACK NO_MODULE)

# Load lapacke targets from the install tree.
if(NOT TARGET lapacke)
  include(${_LAPACKE_SELF_DIR}/lapacke-targets.cmake)
endif()

# Hint for project building against lapack
set(LAPACKE_Fortran_COMPILER_ID ${LAPACK_Fortran_COMPILER_ID})

# Report lapacke header search locations.
set(LAPACKE_INCLUDE_DIRS ${_LAPACKE_PREFIX}/include)

# Report lapacke libraries.
set(LAPACKE_LIBRARIES lapacke ${LAPACK_LIBRARIES})

unset(_LAPACKE_PREFIX)
unset(_LAPACKE_SELF_DIR)
