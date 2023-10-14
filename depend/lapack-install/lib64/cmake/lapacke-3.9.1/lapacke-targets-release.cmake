#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "lapacke" for configuration "Release"
set_property(TARGET lapacke APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(lapacke PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/liblapacke.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS lapacke )
list(APPEND _IMPORT_CHECK_FILES_FOR_lapacke "${_IMPORT_PREFIX}/lib64/liblapacke.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
