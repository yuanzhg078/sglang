cmake_minimum_required(VERSION 3.18)
project(kvclient_python LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
if(NOT TARGET kv_client)
    set(KVCLIENT_SDK_ROOT "" CACHE PATH "KvClient SDK source or install directory")
    find_path(KVCLIENT_INCLUDE_DIR kv_client.h
        HINTS ${KVCLIENT_SDK_ROOT}/include ${KVCLIENT_SDK_ROOT}/kv_semantics/include REQUIRED)
    find_library(KVCLIENT_LIBRARY kv_client
        HINTS ${KVCLIENT_SDK_ROOT}/lib ${KVCLIENT_SDK_ROOT}/src
              ${KVCLIENT_SDK_ROOT}/build/src ${KVCLIENT_SDK_ROOT}/kv_semantics/src REQUIRED)
    add_library(kv_client SHARED IMPORTED)
    set_target_properties(kv_client PROPERTIES IMPORTED_LOCATION "${KVCLIENT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${KVCLIENT_INCLUDE_DIR}")
endif()
pybind11_add_module(_kvclient kvclient_pybind.cc)
target_compile_features(_kvclient PRIVATE cxx_std_17)
target_link_libraries(_kvclient PRIVATE kv_client)
set_target_properties(_kvclient PROPERTIES INSTALL_RPATH "$ORIGIN")
set(KVCLIENT_PYTHON_INSTALL_DIR "${Python_SITEARCH}" CACHE PATH "Python extension install directory")
install(TARGETS _kvclient LIBRARY DESTINATION "${KVCLIENT_PYTHON_INSTALL_DIR}")
