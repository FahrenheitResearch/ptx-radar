cmake_minimum_required(VERSION 3.24)
project(ptx-radar LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Blackwell RTX 5090 is sm_120. Also build sm_86 (Ampere) and sm_89 (Ada)
# real SASS, plus a compute_120 PTX fallback for newer cards.
set(CMAKE_CUDA_ARCHITECTURES "86;89;120;120-virtual"
    CACHE STRING "CUDA archs to compile real SASS + PTX fallback" FORCE)

include(FetchContent)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.8-docking
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(imgui)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui PUBLIC glfw)

find_package(BZip2 QUIET)
if(NOT BZip2_FOUND)
    FetchContent_Declare(
        bzip2_src
        URL      https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz
        URL_HASH SHA256=ab5a03176ee106d3f0fa90e381da478ddae405918153cca248e682cd0c4a2269
    )
    FetchContent_MakeAvailable(bzip2_src)
    add_library(bzip2 STATIC
        ${bzip2_src_SOURCE_DIR}/blocksort.c
        ${bzip2_src_SOURCE_DIR}/huffman.c
        ${bzip2_src_SOURCE_DIR}/crctable.c
        ${bzip2_src_SOURCE_DIR}/randtable.c
        ${bzip2_src_SOURCE_DIR}/compress.c
        ${bzip2_src_SOURCE_DIR}/decompress.c
        ${bzip2_src_SOURCE_DIR}/bzlib.c
    )
    target_include_directories(bzip2 PUBLIC ${bzip2_src_SOURCE_DIR})
    add_library(BZip2::BZip2 ALIAS bzip2)
endif()

find_package(CUDAToolkit REQUIRED)

# Embed src/cuda/ultra_kernels.ptx into the binary as a C++ string literal
# (configure-time substitution into ultra_kernels_ptx.h). The file is loaded
# at runtime via the CUDA Driver API and JIT'd by the driver to native SASS.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/ultra_kernels.ptx"
     ULTRA_PTX_CONTENT)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/ultra_kernels_ptx.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/generated/ultra_kernels_ptx.h"
               @ONLY)
# Force CMake to re-run when the .ptx file changes so the embedded copy
# stays in sync.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/ultra_kernels.ptx")

add_executable(ptx-radar
    src/main.cpp
    src/app.cpp
    src/historic.cpp
    src/nexrad/level2_parser.cpp
    src/net/downloader.cpp
    src/net/polling_links.cpp
    src/net/warnings.cpp
    src/render/basemap.cpp
    src/render/color_table.cpp
    src/render/gl_interop.cpp
    src/render/gl_texture.cpp
    src/ui/ui.cpp
    src/ui/workstation.cpp
    src/cuda/renderer.cu
    src/cuda/preprocess.cu
    src/cuda/gpu_tensor.cu
    src/cuda/gpu_detection.cu
    src/cuda/gpu_pipeline.cu
    src/cuda/volume3d.cu
)

target_include_directories(ptx-radar PRIVATE
    src
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

if(WIN32)
    target_compile_definitions(ptx-radar PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

target_link_libraries(ptx-radar PRIVATE
    imgui
    glfw
    BZip2::BZip2
    nlohmann_json::nlohmann_json
    CUDA::cudart
    CUDA::cuda_driver
)

if(WIN32)
    target_link_libraries(ptx-radar PRIVATE opengl32 winhttp psapi ole32 windowscodecs)
else()
    find_package(CURL REQUIRED)
    target_link_libraries(ptx-radar PRIVATE GL CURL::libcurl dl pthread)
endif()

set_target_properties(ptx-radar PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(ptx-radar PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --expt-relaxed-constexpr
        -Xptxas=-v
    >
)
