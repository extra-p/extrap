cmake_minimum_required (VERSION 3.15)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
# Add dependencies
include(FetchContent)
FetchContent_Declare(
  kaitai_struct_cpp_stl_runtime
  GIT_REPOSITORY https://github.com/kaitai-io/kaitai_struct_cpp_stl_runtime.git
  GIT_TAG        0.9
  CMAKE_ARGS -U STRING_ENCODING_TYPE -D STRING_ENCODING_TYPE="NONE"
)

FetchContent_Declare(
        protobuf
        GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
        GIT_TAG        v3.14.0
        SOURCE_SUBDIR  cmake
)

FetchContent_Declare(sqlite3 URL "https://www.sqlite.org/2022/sqlite-amalgamation-3380100.zip")
FetchContent_Declare(
        SQLiteCpp
        GIT_REPOSITORY https://github.com/SRombauts/SQLiteCpp.git
        GIT_TAG        3.1.1
)
set(protobuf_BUILD_TESTS OFF)
set(STRING_ENCODING_TYPE "NONE")
set(SQLITECPP_RUN_CPPLINT OFF)

FetchContent_MakeAvailable(kaitai_struct_cpp_stl_runtime protobuf sqlite3 SQLiteCpp)

# Load generated protobuf classes
file(GLOB_RECURSE PROTOBUF_FILES  "proto_files/*.cc" "proto_files/*.h" )

# add_executable
add_executable (AggregateNCUReport "AggregateNCUReport.cpp" "nsight_cuprof_report.cpp" ${PROTOBUF_FILES} "${sqlite3_SOURCE_DIR}/sqlite3.c" "${kaitai_struct_cpp_stl_runtime_SOURCE_DIR}/kaitai/kaitaistream.cpp" "AggregateNCUReport.h" "nsight_cuprof_report.h" "file_header.h" "block_header.h" "profile_result.h" "profile_source.h")
set_property(TARGET AggregateNCUReport PROPERTY CXX_STANDARD 17)
target_compile_definitions(AggregateNCUReport PUBLIC KS_STR_ENCODING_NONE)
target_compile_definitions(AggregateNCUReport PUBLIC SQLITE_ENABLE_COLUMN_METADATA)
target_link_libraries(AggregateNCUReport PUBLIC libprotobuf)
target_link_libraries(AggregateNCUReport PUBLIC SQLiteCpp)
target_include_directories(AggregateNCUReport PUBLIC ${kaitai_struct_cpp_stl_runtime_SOURCE_DIR} )
target_include_directories(AggregateNCUReport PUBLIC "${protobuf_SOURCE_DIR}/src/")