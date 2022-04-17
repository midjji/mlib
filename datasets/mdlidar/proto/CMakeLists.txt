project(mdlidarproto C CXX)

find_package(Threads REQUIRED)
INCLUDE(FindProtobuf)
FIND_PACKAGE(Protobuf REQUIRED)
PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HEADER mdlidar.proto)

ADD_LIBRARY(mdlidarproto ${PROTO_HEADER} ${PROTO_SRC})
target_compile_features(mdlidarproto PUBLIC cxx_std_17)
target_link_libraries(mdlidarproto ${PROTOBUF_LIBRARY})
target_link_libraries(mdlidarproto Threads::Threads)

message("protos in: ${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(mdlidarproto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
