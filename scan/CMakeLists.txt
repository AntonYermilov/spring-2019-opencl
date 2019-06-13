set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework OpenCL -O3 -Wall -Wextra -pedantic -DLOCAL -DCL_SILENCE_DEPRECATION")

add_executable(scan scan.cpp)
add_executable(generator generator.cpp)
add_executable(checker checker.cpp)
