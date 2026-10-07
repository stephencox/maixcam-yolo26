set shell := ["bash", "-euo", "pipefail", "-c"]

cxx := env_var_or_default("CXX", "g++")
cxxflags := "-std=c++17 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined -fno-omit-frame-pointer -Imain/include -Itest"

# Host tests for the pure headers
test:
    mkdir -p test/bin
    for t in test/test_*.cpp; do \
        n=$(basename "$t" .cpp); \
        {{cxx}} {{cxxflags}} "$t" -o "test/bin/$n" && "test/bin/$n"; \
    done
