###############################################################################
# Compiler and base flags
###############################################################################
CXX      := g++
CXXFLAGS := -std=c++17 

###############################################################################
# Determine OS and set OS-specific variables
###############################################################################
ifeq ($(OS),Windows_NT)
    OS_NAME := Windows
    RM      := del /Q
    MKDIR   := mkdir
    EXE     := Thrawn.exe
    STACK_FLAG := -Wl,--stack,8388608
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        OS_NAME := Linux
        RM      := rm -f
        MKDIR   := mkdir -p
    else ifeq ($(UNAME_S),Darwin)
        OS_NAME := macOS
        RM      := rm -f
        MKDIR   := mkdir -p
    endif
    EXE := Thrawn
    STACK_FLAG := # No stack flag for Linux/macOS
endif

###############################################################################
# Source and object files
###############################################################################
SOURCES := $(wildcard *.cpp)
OBJECTS := $(SOURCES:.cpp=.o)

###############################################################################
# Build type and flags
###############################################################################
BUILD ?= release

OPT_FLAGS  := -O3 -flto -march=native
ARCH_FLAGS :=

ifeq ($(OS_NAME),Linux)
    ARCH_DETECTED := $(shell uname -m)
    ifeq ($(ARCH_DETECTED),x86_64)
        ARCH_FLAGS += -mavx2 -msse4.1 -msse3 -msse2 -mmmx -msse \
                      -DUSE_AVX2 -DUSE_SSE41 -DUSE_SSE3 -DUSE_SSE2 -DUSE_SSE
    else ifeq ($(ARCH_DETECTED),aarch64)
        ARCH_FLAGS += -DARM64 -DUSE_NEON -DUSE_NEON_DOTPROD -flax-vector-conversions
    endif
else ifeq ($(OS_NAME),macOS)
    ARCH_FLAGS += -mmacosx-version-min=10.15
    ARCH_DETECTED := $(shell uname -m)
    ifeq ($(ARCH_DETECTED),arm64)
        ARCH_FLAGS += -DARM64 -DUSE_NEON -DUSE_NEON_DOTPROD -flax-vector-conversions
    else
        ARCH_FLAGS += -mavx2 -msse4.1 -msse3 -msse2 -mmmx -msse \
                      -DUSE_AVX2 -DUSE_SSE41 -DUSE_SSE3 -DUSE_SSE2 -DUSE_SSE
    endif
else ifeq ($(OS_NAME),Windows)
    ARCH_DETECTED := $(PROCESSOR_ARCHITECTURE)
    ARCH_FLAGS    += -DWIN32
    ifeq ($(ARCH_DETECTED),AMD64)
        ARCH_FLAGS += -mavx2 -msse4.1 -msse3 -msse2 -mmmx -msse \
                      -DUSE_AVX2 -DUSE_SSE41 -DUSE_SSE3 -DUSE_SSE2 -DUSE_SSE
    else ifeq ($(ARCH_DETECTED),ARM64)
        ARCH_FLAGS += -DARM64 -DUSE_NEON -DUSE_NEON_DOTPROD -flax-vector-conversions
    endif
endif

ifeq ($(BUILD),debug)
    CXXFLAGS += -g $(ARCH_FLAGS) -DDEBUG_BUILD -DDLL_EXPORT
else ifeq ($(BUILD),release)
    CXXFLAGS += $(OPT_FLAGS) $(ARCH_FLAGS) -DRELEASE_BUILD -DDLL_EXPORT
endif

###############################################################################
# Targets
###############################################################################
.PHONY: all release debug clean

all:
	$(MAKE) BUILD=$(BUILD) clean $(EXE)

release:
	$(MAKE) BUILD=release clean $(EXE)

debug:
	$(MAKE) BUILD=debug clean $(EXE)

$(EXE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(STACK_FLAG) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	$(RM) $(OBJECTS) $(EXE)
