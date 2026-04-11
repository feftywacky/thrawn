###############################################################################
# Compiler and base flags
###############################################################################
CXX      := g++
CXXFLAGS := -std=c++17
CPPFLAGS :=
LDFLAGS  :=

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
MACOS_MIN_VERSION := 10.15

ifeq ($(OS_NAME),Linux)
    ARCH_DETECTED := $(shell uname -m)
    ifeq ($(ARCH_DETECTED),x86_64)
        CPPFLAGS += -DUSE_AVX2
        ARCH_FLAGS += -mavx2
    else ifeq ($(ARCH_DETECTED),aarch64)
        CPPFLAGS += -DUSE_NEON
    endif
else ifeq ($(OS_NAME),macOS)
    ARCH_DETECTED := $(shell uname -m)
    ifeq ($(ARCH_DETECTED),arm64)
        MACOS_MIN_VERSION := 11.0
        CPPFLAGS += -DUSE_NEON
    else
        CPPFLAGS += -DUSE_AVX2
        ARCH_FLAGS += -mavx2
    endif
    ARCH_FLAGS += -mmacosx-version-min=$(MACOS_MIN_VERSION)
else ifeq ($(OS_NAME),Windows)
    ARCH_DETECTED := $(PROCESSOR_ARCHITECTURE)
    CPPFLAGS += -DWIN32
    ifeq ($(ARCH_DETECTED),AMD64)
        CPPFLAGS += -DUSE_AVX2
        ARCH_FLAGS += -mavx2
    else ifeq ($(ARCH_DETECTED),ARM64)
        CPPFLAGS += -DUSE_NEON
    endif
endif

ifeq ($(BUILD),debug)
    CPPFLAGS += -DDEBUG_BUILD
    CXXFLAGS += -g $(ARCH_FLAGS)
else ifeq ($(BUILD),release)
    CPPFLAGS += -DRELEASE_BUILD
    CXXFLAGS += $(OPT_FLAGS) $(ARCH_FLAGS)
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
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(STACK_FLAG) -o $@ $^

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

clean:
	$(RM) $(OBJECTS) $(EXE)
