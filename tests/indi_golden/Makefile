# Makefile for INDI golden-reference generator
#
# Compiles gen_golden.c (self-contained stubs + INDI math) together with
# the ActiveSetCtlAlloc library sources from the indiflight tree.
# Produces a native x86-64 binary that writes CSV to stdout.

INDIFLIGHT := /home/noidvan/workspaces/indiflight

# ActiveSetCtlAlloc paths
AS_ROOT  := $(INDIFLIGHT)/lib/main/ActiveSetCtlAlloc/src
AS_LIB   := $(AS_ROOT)/lib
AS_QR    := $(AS_LIB)/qr_solve

# WLS library sources (all that the QR-based solver needs)
AS_SRCS := \
	$(AS_ROOT)/common/solveActiveSet.c \
	$(AS_ROOT)/common/setupWLS.c \
	$(AS_ROOT)/solveActiveSet_qr.c \
	$(AS_ROOT)/solveActiveSet_qr_naive.c \
	$(AS_ROOT)/solveActiveSet_chol.c \
	$(AS_LIB)/qr_wrapper.c \
	$(AS_LIB)/qr_updates.c \
	$(AS_LIB)/sparse_math.c \
	$(AS_LIB)/chol_math.c \
	$(AS_QR)/qr_solve.c \
	$(AS_QR)/r8lib_min.c

# Include paths: the solver headers live in several places
INCLUDES := \
	-I$(AS_ROOT)/common \
	-I$(AS_LIB) \
	-I$(AS_QR)

# Preprocessor defines needed by the WLS library
DEFINES := \
	-DAS_N_U=4 \
	-DAS_N_V=6 \
	-DAS_SINGLE_FLOAT

CC      := gcc
CFLAGS  := -std=gnu11 -O2 -Wall -Wextra -Wno-unused-parameter \
           -Wno-sign-compare -Wno-unused-variable -Wno-missing-field-initializers \
           $(INCLUDES) $(DEFINES)
LDFLAGS := -lm

TARGET  := gen_golden
CSV     := golden.csv

.PHONY: all clean csv

all: $(TARGET)

$(TARGET): gen_golden.c $(AS_SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

csv: $(TARGET)
	./$(TARGET) > $(CSV)
	@echo "Wrote $(CSV) ($$(wc -l < $(CSV)) lines)"

clean:
	rm -f $(TARGET) $(CSV)
