# environment
SM := 35

# compilers
CUDA_PATH ?= $(shell test -d /shared/centos7/cuda/9.0 && echo /shared/centos7/cuda/9.0)

ifeq ($(CUDA_PATH),)
CUDA_PATH := $(shell test -d /usr/local/cuda-7.0 && echo /usr/local/cuda-7.0)
endif 

GCC := g++
NVCC := nvcc #$(CUDA_PATH)/bin/nvcc

# libraries
CUDA_LIB_PATH := $(CUDA_PATH)/lib

# Remove function
RM = rm -f
 
# Compiler flags:
# -g    debugging information
# -Wall turns on most compiler warnings
GENCODE_FLAGS := -gencode arch=compute_$(SM),code=sm_$(SM)
LIB_FLAGS := -lcudadevrt -lcudart
ifeq ($(OS), DARWIN)
	CCFLAGS := -stdlib=libstdc++
else
	CCFLAGS := -O3
endif
NVCCFLAGS :=
GccFLAGS = -fopenmp -O3 
MPIFLAGS = -Wno-deprecated

debug: GccFLAGS += -DDEBUG -g -Wall
debug: MPIFLAGS += -DDEBUG
debug: NVCCFLAGS += -g -G
debug: all

# The build target executable:
TARGET = sobel

all: build

build: $(TARGET)

$(TARGET): lib/dlink.o lib/main.o lib/alg/$(TARGET).o lib/img/imghandler.o lib/alg/locationhandler.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GENCODE_FLAGS) -link

lib/dlink.o: lib/alg/$(TARGET).o 
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GENCODE_FLAGS) -dlink

lib/main.o: lib/main.cpp lib/config.h
	$(GCC) $(GccFLAGS) -c $< -o $@

lib/alg/locationhandler.o: lib/alg/locationhandler.cpp lib/alg/locationhandler.h lib/config.h
	$(GCC) $(GccFLAGS) -c $< -o $@
	
lib/alg/$(TARGET).o: lib/alg/$(TARGET).cu lib/config.h
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS) 

lib/img/imghandler.o: lib/img/imghandler.cpp lib/config.h
	$(GCC) $(GccFLAGS) -c $< -o $@
	
clean:
	$(RM) $(TARGET) *.o lib/*/*.o lib/*.o *.tar* *.core*
