NVCCDIR = /Developer/NVIDIA/CUDA-5.0
NVCC = $(NVCCDIR)/bin/nvcc
NVCCLIBS = -L $(NVCCDIR)/lib -lcudart
NVCCFLAGS = -gencode arch=compute_20,code=sm_20
NVCCFLAGS += -gencode arch=compute_20,code=compute_20
NVCCFLAGS += -gencode arch=compute_20,code=sm_21
NVCCFLAGS += -gencode arch=compute_30,code=sm_30
NVCCFLAGS += -gencode arch=compute_35,code=sm_35
NVCCFLAGS += --ptxas-options=-v -m 64
NVCCOPT = -O3
NVCCPROF = -G -pg

CXXFLAGS = -v -Wall -Wextra

cuda: pairwise.cu
	$(NVCC) -o pairwise $(NVCCFLAGS) $(NVCCOPT) pairwise.cu $(NVCCLIBS)

debug:
	$(NVCC) -o pairwise $(NVCCFLAGS) -g -G pairwise.cu $(NVCCLIBS)

profile:
	$(NVCC) -o pairwise $(NVCCFLAGS) $(NVCCPROF) $(NVCCOPT) pairwise.cu $(NVCCLIBS)

matmul: pairwise.cu
	$(NVCC) -o pairwise-matmul $(NVCCFLAGS) $(NVCCOPT) -DMATMUL pairwise.cu $(NVCCLIBS)

all: cuda unsliced sliced general

unsliced: pairwise-unsliced.cu
	$(NVCC) -o pairwise-unsliced $(NVCCFLAGS) $(NVCCOPT) pairwise-unsliced.cu $(NVCCLIBS)

sliced: pairwise-sliced.cu
	$(NVCC) -o pairwise-sliced $(NVCCFLAGS) $(NVCCOPT) pairwise-sliced.cu $(NVCCLIBS)

general: pairwise-general.cu
	$(NVCC) -o pairwise-general $(NVCCFLAGS) $(NVCCOPT) pairwise-general.cu $(NVCCLIBS)

.PHONY: clean

clean:
	rm -rf pairwise pairwise-general pairwise-unsliced pairwise-sliced
