NVCC = nvcc
NVCCFLAGS = -gencode arch=compute_20,code=sm_20
NVCCFLAGS += -gencode arch=compute_30,code=sm_30
NVCCFLAGS += --ptxas-options=-v
NVCCOPT = -O3
NVCCPROF = -G -pg

cuda: pairwise.cu
	$(NVCC) -o pairwise $(NVCCFLAGS) $(NVCCOPT) pairwise.cu

debug:
	$(NVCC) -o pairwise $(NVCCFLAGS) -g -G pairwise.cu

profile:
	$(NVCC) -o pairwise $(NVCCFLAGS) $(NVCCPROF) $(NVCCOPT) pairwise.cu

matmul: pairwise.cu
	$(NVCC) -o pairwise-matmul $(NVCCFLAGS) $(NVCCOPT) -DMATMUL pairwise.cu

all: cuda unsliced sliced general

unsliced: pairwise-unsliced.cu
	$(NVCC) -o pairwise-unsliced $(NVCCFLAGS) $(NVCCOPT) pairwise-unsliced.cu

sliced: pairwise-sliced.cu
	$(NVCC) -o pairwise-sliced $(NVCCFLAGS) $(NVCCOPT) pairwise-sliced.cu

general: pairwise-general.cu
	$(NVCC) -o pairwise-general $(NVCCFLAGS) $(NVCCOPT) pairwise-general.cu

clean:
	rm -rf pairwise pairwise-general pairwise-unsliced pairwise-sliced
