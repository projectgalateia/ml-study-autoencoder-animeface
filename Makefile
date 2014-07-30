all:
	$(CUDA_ROOT)/bin/nvcc -g -lcublas -lcuda -arch=compute_20 main.cu -o run
	./run

