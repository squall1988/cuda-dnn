cc = nvcc
gcc = g++
libs =  --compiler-options "-std=c++0x" -L/usr/local/cuda/lib64/ -lcudart -lcublas 
cflags = -Wall --ptxas-options=-v

main: main.o cudamat_kernels.o matrix.o
	$(cc) $(libs)  $^ -o $@
main.o : main.cpp 
	$(cc) $(libs) -c $^
matrix.o: matrix.cu
	$(cc) -c $^
cudamat_kernels.o : cudamat_kernels.cu
	$(cc)  -c $^
clean:
	rm main *.o
