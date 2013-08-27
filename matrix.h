#include <cublas.h>
#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <functional>
#include <algorithm>
#include "cudamat_kernels.cuh"


#define ERROR_TRANSPOSEDNESS -7
#define ERROR_NOT_ON_DEVICE -8
#define ERROR_UNSUPPORTED -9
#define ERROR_INCOMPATIBLE_DIMENSIONS -1
#define CUBLAS_ERROR -2
#define CUDA_ERROR -3
#define VIEW_ERROR -4
#define ERROR_TRANSPOSED -5
#define ERROR_GENERIC -6
#define ERROR_TRANSPOSEDNESS -7
#define ERROR_NOT_ON_DEVICE -8
#define ERROR_UNSUPPORTED -9
typedef struct Matirx {
	float* data_host;
	float* data_device;
	int on_device;
	int on_host;
	int size[2];
	int is_trans; // 0 or 1
	int owns_data;
} Matrix;
inline bool check_cublas_error();
inline bool checkCUDAError();
extern const char* get_last_cuda_error();

extern int cublas_init();
extern int cuda_set_device(int deviceId);
extern int get_leading_dimension(Matrix* mat);
extern int get_nonleading_dimension(Matrix* mat);
extern void set_transpose(Matrix* mat, int is_trans);
extern int copy_to_host(Matrix* mat);
extern int copy_to_device(Matrix* mat);
extern int copy_on_device(Matrix* mat1, Matrix* mat2) ;
extern int free_device_memory(Matrix* mat);
extern int dot(Matrix* mat1, Matrix* mat2, Matrix* target, float beta, float alpha);
extern int apply_sigmoid(Matrix* mat, Matrix* target);
extern int add_row_vec(Matrix* mat, Matrix* vec, Matrix* target) ;
extern void init_from_array(Matrix* mat, float* data, int m, int n) ;
extern int apply_log(Matrix* mat, Matrix* target);
extern int add_mult(Matrix* mat1, Matrix* mat2, float alpha);
extern int mult_elementwise(Matrix* mat1, Matrix* mat2, Matrix* target);
extern int get_row_slice(Matrix* source, Matrix* target, unsigned int start, unsigned int end);
extern void subself_mult_elementwise(Matrix* mat1, Matrix* target);
extern int init_empty(Matrix* mat, int m, int n);
extern int copy_transpose(Matrix* source, Matrix* target);
extern void init_zeros(Matrix *mat, int row, int col);
extern int mult_by_scalar(Matrix *mat, float alpha, Matrix* target);
extern int sub_mult(Matrix *mat, Matrix *target);
extern int add_elementwise(Matrix* mat1, Matrix* mat2, Matrix* target) ;
extern int divide_by_scalar(Matrix* mat, float alpha, Matrix* target);