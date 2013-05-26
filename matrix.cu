#include "matrix.h"
/* ------------------------------ CUBLAS init/shutdown ------------------------------ */

inline bool check_cublas_error() {
    cublasStatus status = cublasGetError();

    return status != CUBLAS_STATUS_SUCCESS;
}

inline bool checkCUDAError() {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
        printf("%s\n", cudaGetErrorString( err));
    return cudaSuccess != err;
}

 extern const char* get_last_cuda_error() {
    cudaError_t err = cudaGetLastError();

    return cudaGetErrorString( err);
}

 extern int cublas_init() {
    cublasInit();
    if (check_cublas_error())
        return CUBLAS_ERROR;
    else
        return 0;
}

 extern int cublas_shutdown() {
    cublasShutdown();
    if (check_cublas_error())
        return CUBLAS_ERROR;
    else
        return 0;
}


 extern int cuda_set_device(int deviceId) {
    cudaSetDevice(deviceId);
    
    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}


/* ------------------------------ Utility routines ------------------------------ */

 extern int get_leading_dimension(Matrix* mat) {
    return mat->is_trans ? mat->size[1] : mat->size[0];
}

 extern int get_nonleading_dimension(Matrix* mat) {
    return mat->is_trans ? mat->size[0] : mat->size[1];
}

 extern void set_transpose(Matrix* mat, int is_trans) {
    mat->is_trans = is_trans;
}

inline char get_transpose_char(Matrix* mat) {
    return mat->is_trans ? 't' : 'n';
}

 extern void cuda_sync_threads() {
    cudaThreadSynchronize();
}

/* ------------------------------ Allocating/moving data ------------------------------ */

 extern int allocate_device_memory(Matrix* mat) {
    int len = mat->size[0]*mat->size[1];

    cublasAlloc(len, sizeof(mat->data_device[0]), (void**)&mat->data_device);

    if (check_cublas_error())
        return CUBLAS_ERROR;
    else
        return 0;
}

 extern int copy_to_host(Matrix* mat) {
    int len = mat->size[0]*mat->size[1];

    if (mat->on_device) {
            cublasGetVector(len, sizeof(mat->data_host[0]), mat->data_device, 1, mat->data_host, 1);

        if (check_cublas_error())
            return CUBLAS_ERROR;
    } else
       return ERROR_NOT_ON_DEVICE;
 
    return 0;
}

 extern int copy_to_device(Matrix* mat) {
    int len = mat->size[0]*mat->size[1];
    int err_code = 0;

    if (!mat->owns_data)
        return VIEW_ERROR;

    if (!mat->on_device) {
        err_code = allocate_device_memory(mat);
        if (err_code)
            return err_code;
    }

    cublasSetVector(len, sizeof(mat->data_host[0]), mat->data_host, 1, mat->data_device, 1);
    
    if (check_cublas_error())
        return CUBLAS_ERROR;
    else {
        mat->on_device = 1;
        return 0;
    }
}

 extern int copy_on_device(Matrix* mat1, Matrix* mat2) {
    int len = mat1->size[0]*mat1->size[1];

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    cublasScopy(len, mat1->data_device, 1, mat2->data_device, 1);

    if (check_cublas_error())
        return CUBLAS_ERROR;
    else
        return 0;
}

 extern int get_row_slice(Matrix* source, Matrix* target, unsigned int start, unsigned int end) {
    int height = source->size[0];
    int width = source->size[1];

    if ((end - start) != target->size[0] || source->size[1] != target->size[1] || start >= end || end > height)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    dim3 kernelBlockGrid((int)ceil((end - start)/32.), (int)ceil(width/32.), 1);
    dim3 kernelBlockDim(32, 1, 1);

    kGetRowSlice<<<kernelBlockGrid,kernelBlockDim>>>(source->data_device, target->data_device, start, end, width, height);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}



 extern int copy_transpose(Matrix* source, Matrix* target) {
    unsigned int height = source->size[0];
    unsigned int width = source->size[1];

    if (source->size[0] != target->size[1] || source->size[1] != target->size[0])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    // setup execution parameters
    unsigned int grid_x = height / COPY_BLOCK_SIZE;
    if (height % COPY_BLOCK_SIZE)
        grid_x++;

    unsigned int grid_y = width / COPY_BLOCK_SIZE;
    if (width % COPY_BLOCK_SIZE)
        grid_y++;

    dim3 grid(grid_x, grid_y, 1);
    dim3 threads(COPY_BLOCK_SIZE, COPY_BLOCK_SIZE, 1);

    kTranspose<<< grid, threads >>>(target->data_device, source->data_device, height, width);

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

 extern int free_device_memory(Matrix* mat) {
    if (mat->owns_data && mat->on_device) {
        cublasFree(mat->data_device);
        mat->on_device = 0;

        if (check_cublas_error())
            return CUBLAS_ERROR;
    }

    return 0;
}

 extern int reshape(Matrix* mat, unsigned int m, unsigned int n) {
    if (mat->size[0] * mat->size[1] != m * n)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat->size[0] = m;
    mat->size[1] = n;

    return 0;
}

 extern int get_slice(Matrix* source, Matrix* target, unsigned int first_col, unsigned int last_col) {
    if (source->is_trans)
        return ERROR_TRANSPOSED;

    if (!source->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (last_col > source->size[1] || (first_col >= last_col))
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    int num_rows = source->size[0];

    target->data_host = 0;
    target->data_device = source->data_device + first_col * num_rows;
    target->on_device = 1;
    target->on_host = 0;
    target->size[0] = source->size[0];
    target->size[1] = last_col - first_col;
    target->is_trans = 0;
    target->owns_data = 0;

    return 0;
}

 extern int get_vector_slice(Matrix* source, Matrix* target, unsigned int first_ind, unsigned int last_ind) {
    // source must be a vector
    if (source->size[0] > 1 && source->size[1] > 1)
        return ERROR_GENERIC;

    if (source->is_trans)
        return ERROR_TRANSPOSED;

    if (!source->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (first_ind >= last_ind)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    int num_rows = source->size[0];

    target->data_host = 0;
    target->data_device = source->data_device + first_ind * num_rows;
    target->on_device = 1;
    target->on_host = 0;
    target->is_trans = 0;
    target->owns_data = 0;

    if (source->size[0] > 1) {
        if (last_ind > source->size[0])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        target->size[0] = last_ind - first_ind;
        target->size[1] = 1;
    } else {
        if (last_ind > source->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        target->size[0] = 1;
        target->size[1] = last_ind - first_ind;
    }

    return 0;
}

/* ------------------------------ Initialization routines ------------------------------ */

 extern void init_from_array(Matrix* mat, float* data, int m, int n) {
    mat->data_host = data;
    mat->size[0] = m;
    mat->size[1] = n;
    mat->on_device = 0;
    mat->on_host = 1;
    mat->is_trans = 0;
    mat->owns_data = 1;
}

 extern int init_empty(Matrix* mat, int m, int n) {
    mat->size[0] = m;
    mat->size[1] = n;
    mat->on_device = 1;
    mat->on_host = 0;
    mat->is_trans = 0;
    mat->owns_data = 1;

    return allocate_device_memory(mat);
}

/* ------------------------------ Random number generation ------------------------------ */
//  extern int fill_with_rand(rnd_struct* rnd_state, Matrix* mat) {
//     int len = mat->size[0] * mat->size[1];
// 
//     if (!mat->on_device)
//         return ERROR_NOT_ON_DEVICE;
// 
//     kRandomUniform<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, mat->data_device, len);
// 
//     cudaThreadSynchronize();
// 
//     if (checkCUDAError())
//         return CUDA_ERROR;
//     else
//         return 0;
// }

//  extern int fill_with_randn(rnd_struct* rnd_state, Matrix* mat) {
//     int len = mat->size[0] * mat->size[1];
// 
//     if (!mat->on_device)
//         return ERROR_NOT_ON_DEVICE;
// 
//     kRandomGaussian<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, mat->data_device, len);
// 
//     cudaThreadSynchronize();
// 
//     if (checkCUDAError())
//         return CUDA_ERROR;
//     else
//         return 0;
// }
/* ------------------------------ Algebraic operations ------------------------------ */

 extern int add_col_vec(Matrix* mat, Matrix* vec, Matrix* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kAddColVector<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, vec->data_device, target->data_device, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError()) {
        return CUDA_ERROR;
    }

    return 0;
}

 extern int add_col_mult(Matrix* mat, Matrix* vec, Matrix* target, float mult) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kAddColMult<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, vec->data_device, target->data_device, mult, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int add_row_vec(Matrix* mat, Matrix* vec, Matrix* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[1] != vec->size[1] || vec->size[0] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kAddRowVector<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, vec->data_device, target->data_device, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int mult_by_col_vec(Matrix* mat, Matrix* vec, Matrix* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kMultByColVector<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, vec->data_device, target->data_device, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int mult_by_row_vec(Matrix* mat, Matrix* vec, Matrix* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[1] != vec->size[1] || vec->size[0] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kMultByRowVector<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, vec->data_device, target->data_device, w, h);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int less_than(Matrix* mat1, Matrix* mat2, Matrix* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kLessThan<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int less_than_scalar(Matrix* mat, float val, Matrix* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kLessThanScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, val, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int greater_than(Matrix* mat1, Matrix* mat2, Matrix* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kGreaterThan<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int greater_than_scalar(Matrix* mat, float val, Matrix* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kGreaterThanScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, val, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int max_by_axis(Matrix* mat, Matrix* target, int axis) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (axis == 0) {
        if (target->size[0] != 1 || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        kMaxColumnwise<<<w,32>>>(mat->data_device, target->data_device, w, h);

        cudaThreadSynchronize();
    } else
        return ERROR_UNSUPPORTED;

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int sign(Matrix* mat, Matrix* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kSign<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int apply_sigmoid(Matrix* mat, Matrix* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kApplySigmoid<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int apply_log(Matrix* mat, Matrix* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kLog<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int apply_exp(Matrix* mat, Matrix* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kExp<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int apply_sqrt(Matrix* mat, Matrix* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kSqrt<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int apply_pow(Matrix* mat, float pow, Matrix* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kPow<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, pow, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int reciprocal(Matrix* mat, Matrix* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kReciprocal<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int dot(Matrix* mat1, Matrix* mat2, Matrix* target, float beta, float alpha) {
    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (get_leading_dimension(mat1) != get_leading_dimension(target) ||
        get_nonleading_dimension(mat2) != get_nonleading_dimension(target) ||
        get_nonleading_dimension(mat1) != get_leading_dimension(mat2)) {
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    }
    int m = get_leading_dimension(mat1),
        k = get_leading_dimension(mat2),
        n = get_nonleading_dimension(mat2);

    cublasSgemm(get_transpose_char(mat1), get_transpose_char(mat2), 
                m, n, k,
                alpha, mat1->data_device, mat1->size[0],
                mat2->data_device, mat2->size[0],
                beta, target->data_device, target->size[0]);

    if (check_cublas_error())
        return CUBLAS_ERROR;

    return 0;
}

 extern float vdot(Matrix* mat1, Matrix* mat2, int* err_code) {
    int len = mat1->size[0]*mat1->size[1];
    float res;

    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans) {
        *err_code = ERROR_TRANSPOSEDNESS;
        return 0;
    }

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1]) { 
        *err_code = ERROR_INCOMPATIBLE_DIMENSIONS;
        return 0;
    }

    res = cublasSdot(len, mat1->data_device, 1, mat2->data_device, 1);

    if (check_cublas_error()) {
        *err_code = CUBLAS_ERROR;
        return -1.;
    } else {
        *err_code = 0;
        return res;
    }
}

/* Perform the operation mat1 = mat1 + alpha * mat2. mat1 and mat2 must
   have the same transposedness. */
 extern int add_mult(Matrix* mat1, Matrix* mat2, float alpha) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cublasSaxpy(len, alpha, mat2->data_device, 1, mat1->data_device, 1);

    if (check_cublas_error())
        return CUBLAS_ERROR;

    return 0;
}

 extern int add_elementwise(Matrix* mat1, Matrix* mat2, Matrix* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kAdd<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int subtract_elementwise(Matrix* mat1, Matrix* mat2, Matrix* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kSubtract<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int divide_elementwise(Matrix* mat1, Matrix* mat2, Matrix* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kDivide<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

/* Elementwise multiplication of 2 matrices */
 extern int mult_elementwise(Matrix* mat1, Matrix* mat2, Matrix* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kMult<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int assign_scalar(Matrix* mat, float alpha) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    kAssignScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, alpha, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int mult_by_scalar(Matrix* mat, float alpha, Matrix* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kMultScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, alpha, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int divide_by_scalar(Matrix* mat, float alpha, Matrix* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kDivideScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, alpha, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern int add_scalar(Matrix* mat, float alpha, Matrix* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kAddScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, alpha, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

 extern float euclid_norm(Matrix* mat, int* err_code) {
    int len = mat->size[0]*mat->size[1];

    float res =  cublasSnrm2(len, mat->data_device, 1);

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (check_cublas_error()) {
        *err_code = CUBLAS_ERROR;
        return -1.;
    } else {
        *err_code = 0;
        return res;
    }
}
extern void init_zeros(Matrix *mat, int row, int col){
	int len = row*col;
	float *data = (float *) calloc(len, sizeof(float));
	init_from_array(mat, data, row, col);
	copy_to_device(mat);
}
extern void subself_mult_elementwise(Matrix* mat1, Matrix* target){
	if(mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
		return ;
	Matrix* tmp = (Matrix* )malloc(sizeof(Matrix));
	init_zeros(tmp, mat1->size[0], mat1->size[1]);
	mult_by_scalar(mat1, -1.0, tmp);
	add_scalar(tmp, 1.0, tmp);
	mult_elementwise(mat1, tmp, target);
}
