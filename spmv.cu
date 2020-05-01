#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "util.h"

constexpr float THRESHOLD = 1e-5f;

/*

WRITE CUDA KERNEL FOR COUNT HERE

*/
__global__
void parallel_implementation(float * sparse_matrix, int * ptr, int * indices, float * dense_vector, int rows, float * output_vector) {
    //Bound Problem For Block and Thread - [WARNING] this only works for large matrices 
    __syncthreads();
    int N = rows;
    int rows_per_block = N/gridDim.x;
    int rows_per_thread = rows_per_block/blockDim.x;
    int row_0 = (rows_per_block*blockIdx.x) + rows_per_thread*threadIdx.x;
    int row_n = row_0 + rows_per_thread;
    //printf("Thread[%d:%d] = [%d->%d]\n",blockIdx.x,threadIdx.x,row_0,row_n);
    __syncthreads();

    //Loop through all rows allocated to the thread
    for(int i = row_0; i < row_n; i++){
	//printf("[%d:%d] -> row %d\n",blockIdx.x,threadIdx.x,i);
        //Loop through all columns in row
	float sum = 0.0;
	for(int j = ptr[i]; j< ptr[i+1]; j++){
            int col = indices[j];
	    float element = sparse_matrix[j];
	    float vector_element = dense_vector[col];
	    //printf("out[%d]+=M%.2f x V%.2f = %.2f \n",i,element,vector_element,element*vector_element);
	    sum+=(element*vector_element);
		
	}
	//printf("out[%d] = %.2f\n",i,sum);
	output_vector[i] = sum;
    }
    __syncthreads();
   return;


}

float * serial_implementation(float * sparse_matrix, int * ptr, int * indices, float * dense_vector, int rows) {
    float * output = (float *)malloc(sizeof(float) * rows);
    
    for (int i = 0; i < rows; i++) {
        float accumulator = 0.f;
        for (int j = ptr[i]; j < ptr[i+1]; j++) {
            accumulator += sparse_matrix[j] * dense_vector[indices[j]];
        }
        output[i] = accumulator;
    }
    
    return output;
}

void print_attempt(float * correct, float * attempt, int rows){
    printf("Ser | Par\n");
    for(int i = 0;i<rows;i++){
	printf("%.2f | %.2f \n",correct[i],attempt[i]);
    }
    return;
}

int main(int argc, char ** argv) {
    
    assert(argc == 2);
    
    float * sparse_matrix = nullptr; 
    float * dense_vector = nullptr;
    
    int * ptr = nullptr;
    int * indices = nullptr;
    int values = 0, rows = 0, cols = 0;
    
    read_sparse_file(argv[1], &sparse_matrix, &ptr, &indices, &values, &rows, &cols);
    printf("%d %d %d\n", values, rows, cols);
    dense_vector = (float *)malloc(sizeof(float) * cols);

    // Generate "random" vector
    std::mt19937 gen(13); // Keep constant to maintain determinism between runs
    std::uniform_real_distribution<> dist(-10.0f, 10.0f);
    for (int i = 0; i < cols; i++) {
        dense_vector[i] = dist(gen);
    }

    cudaStream_t stream;
    cudaEvent_t begin, end;
    cudaStreamCreate(&stream);
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    float * h_output = (float *)malloc(sizeof(float) * rows); // THIS VARIABLE SHOULD HOLD THE TOTAL COUNT BY THE END


    //PERFORM NECESSARY VARIABLE DECLARATIONS HERE
    float * gpu_sparse_matrix;
    int * gpu_ptr;
    int * gpu_indices; 
    float * gpu_dense_vector; 
    float * gpu_output_vector;

    //CALCULATE GPU MEMORY SIZES
    int sparse_size = values*sizeof(float);
    int ptr_size = (rows+1)*sizeof(int);
    int indices_size = values*sizeof(int);
    int dense_size = cols*sizeof(float);
    int output_size = rows*sizeof(float);

    //ALLOCATE GPU MEMORY
    cudaMalloc(&gpu_sparse_matrix,sparse_size);
    cudaMalloc(&gpu_ptr,ptr_size);
    cudaMalloc(&gpu_indices,indices_size);
    cudaMalloc(&gpu_dense_vector,dense_size);
    cudaMalloc(&gpu_output_vector,output_size);


    //COPY TO GPU
    cudaMemcpy(gpu_sparse_matrix,sparse_matrix,sparse_size,cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_ptr,ptr,ptr_size,cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_indices,indices,indices_size,cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_dense_vector,dense_vector,dense_size,cudaMemcpyHostToDevice);
    //No copy needed for output vector
    

    

    cudaEventRecord(begin, stream);

    //error check
    cudaError_t err = cudaGetLastError();
    if(err!=cudaSuccess) printf("%s",cudaGetErrorString(err));

    //LAUNCH KERNEL HERE
    int THREADS_PER_BLOCK = 4;//128 ideal
    int BLOCKS = 2;//16 ideal
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCKS);
    parallel_implementation<<<grid,block,0,stream>>>(gpu_sparse_matrix, gpu_ptr, gpu_indices, gpu_dense_vector, rows, gpu_output_vector);
    //parallel_implementation<<<1,1>>>(gpu_data_in,gpu_data_out, rows, cols);

    //error check
    err = cudaGetLastError();
    if(err!=cudaSuccess) printf("%s",cudaGetErrorString(err));
    
    cudaEventRecord(end, stream);

     

    //PERFORM NECESSARY DATA TRANSFER HERE
    cudaMemcpy(h_output, gpu_output_vector, output_size, cudaMemcpyDeviceToHost);

    

    cudaStreamSynchronize(stream);

    float ms;
    cudaEventElapsedTime(&ms, begin, end);
    printf("Elapsed time: %f ms\n", ms);

     

    //DEALLOCATE RESOURCES HERE
    cudaFree(gpu_sparse_matrix);
    cudaFree(gpu_ptr);
    cudaFree(gpu_indices);
    cudaFree(gpu_dense_vector);
    cudaFree(gpu_output_vector);

   

    float * reference_output = serial_implementation(sparse_matrix, ptr, indices, dense_vector, rows);
    for (int i = 0; i < rows; i++) {
        if (fabs(reference_output[i] - h_output[i]) > THRESHOLD) {
            printf("ERROR: %f != %f at index %d\n", reference_output[i], h_output[i], i);
            //print_attempt(reference_output,h_output,rows);
	    abort();
        }
    }

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);

    free(sparse_matrix);
    free(dense_vector);
    free(ptr);
    free(indices);
    free(reference_output);
    free(h_output);

    return 0;
}
