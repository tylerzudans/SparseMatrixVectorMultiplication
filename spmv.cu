#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "util.h"

constexpr float THRESHOLD = 1e-6f;

/*

WRITE CUDA KERNEL FOR COUNT HERE

*/

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

    /*

    PERFORM NECESSARY VARIABLE DECLARATIONS HERE

    PERFORM NECESSARY DATA TRANSFER HERE

    */

    cudaEventRecord(begin, stream);

    /*

    LAUNCH KERNEL HERE

    */

    cudaEventRecord(end, stream);

    /* 

    PERFORM NECESSARY DATA TRANSFER HERE

    */

    cudaStreamSynchronize(stream);

    float ms;
    cudaEventElapsedTime(&ms, begin, end);
    printf("Elapsed time: %f ms\n", ms);

    /* 

    DEALLOCATE RESOURCES HERE

    */

    float * reference_output = serial_implementation(sparse_matrix, ptr, indices, dense_vector, rows);
    for (int i = 0; i < rows; i++) {
        if (fabs(reference_output[i] - h_output[i]) > THRESHOLD) {
            printf("ERROR: %f != %f at index %d\n", reference_output[i], h_output[i], i);
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
