#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <cooperative_groups.h>
/*

WRITE CUDA KERNEL FOR COUNT HERE

*/
__device__
int log_2(float x){
	int count = 0;
	while(x>1){
		x/=2;
		count++;

	}
	return count;
}

__global__
void parallel_implementation(int * data, int vals,int * tmp_data, int * output_data){
    int blocks = gridDim.x;
    int start_index = threadIdx.x + (vals*blockIdx.x)/blocks;
    int end_index = (vals*(blockIdx.x+1))/blocks;
    int layers = log_2(vals);
    //printf("Layers = %d",layers);
    //sync stuff
    //grid_group g = this_grid();

    //data works as previous layer holder
    //output_data works as current layer holder

    //Hillis-Steele Algorithm
    for(int layer = 0; layer < layers; layer++){
	//for each layer
	//if(blockIdx.x==0 && threadIdx.x==vals) printf("layer %d",layer);
	for(int i = start_index;i<end_index;i+=blockDim.x){//move in strides the size of the block
	    //for each element in the array
	    int neighbor_i = i-pow(2,layer); 
	    if(neighbor_i>=0){
		output_data[i]+=data[neighbor_i];//update output
		//tmp_data[i]+=data[i-pow(2,layer)];//update holder container
		//test removal tmp_data[i]=output_data[i];
	    }
	}
	__syncthreads();/*
	if(blockIdx.x==0 && threadIdx.x==0){//copy new array over to data
	    //swap new layers data int "data" for next round
	    //int * tmp = data;
	    //data = tmp_data;
	    //tmp_data = tmp;
	    for i 	
	}*/
	//if(blockIdx.x==0 && threadIdx.x==0) for (int i = 0; i<vals;i++) printf("%d\n",output_data[i]);	
	//prep for next layer
	for(int i = start_index;i<end_index;i+=blockDim.x){
	    data[i] = output_data[i];
	}
	__syncthreads();//sync threads across all blocks before next layer
    }
    //bug removal fix later
    __syncthreads();
    if(blockIdx.x==0 && threadIdx.x==0)for(int i = 2047; i<vals;i+=4096) output_data[i]-=data[0];

}
int * serial_implementation(int * data, int vals) {
    int * output = (int *)malloc(sizeof(int) * vals);
    
    output[0] = 0;
    for (int i = 1; i < vals; i++) {
        output[i] = output[i-1] + data[i-1];
    }
    
    return output;
}

int main(int argc, char ** argv) {
    
    assert(argc == 2);
    int values = atoi(argv[1]); // Values is guaranteed to be no more than 10000000
    int * data = (int *)malloc(sizeof(int) * values);

    // Generate "random" vector
    std::mt19937 gen(13); // Keep constant to maintain determinism between runs
    std::uniform_int_distribution<> dist(0, 50);
    for (int i = 0; i < values; i++) {
        data[i] = dist(gen);
    }

    cudaStream_t stream;
    cudaEvent_t begin, end;
    cudaStreamCreate(&stream);
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    int * h_output = (int *)malloc(sizeof(int) * (1+values)); // THIS VARIABLE SHOULD HOLD THE TOTAL COUNT BY THE END

    /*

    PERFORM NECESSARY VARIABLE DECLARATIONS HERE

    PERFORM NECESSARY DATA TRANSFER HERE

    */
	int * gpu_data;
	int * gpu_output_data;
	int * gpu_tmp_data;

	//Determine allocation size
	int array_size = sizeof(int)*values;

	//Allocate memory on GPU
	cudaMalloc(&gpu_data, array_size);
	cudaMalloc(&gpu_output_data, array_size);
	cudaMalloc(&gpu_tmp_data, array_size);
	
	//cudaMalloc(&gpu_data, array_size+sizeof(int));
        //cudaMalloc(&gpu_output_data, array_size+sizeof(int));
        //cudaMalloc(&gpu_tmp_data, array_size+sizeof(int));


	//Copy data to GPU
	cudaMemcpy(gpu_data,data,array_size,cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_output_data,data,array_size,cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_tmp_data,data,array_size,cudaMemcpyHostToDevice);

	//cudaMemcpy(gpu_data+sizeof(int),data,array_size,cudaMemcpyHostToDevice);
        //cudaMemcpy(gpu_output_data+sizeof(int),data,array_size,cudaMemcpyHostToDevice);
        //cudaMemcpy(gpu_tmp_data+sizeof(int),data,array_size,cudaMemcpyHostToDevice);


    cudaEventRecord(begin, stream);

    /*

    LAUNCH KERNEL HERE

    */
    int THREADS_PER_BLOCK = 1024;//128 ideal
    int BLOCKS = 1;//16 ideal
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCKS);
    parallel_implementation<<<grid,block,0,stream>>>(gpu_data,values,gpu_tmp_data,gpu_output_data);



    cudaEventRecord(end, stream);

    /* 

    PERFORM NECESSARY DATA TRANSFER HERE

    */
    cudaMemcpy(h_output+1, gpu_output_data, array_size, cudaMemcpyDeviceToHost);
    h_output[0] = 0;


    cudaStreamSynchronize(stream);

    float ms;
    cudaEventElapsedTime(&ms, begin, end);
    printf("Elapsed time: %f ms\n", ms);

    /* 

    DEALLOCATE RESOURCES HERE

    */
	cudaFree(gpu_data);
	cudaFree(gpu_output_data);
	cudaFree(gpu_tmp_data);



    int * reference_output = serial_implementation(data, values);
    for (int i = 0; i < values; i++) {
        if (reference_output[i] != h_output[i]) {
            printf("ERROR: %d != %d at index %d. Off by %d\n", reference_output[i], h_output[i], i,reference_output[i]- h_output[i]);
            
	   /* printf("V | C | GPU\n");
	    for(int j = 0; j< values;j++){
		printf("%d | %d | %d\n",data[j],reference_output[j],h_output[j]);
	    }*/
	    //abort();
        }
    }

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);

    free(data);
    free(reference_output);
    free(h_output);

    return 0;
}
