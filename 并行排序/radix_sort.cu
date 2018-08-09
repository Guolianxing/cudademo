#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#define MAX_NUM_LISTS 32
#define NUM_ELEM 256
#define ARR_SIZE (NUM_ELEM * sizeof(int))
#define MAX_VAL 0xFFFFFFFF

/**
 * 这种基数排序是对一个整数的每个二进制位进行比较的，所以只能用于比较无符号整数（因为有符号整数小于0时最高二进制位总是1，会将负数排到正数后面）
 * 一共生成256个随机数，每32个为一组分为8组，可以看成是一个8 x 32的矩阵，然后每一列可以看成一个列表，32个线程分别对每一列的8个元素进行基数排序，
 * 然后将32个列表并行合并，32个线程正好是一个wrap的大小。
 */
__device__ void radix_sort(unsigned int *const sort_tmp, const unsigned int num_lists, const unsigned int num_elements, const unsigned int tid, unsigned int *const sort_tmp_1) {
    for(unsigned int bit = 0; bit < 32; bit++) {
        const unsigned int bit_mask = (1 << bit);
        unsigned int base_cnt_0 = 0;
        unsigned int base_cnt_1 = 0;

        for (unsigned int i = 0; i < num_elements; i += num_lists) {
            const unsigned int elem = sort_tmp[tid + i];
            if((elem & bit_mask) > 0) {
                sort_tmp_1[base_cnt_1 + tid] = elem;
                base_cnt_1 += num_lists;
            } else {
                sort_tmp[tid + base_cnt_0] = elem;
                base_cnt_0 += num_lists;
            }
        }

        for (unsigned int i = 0; i < base_cnt_1; i += num_lists) {
            sort_tmp[base_cnt_0 + i + tid] = sort_tmp_1[i + tid];
        }
    }
    __syncthreads();
}

/**
 * 复制数据到共享内存，对共享内存中的数据进行排序，在合并列表时再复制回原列表
 */
__device__ void copy_data_to_shared(const unsigned int *const data, unsigned int *const sort_tmp, const unsigned int num_lists, const unsigned int num_elements, const int tid) {
    for (unsigned int i = 0; i < num_elements; i += num_lists) {
        sort_tmp[i + tid] = data[i + tid];
    }
    __syncthreads();
}

/**
 * 合并列表
 */
__device__ void merge_array(const unsigned int *const src_array, unsigned int *const dest_array, const unsigned int num_lists, const unsigned int num_elements, const unsigned int tid) {
    const unsigned int num_elements_per_list = num_elements / num_lists;
    __shared__ unsigned int list_indexs[MAX_NUM_LISTS];
    list_indexs[tid] = 0;

    __syncthreads();

    for (unsigned int i = 0; i < num_elements; i++) {
        __shared__ unsigned int min_val;
        __shared__ unsigned int min_tid;
        unsigned int data;
        if (list_indexs[tid] < num_elements_per_list) {
            const unsigned int src_idx = tid + (list_indexs[tid] * num_lists);
            data = src_array[src_idx];
        } else {
            data = MAX_VAL;
        }

        if (tid == 0) {
            min_val = MAX_VAL;
            min_tid = MAX_VAL;
        }
 
        __syncthreads();
        atomicMin(&min_val, data);
        __syncthreads();

        if (min_val == data) {
            atomicMin(&min_tid, tid);
        }
        __syncthreads();

        if (tid == min_tid) {
            list_indexs[tid]++;
            dest_array[i] = data;
        }
    }
}

__global__ void gpu_sort_array(unsigned int *const data, const unsigned int num_lists, const unsigned int num_elements) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ unsigned int sort_tmp[NUM_ELEM];
    __shared__ unsigned int sort_tmp_1[NUM_ELEM];

    copy_data_to_shared(data, sort_tmp, num_lists, num_elements, tid);

    radix_sort(sort_tmp, num_lists, num_elements, tid, sort_tmp_1);

    merge_array(sort_tmp, data, num_lists, num_elements, tid);
}


int main() {
    unsigned int cpu_arr[NUM_ELEM];
    unsigned int *gpu_arr;
    srand((unsigned int)time(NULL));
    unsigned int i;
    for (i = 0; i < NUM_ELEM; i++) {
        cpu_arr[i] = rand() % 101;
    }
    printf("before sort:\n");
    for (i = 0; i < NUM_ELEM; i++) {
        printf("%d ", cpu_arr[i]);
    }
    printf("\n");
    cudaMalloc((void**)&gpu_arr, ARR_SIZE);
    cudaMemcpy(gpu_arr, cpu_arr, ARR_SIZE, cudaMemcpyHostToDevice);
    gpu_sort_array<<<1, 32>>>(gpu_arr, 32, NUM_ELEM);
    cudaDeviceSynchronize();
    cudaMemcpy(cpu_arr, gpu_arr, ARR_SIZE, cudaMemcpyDeviceToHost);
    cudaFree(gpu_arr);
    printf("after sort:\n");
    for (i = 0; i < NUM_ELEM; i++) {
        printf("%d ", cpu_arr[i]);
    }
    printf("\n");
    return 0;
}