#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * 矩阵a：32 x 16   b：16 x 32，随机生成
 * 一个block中32 x 4个线程，共1 x 8个block
 */
#define A_ROW_NUM 32
#define A_COL_NUM 16
#define B_ROW_NUM 16
#define B_COL_NUM 32
#define TOTAL_SIZE (A_COL_NUM * A_ROW_NUM * sizeof(int))
#define RES_SIZE (A_ROW_NUM * A_ROW_NUM * sizeof(int))

__global__ void cal_by_gpu(int *a, int *b, int *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int i, res = 0;
    for (i = 0; i < A_COL_NUM; i++) {
        res += a[row * A_COL_NUM + i] * b[i * B_COL_NUM + col];
    }

    c[row * A_ROW_NUM + col] = res;
}

void cal_by_cpu(int a[][A_COL_NUM], int b[][B_COL_NUM], int c[][A_ROW_NUM]) {
    int i, j, k, res;
    for (i = 0; i < A_ROW_NUM; i++) {
        for (j = 0; j < A_ROW_NUM; j++) {
            res = 0;
            for (k = 0; k < A_COL_NUM; k++) {
                res += a[i][k] * b[k][j];
            }
            c[i][j] = res;
        }
    }
}

int main() {
    int cpu_a[A_ROW_NUM][A_COL_NUM], cpu_b[B_ROW_NUM][B_COL_NUM], cpu_res[A_ROW_NUM][A_ROW_NUM]; 
    int *gpu_a, *gpu_b, *gpu_res;
    int i, j;

    srand((unsigned int)time(NULL));
    for (i = 0; i < A_ROW_NUM; i++) {
        for (j = 0; j < A_COL_NUM; j++) {
            cpu_a[i][j] = rand() % 41 - 20;
            cpu_b[j][i] = rand() % 41 - 20;
        }
    }

    cudaMalloc((void**)&gpu_a, TOTAL_SIZE);
    cudaMalloc((void**)&gpu_b, TOTAL_SIZE);
    cudaMalloc((void**)&gpu_res, RES_SIZE);

    cudaMemcpy(gpu_a, cpu_a, TOTAL_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, cpu_b, TOTAL_SIZE, cudaMemcpyHostToDevice);

    dim3 thread_square(32, 4);
    dim3 block_square(1, 8);

    cal_by_gpu<<<block_square, thread_square>>>(gpu_a, gpu_b, gpu_res);

    cudaDeviceSynchronize();

    cudaMemcpy(cpu_res, gpu_res, RES_SIZE, cudaMemcpyDeviceToHost);

    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_res);

    printf("a:\n");
    for (i = 0; i < A_ROW_NUM; i++) {
        for (j = 0; j < A_COL_NUM; j++) {
            printf("%d ", cpu_a[i][j]);
        }
        printf("\n");
    }
    printf("--------------------------------------------\n");
    printf("b:\n");
    for (i = 0; i < B_ROW_NUM; i++) {
        for (j = 0; j < B_COL_NUM; j++) {
            printf("%d ", cpu_b[i][j]);
        }
        printf("\n");
    }
    printf("--------------------------------------------\n");
    printf("cal_by_gpu:\n");
    for (i = 0; i < A_ROW_NUM; i++) {
        for (j = 0; j < A_ROW_NUM; j++) {
            printf("%d ", cpu_res[i][j]);
        }
        printf("\n");
    }
    printf("--------------------------------------------\n");
    cal_by_cpu(cpu_a, cpu_b, cpu_res);
    printf("cal_by_cpu:\n");
    for (i = 0; i < A_ROW_NUM; i++) {
        for (j = 0; j < A_ROW_NUM; j++) {
            printf("%d ", cpu_res[i][j]);
        }
        printf("\n");
    }
    return 0;
    
}