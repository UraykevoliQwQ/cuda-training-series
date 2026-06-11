#include <iostream>
// 线程块大小
#define BLOCK_SIZE 32

// 矩阵按行主序存储：
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

// 获取矩阵元素
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// 设置矩阵元素
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}

// 获取 A 中 BLOCK_SIZExBLOCK_SIZE 的子矩阵 Asub，它位于
// 相对于 A 左上角向右 col 个子矩阵、向下 row 个子矩阵的位置
// 从 A 的左上角开始计算
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}


// 矩阵乘法核函数的前向声明
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// 矩阵乘法 - 主机端代码
// 假设矩阵维度是 BLOCK_SIZE 的倍数
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // 将 A 和 B 加载到设备内存
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
    cudaMemcpyHostToDevice);

    // 在设备内存中分配 C
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // 调用核函数
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // 从设备内存读取 C
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// 由 MatMul() 调用的矩阵乘法核函数
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // 线程块所在的行和列
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // 每个线程块计算 C 的一个子矩阵 Csub
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // 每个线程计算 Csub 的一个元素
    // 通过累加到 Cvalue 中得到结果
    float Cvalue = 0;

    // 线程在 Csub 内的行和列
    int row = threadIdx.y;
    int col = threadIdx.x;

    // 遍历计算 Csub 所需的 A 和 B 的所有子矩阵
    // 计算 Csub 所需
    // 将每对子矩阵相乘
    // 并累加结果
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        // 获取 A 的子矩阵 Asub
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // 获取 B 的子矩阵 Bsub
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // 共享内存分别用于存储 Asub 和 Bsub
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // 将 Asub 和 Bsub 从设备内存加载到共享内存
        // 每个线程加载每个子矩阵的一个元素
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // 同步以确保子矩阵已加载完成
        // 再开始计算
        __syncthreads();
        // 将 Asub 和 Bsub 相乘
        for (int e = 0; e <= BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

    }

    // 将 Csub 写入设备内存
    // 每个线程写入一个元素
    SetElement(Csub, row, col, Cvalue);
}

int main(){
    const int num_m = 3;  // 需要 3 个矩阵
    const int side_dim = 128;  // 方阵边长
    Matrix *m = new Matrix[num_m]; // 分配矩阵存储空间第 1 部分
    for (int i = 0; i < num_m; i++){
        m[i].width = m[i].height = m[i].stride = side_dim; // 设置矩阵参数
        m[i].elements = new float[side_dim*side_dim];      // 分配矩阵存储空间第 2 部分
        if (i < 2)                                         // 初始化前两个矩阵
            for (int j = 0; j < side_dim*side_dim; j++) m[i].elements[j] = 1.0f; }
    MatMul(m[0], m[1], m[2]);  // 执行矩阵乘法
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    for (int i = 0; i < side_dim*side_dim; i++) // 执行结果检查
        if (m[2].elements[i] != (float)side_dim) {std::cout << "Mismatch at index: " << i << " expected: " << (float)side_dim << " got: " << m[2].elements[i] << std::endl; return 0;}
    std::cout << "Success!" << std::endl;
    for (int i = 0; i < num_m; i++)
        delete[] m[i].elements;
    delete[] m;
    return 0;
}