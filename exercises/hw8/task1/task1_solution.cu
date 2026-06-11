/*
 *  Copyright 2014 NVIDIA Corporation
 *
 *  根据 Apache License 2.0（“许可证”）授权；
 *  除非遵守该许可证，否则不得使用此文件。
 *  你可以在以下地址获取许可证副本：
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  除非适用法律要求或书面同意，软件
 *  按“原样”依据许可证分发，
 *  不附带任何明示或暗示的保证或条件。
 *  有关许可证下具体权限和限制，
 *  请参见许可证。
 */

#include <stdio.h>

#ifdef DEBUG
#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);} 
#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);} 
#else
#define CUDA_CALL(F) (F)
#define CUDA_CHECK() 
#endif

/* X 和 Y 方向线程块大小的定义 */

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

/* 矩阵线性维度定义 */

#define SIZE 4096

/* 用二维索引按列主序访问一维内存数组的宏 */

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

/* 朴素矩阵转置的 CUDA 核函数 */

__global__ void naive_cuda_transpose( const int m, 
                                      const double * const a, 
                                      double * const c )
{
  const int myRow = blockDim.x * blockIdx.x + threadIdx.x;
  const int myCol = blockDim.y * blockIdx.y + threadIdx.y;

  if( myRow < m && myCol < m )
  {
    c[INDX( myRow, myCol, m )] = a[INDX( myCol, myRow, m )];
  } /* if 结束 */
  return;

} /* naive_cuda_transpose 结束 */

void host_transpose( const int m, const double * const a, double *c )
{
	
/* 
 *  朴素矩阵转置写在这里。
 */
 
  for( int j = 0; j < m; j++ )
  {
    for( int i = 0; i < m; i++ )
      {
        c[INDX(i,j,m)] = a[INDX(j,i,m)];
      } /* for i 结束 */
  } /* for j 结束 */

} /* host_dgemm 结束 */

int main( int argc, char *argv[] )
{

  int size = SIZE;

  fprintf(stdout, "Matrix size is %d\n",size);

/* 声明数组指针 */

  double *h_a, *h_c;
  double *d_a, *d_c;
 
  size_t numbytes = (size_t) size * (size_t) size * sizeof( double );

/* 分配主机内存 */

  h_a = (double *) malloc( numbytes );
  if( h_a == NULL )
  {
    fprintf(stderr,"Error in host malloc h_a\n");
    return 911;
  }

  h_c = (double *) malloc( numbytes );
  if( h_c == NULL )
  {
    fprintf(stderr,"Error in host malloc h_c\n");
    return 911;
  }

/* 分配设备内存 */

  CUDA_CALL( cudaMalloc( (void**) &d_a, numbytes ) );
  CUDA_CALL( cudaMalloc( (void**) &d_c, numbytes ) );

/* 将结果矩阵置零 */

  memset( h_c, 0, numbytes );
  CUDA_CALL( cudaMemset( d_c, 0, numbytes ) );

  fprintf( stdout, "Total memory required per matrix is %lf MB\n", 
     (double) numbytes / 1000000.0 );

/* 用随机值初始化输入矩阵 */

  for( int i = 0; i < size * size; i++ )
  {
    h_a[i] = double( rand() ) / ( double(RAND_MAX) + 1.0 );
  }

/* 将输入矩阵从主机端复制到设备端 */

  CUDA_CALL( cudaMemcpy( d_a, h_a, numbytes, cudaMemcpyHostToDevice ) );

/* 创建并启动计时器 */

  cudaEvent_t start, stop;
  CUDA_CALL( cudaEventCreate( &start ) );
  CUDA_CALL( cudaEventCreate( &stop ) );
  CUDA_CALL( cudaEventRecord( start, 0 ) );

/* 调用朴素 CPU 转置函数 */

  host_transpose( size, h_a, h_c );

/* 停止 CPU 计时器 */

  CUDA_CALL( cudaEventRecord( stop, 0 ) );
  CUDA_CALL( cudaEventSynchronize( stop ) );
  float elapsedTime;
  CUDA_CALL( cudaEventElapsedTime( &elapsedTime, start, stop ) );

/* 打印 CPU 计时信息 */

  fprintf(stdout, "Total time CPU is %f sec\n", elapsedTime / 1000.0f );
  fprintf(stdout, "Performance is %f GB/s\n", 
    8.0 * 2.0 * (double) size * (double) size / 
    ( (double) elapsedTime / 1000.0 ) * 1.e-9 );

/* 设置线程块大小和网格大小 */

  dim3 threads( THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1 );
  dim3 blocks( ( size / THREADS_PER_BLOCK_X ) + 1, 
               ( size / THREADS_PER_BLOCK_Y ) + 1, 1 );

/* 启动计时器 */
  CUDA_CALL( cudaEventRecord( start, 0 ) );

/* 调用朴素 GPU 转置核函数 */

  naive_cuda_transpose<<< blocks, threads >>>( size, d_a, d_c );
  CUDA_CHECK()
  CUDA_CALL( cudaDeviceSynchronize() );

/* 停止计时器 */

  CUDA_CALL( cudaEventRecord( stop, 0 ) );
  CUDA_CALL( cudaEventSynchronize( stop ) );
  CUDA_CALL( cudaEventElapsedTime( &elapsedTime, start, stop ) );

/* 打印 GPU 计时信息 */

  fprintf(stdout, "Total time GPU is %f sec\n", elapsedTime / 1000.0f );
  fprintf(stdout, "Performance is %f GB/s\n", 
    8.0 * 2.0 * (double) size * (double) size / 
    ( (double) elapsedTime / 1000.0 ) * 1.e-9 );

/* 将数据从设备端复制到主机端 */

  CUDA_CALL( cudaMemset( d_a, 0, numbytes ) );
  CUDA_CALL( cudaMemcpy( h_a, d_c, numbytes, cudaMemcpyDeviceToHost ) );

/* 将 GPU 结果与 CPU 结果比较以验证正确性 */

  for( int j = 0; j < size; j++ )
  {
    for( int i = 0; i < size; i++ )
    {
      if( h_c[INDX(i,j,size)] != h_a[INDX(i,j,size)] ) 
      {
        printf("Error in element %d,%d\n", i,j );
        printf("Host %f, device %f\n",h_c[INDX(i,j,size)],
                                      h_a[INDX(i,j,size)]);
        printf("FAIL\n");
        goto end;
      } /* end fi */
    } /* for i 结束 */
  } /* for j 结束 */

/* 释放内存 */
  printf("PASS\n");

  end:
  free( h_a );
  free( h_c );
  CUDA_CALL( cudaFree( d_a ) );
  CUDA_CALL( cudaFree( d_c ) );
  CUDA_CALL( cudaDeviceReset() );

  return 0;
} /* end main */
