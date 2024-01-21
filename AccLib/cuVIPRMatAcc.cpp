/*
 * cuVIPRMatAcc.cpp
 *
 *  Created on: Jun 28, 2023
 *      Author: fangming
 */
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <math.h>
#include <cuda_runtime.h>

#include "cublas_v2.h"
#include <cublasLt.h>
#include "cuda_fp8.h"
#include "cuVIPRMatAcc.hpp"
using namespace std;
static cuVIPRMatAccEnv MatAccEnv;
int32 Init_cuVIPRMatAcc(void)
{
	cublasStatus_t stat;

	MatAccEnv.fp32_gpu_db_hol_mat = NULL;
	MatAccEnv.fp32_db_hol_mat_h = 0;
	MatAccEnv.fp32_db_hol_mat_w = 0;

    // query holistic feature memory
	MatAccEnv.fp32_gpu_q_hol_mat = NULL;
    MatAccEnv.fp32_q_hol_mat_h = 0;
    MatAccEnv.fp32_q_hol_mat_w = 0;
    MatAccEnv.fp32_q_hol_mat_size = 0;
    // query out holistic feature memory
    MatAccEnv.fp32_gpu_out_hol_mat = NULL;
    MatAccEnv.fp32_out_hol_mat_h = 0;
    MatAccEnv.fp32_out_hol_mat_w = 0;
    MatAccEnv.fp32_out_hol_mat_size = 0;


	MatAccEnv.fp32_base_mat_h = 0;
	MatAccEnv.fp32_base_mat_w = 0;
	MatAccEnv.fp32_base_mat_size = 0;
	MatAccEnv.fp32_gpu_base_mat = NULL;
	MatAccEnv.fp32_base_mat_end_col_pos = 0;

	MatAccEnv.db_frame_num = 0;
	MatAccEnv.db_each_frame_keypoint_num = NULL;
	MatAccEnv.attached_db_frame_num = 0;

	MatAccEnv.fp32_query_mat_h = 0;
	MatAccEnv.fp32_query_mat_w = 0;
	MatAccEnv.fp32_query_mat_size = 0;
	MatAccEnv.fp32_gpu_query_mat = NULL;

	MatAccEnv.fp32_out_mat_h = 0;
	MatAccEnv.fp32_out_mat_w = 0;
	MatAccEnv.fp32_out_mat_size = 0;
	MatAccEnv.fp32_gpu_out_mat = NULL;



	MatAccEnv.fp32_norm_vects = NULL;
	MatAccEnv.fp32_vect_num = 0;
	MatAccEnv.fp32_vect_dim = 0;
	MatAccEnv.fp32_vects_norm2 = NULL;

	stat = cublasCreate(&MatAccEnv.blas_handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
	  printf ("CUBLAS initialization failed\n");
	  return CU_VIPR_MAT_ACC_ERR;
	}
	printf("Successfully create cublas handle!!\n");
}

int32 malloDataBaseMatFP32(const int32 mat_h,  const int32 mat_w, const int32 db_frame_num)
{
	//printf("%d  %d\n", mat_h, mat_w);
    cudaError_t cudaStat;
    cublasStatus_t stat;
    // if previously allocate gpu memory for MatAccEnv.fp32_gpu_base_mat, free it
    if (MatAccEnv.fp32_gpu_base_mat!=NULL) {
      cudaFree (MatAccEnv.fp32_gpu_base_mat);
    }
    // allocate cpu memory to hole the number of keypoint for each frame in db
    MatAccEnv.db_frame_num = db_frame_num;
    MatAccEnv.db_each_frame_keypoint_num = (int32*)malloc(sizeof(int32)*MatAccEnv.db_frame_num);
    if (MatAccEnv.db_each_frame_keypoint_num==NULL) {
      printf("Allocate CPU memory for db_each_frame_keypoint_num failed!\n");
      return CU_VIPR_MAT_ACC_ERR;
    }

    MatAccEnv.db_frame_base_mat_start_col = (int32*)malloc(sizeof(int32)*MatAccEnv.db_frame_num);
    if (MatAccEnv.db_frame_base_mat_start_col==NULL) {
      printf("Allocate CPU memory for db_frame_base_mat_start_col failed!\n");
      return CU_VIPR_MAT_ACC_ERR;
    }

    // create cuda memory for database mat
    cudaStat = cudaMalloc ((void**)&MatAccEnv.fp32_gpu_base_mat, mat_h*mat_w*sizeof(fp32));
    if (cudaStat != cudaSuccess) {
      printf ("device memory allocation failed\n");
      MatAccEnv.fp32_gpu_base_mat = NULL;
      return CU_VIPR_MAT_ACC_ERR;
    }
    printf ("Successfully allocate GPU memory with shape=(%d,%d)!\n", mat_h, mat_w);
    MatAccEnv.fp32_base_mat_h = mat_h;
    MatAccEnv.fp32_base_mat_w = mat_w;
    MatAccEnv.fp32_base_mat_size = mat_h*mat_w;
    MatAccEnv.fp32_base_mat_end_col_pos = 0;

    return CU_VIPR_MAT_ACC_OK;
}


int32 SetDBHolDataFP32(const int32 mat_h,  const int32 mat_w, const fp32* data)
{

    fp32*  fp32_gpu_db_hol_mat;
    uint32 fp32_db_hol_mat_h;
    uint32 fp32_db_hol_mat_w;
	//printf("%d  %d\n", mat_h, mat_w);
    cudaError_t cudaStat;
    cublasStatus_t stat;
    // if previously allocate gpu memory for MatAccEnv.fp32_gpu_base_mat, free it
    if (MatAccEnv.fp32_gpu_db_hol_mat!=NULL) {
      cudaFree (MatAccEnv.fp32_gpu_db_hol_mat);
    }

    // create cuda memory for query mat
    cudaStat = cudaMalloc ((void**)&MatAccEnv.fp32_gpu_db_hol_mat, mat_h*mat_w*sizeof(fp32));
    if (cudaStat != cudaSuccess) {
      printf ("device memory to allocate db holistic feature failed\n");
      MatAccEnv.fp32_gpu_db_hol_mat = NULL;
      return CU_VIPR_MAT_ACC_ERR;
    }
    printf ("Successfully allocate GPU memory with shape=(%d,%d)!\n", mat_h, mat_w);
    MatAccEnv.fp32_db_hol_mat_h = mat_h;
    MatAccEnv.fp32_db_hol_mat_w = mat_w;

    // set the data
    stat = cublasSetMatrix (mat_h, mat_w, sizeof(*data), data, mat_h, MatAccEnv.fp32_gpu_db_hol_mat, mat_h);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("data download failed");
      cudaFree (MatAccEnv.fp32_gpu_base_mat);
      MatAccEnv.fp32_gpu_db_hol_mat =NULL;
      return CU_VIPR_MAT_ACC_ERR;
    }
    return CU_VIPR_MAT_ACC_OK;
}

int32 SetDataBaseMatFP32(
                 const int32 mat_h,
				 const int32 mat_w,
                 const fp32* data )
{
	//printf("%d  %d\n", mat_h, mat_w);
    cudaError_t cudaStat;
    cublasStatus_t stat;

    // if previously allocate gpu memory for MatAccEnv.fp32_gpu_base_mat, free it
    if (MatAccEnv.fp32_gpu_base_mat!=NULL) {
      cudaFree (MatAccEnv.fp32_gpu_base_mat);
    }
    // create cuda memory for query mat
    cudaStat = cudaMalloc ((void**)&MatAccEnv.fp32_gpu_base_mat, mat_h*mat_w*sizeof(fp32));
    if (cudaStat != cudaSuccess) {
      printf ("device memory allocation failed");
      MatAccEnv.fp32_gpu_base_mat = NULL;
      return CU_VIPR_MAT_ACC_ERR;
    }
    MatAccEnv.fp32_base_mat_h = mat_h;
    MatAccEnv.fp32_base_mat_w = mat_w;
    MatAccEnv.fp32_base_mat_size = mat_h*mat_w;

    // set the base mat in gpu with data
    /*stat = cublasSetMatrix (mat_h, mat_w, sizeof(*data), data, mat_h, MatAccEnv.fp32_gpu_base_mat, mat_h);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("data download failed");
      cudaFree (MatAccEnv.fp32_gpu_base_mat);
      MatAccEnv.fp32_gpu_base_mat =NULL;
      return CU_VIPR_MAT_ACC_ERR;
    }*/
    return CU_VIPR_MAT_ACC_OK;
}

int32 SetDatabaseFeatureMatInfo(const int32* db_frame_keypoint_num, const int32* db_frame_base_mat_start_col)
{
  for (int i=0; i<MatAccEnv.db_frame_num; i++) {
	MatAccEnv.db_each_frame_keypoint_num[i]   = db_frame_keypoint_num[i];
	MatAccEnv.db_frame_base_mat_start_col[i]  = db_frame_base_mat_start_col[i];
  }
  return CU_VIPR_MAT_ACC_OK;
}

int32 BatchAttachDataBaseMatrixFP32(const int32 mat_h,  const int32 mat_w,  const fp32* data,
		                                   const int32 batch_db_frame_num, const int32* batch_db_frame_keypoint_num)
{
  cudaError_t cudaStat;
  cublasStatus_t stat;
  /*Attach the keypoint feature*/
  fp32 *col_start = MatAccEnv.fp32_gpu_base_mat+ MatAccEnv.fp32_base_mat_end_col_pos*MatAccEnv.fp32_base_mat_h;
  stat = cublasSetMatrix (mat_h, mat_w, sizeof(*data), data, mat_h, col_start, mat_h);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("data download failed");
    cudaFree (MatAccEnv.fp32_gpu_base_mat);
    MatAccEnv.fp32_gpu_base_mat =NULL;
    return CU_VIPR_MAT_ACC_ERR;
  }
  MatAccEnv.fp32_base_mat_end_col_pos += mat_w;
  /*Set the number keypoint for the attached db frame*/
  int32 next_frame = MatAccEnv.attached_db_frame_num;

  for (int32 i=0; i<batch_db_frame_num; i++ ) {
	  MatAccEnv.db_each_frame_keypoint_num[next_frame+i]=batch_db_frame_keypoint_num[i];
  }
  MatAccEnv.attached_db_frame_num += batch_db_frame_num;

  printf("Successfully attach %d DB frame matrix to GPU at (:,%d:%d)\n",batch_db_frame_num, MatAccEnv.fp32_base_mat_end_col_pos, MatAccEnv.fp32_base_mat_end_col_pos+mat_w);
  return CU_VIPR_MAT_ACC_OK;
}


int32 BatchQueryHolisticFeatureFP32(const int32 mat_h,  const int32 mat_w,  const fp32* data, fp32* result)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    // allocate cuda memory for query data
    // check if query mat allocated previously suits for the current query matrix
    uint64 query_mat_size = mat_h*mat_w;
    // if the currently allocated gpu query memory buffer smaller than the required
    if  (MatAccEnv.fp32_q_hol_mat_size<query_mat_size) {
      // check if had previously allocate gpu memory.  MatAccEnv.fp32_gpu_query_mat not equals to NULL, release the current memory
      if (MatAccEnv.fp32_gpu_q_hol_mat != NULL) {
        cudaFree (MatAccEnv.fp32_gpu_q_hol_mat);
      }
      //printf("Reallocate query memory\n");
      cudaStat = cudaMalloc ((void**)&MatAccEnv.fp32_gpu_q_hol_mat, query_mat_size*sizeof(fp32));
      if (cudaStat != cudaSuccess) {
        printf ("query GPU memory allocation failed\n");
        // if allocate gpu memory failed reset the gpu memory pointer to NULL
        MatAccEnv.fp32_gpu_q_hol_mat = NULL;
        MatAccEnv.fp32_q_hol_mat_h = 0;
        MatAccEnv.fp32_q_hol_mat_w = 0;
        MatAccEnv.fp32_q_hol_mat_size = 0;
        return CU_VIPR_MAT_ACC_ERR;
      }
      // reset the query mat size parameter
      MatAccEnv.fp32_q_hol_mat_size = query_mat_size;
    }
    MatAccEnv.fp32_q_hol_mat_h    = mat_h;
    MatAccEnv.fp32_q_hol_mat_w    = mat_w;

    // allocate result matrix gpu memory
    // check if out mat allocated previously suits for the current query matrix
    uint64 out_mat_size = mat_w * MatAccEnv.fp32_db_hol_mat_w;
    // if the currently allocated gpu out memory buffer smaller than the required
    if (MatAccEnv.fp32_out_hol_mat_size<out_mat_size) {
      // check if had previously allocate gpu memory.  MatAccEnv.fp32_gpu_out_mat not equals to NULL, release the current memory
      if (MatAccEnv.fp32_gpu_out_hol_mat != NULL) {
        cudaFree (MatAccEnv.fp32_gpu_out_hol_mat);
      }

      cudaStat = cudaMalloc ((void**)&MatAccEnv.fp32_gpu_out_hol_mat, out_mat_size*sizeof(fp32));
      if (cudaStat != cudaSuccess) {
        printf ("result GPU memory allocation failed\n");
        // if allocate gpu memory failed reset the gpu memory pointer to NULL
        MatAccEnv.fp32_gpu_out_hol_mat = NULL;
        MatAccEnv.fp32_out_hol_mat_h = 0;
        MatAccEnv.fp32_out_hol_mat_w = 0;
        MatAccEnv.fp32_out_hol_mat_size = 0;
        return CU_VIPR_MAT_ACC_ERR;
      }
      // reset the out mat size parameter

      MatAccEnv.fp32_out_hol_mat_size = out_mat_size;
    }
    MatAccEnv.fp32_out_hol_mat_w    = MatAccEnv.fp32_db_hol_mat_w;
    MatAccEnv.fp32_out_hol_mat_h    = mat_w;


    // set the query mat in gpu with data
    stat = cublasSetMatrix (mat_h, mat_w, sizeof(fp32), data, mat_h, MatAccEnv.fp32_gpu_q_hol_mat, mat_h);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      switch(stat) {
      case CUBLAS_STATUS_INVALID_VALUE:
    	  printf ("data download failed: CUBLAS_STATUS_INVALID_VALUE");
    	  break;
      case CUBLAS_STATUS_MAPPING_ERROR:
    	  printf ("data download failed: CUBLAS_STATUS_MAPPING_ERROR");
    	  break;
      }

      return CU_VIPR_MAT_ACC_ERR;
    }

    // do the matmul in gpu
    float alpha = 1.0;
    float beta = 0.0;
    //printf("gemm\n");
    stat = cublasSgemm(    MatAccEnv.blas_handle,
        		           CUBLAS_OP_T, CUBLAS_OP_N,
						   MatAccEnv.fp32_q_hol_mat_w, MatAccEnv.fp32_db_hol_mat_w, MatAccEnv.fp32_q_hol_mat_h,
    					   &alpha,
						   MatAccEnv.fp32_gpu_q_hol_mat, MatAccEnv.fp32_q_hol_mat_h,
						   MatAccEnv.fp32_gpu_db_hol_mat, MatAccEnv.fp32_db_hol_mat_h,
    					   &beta,
						   MatAccEnv.fp32_gpu_out_hol_mat, MatAccEnv.fp32_out_hol_mat_h);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("gemm failed\n");
      return CU_VIPR_MAT_ACC_ERR;
    }
    //printf("gemm finish\n");
    // get the result matrix from GPU
    stat = cublasGetMatrix (mat_w, MatAccEnv.fp32_db_hol_mat_w,
    		                sizeof(fp32),
							MatAccEnv.fp32_gpu_out_hol_mat, mat_w,
							result,                         mat_w);
    //printf("gemm finish a\n");
    if (stat != CUBLAS_STATUS_SUCCESS) {
        switch(stat) {
        case CUBLAS_STATUS_INVALID_VALUE:
      	  printf ("data upload failed: CUBLAS_STATUS_INVALID_VALUE\n");
      	  break;
        case CUBLAS_STATUS_MAPPING_ERROR:
      	  printf ("data upload failed: CUBLAS_STATUS_MAPPING_ERROR\n");
      	  break;
        }
        return CU_VIPR_MAT_ACC_ERR;
    }
    //printf("gemm finish 2\n");
    return CU_VIPR_MAT_ACC_OK;
}


int32 BatchQueryDataBaseFP32(const int32 mat_h,  const int32 mat_w,  const fp32* data, fp32* result)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;

    // allocate cuda memory for query data
    // check if query mat allocated previously suits for the current query matrix
    uint64 query_mat_size = mat_h*mat_w;
    // if the currently allocated gpu query memory buffer smaller than the required
    if  (MatAccEnv.fp32_query_mat_size<query_mat_size) {
      // check if had previously allocate gpu memory.  MatAccEnv.fp32_gpu_query_mat not equals to NULL, release the current memory
      if (MatAccEnv.fp32_gpu_query_mat != NULL) {
        cudaFree (MatAccEnv.fp32_gpu_query_mat);
      }
      //printf("Reallocate query memory\n");
      cudaStat = cudaMalloc ((void**)&MatAccEnv.fp32_gpu_query_mat, query_mat_size*sizeof(fp32));
      if (cudaStat != cudaSuccess) {
        printf ("query GPU memory allocation failed\n");
        // if allocate gpu memory failed reset the gpu memory pointer to NULL
        MatAccEnv.fp32_gpu_query_mat = NULL;
        MatAccEnv.fp32_query_mat_h = 0;
        MatAccEnv.fp32_query_mat_w = 0;
        MatAccEnv.fp32_query_mat_size = 0;
        return CU_VIPR_MAT_ACC_ERR;
      }
      // reset the query mat size parameter
      MatAccEnv.fp32_query_mat_size = query_mat_size;
    }
    MatAccEnv.fp32_query_mat_h    = mat_h;
    MatAccEnv.fp32_query_mat_w    = mat_w;

    // allocate result matrix gpu memory
    // check if out mat allocated previously suits for the current query matrix
    uint64 out_mat_size = mat_w * MatAccEnv.fp32_base_mat_w;
    // if the currently allocated gpu out memory buffer smaller than the required
    if (MatAccEnv.fp32_out_mat_size<out_mat_size) {
      // check if had previously allocate gpu memory.  MatAccEnv.fp32_gpu_out_mat not equals to NULL, release the current memory
      if (MatAccEnv.fp32_gpu_out_mat != NULL) {
        cudaFree (MatAccEnv.fp32_gpu_out_mat);
      }

      cudaStat = cudaMalloc ((void**)&MatAccEnv.fp32_gpu_out_mat, out_mat_size*sizeof(fp32));
      if (cudaStat != cudaSuccess) {
        printf ("result GPU memory allocation failed\n");
        // if allocate gpu memory failed reset the gpu memory pointer to NULL
        MatAccEnv.fp32_gpu_out_mat = NULL;
        MatAccEnv.fp32_out_mat_h = 0;
        MatAccEnv.fp32_out_mat_w = 0;
        MatAccEnv.fp32_out_mat_size = 0;
        return CU_VIPR_MAT_ACC_ERR;
      }
      // reset the out mat size parameter

      MatAccEnv.fp32_out_mat_size = out_mat_size;
    }
    MatAccEnv.fp32_out_mat_w    = MatAccEnv.fp32_base_mat_w;
    MatAccEnv.fp32_out_mat_h    = mat_w;


    // set the query mat in gpu with data
    stat = cublasSetMatrix (mat_h, mat_w, sizeof(fp32), data, mat_h, MatAccEnv.fp32_gpu_query_mat, mat_h);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      switch(stat) {
      case CUBLAS_STATUS_INVALID_VALUE:
    	  printf ("data download failed: CUBLAS_STATUS_INVALID_VALUE");
    	  break;
      case CUBLAS_STATUS_MAPPING_ERROR:
    	  printf ("data download failed: CUBLAS_STATUS_MAPPING_ERROR");
    	  break;
      }

      return CU_VIPR_MAT_ACC_ERR;
    }
    // do the matmul in gpu
    float alpha = 1.0;
    float beta = 0.0;
    //printf("gemm\n");
    stat = cublasSgemm(    MatAccEnv.blas_handle,
        		           CUBLAS_OP_T, CUBLAS_OP_N,
						   MatAccEnv.fp32_query_mat_w, MatAccEnv.fp32_base_mat_w, MatAccEnv.fp32_query_mat_h,
    					   &alpha,
						   MatAccEnv.fp32_gpu_query_mat, MatAccEnv.fp32_query_mat_h,
						   MatAccEnv.fp32_gpu_base_mat, MatAccEnv.fp32_base_mat_h,
    					   &beta,
						   MatAccEnv.fp32_gpu_out_mat, MatAccEnv.fp32_out_mat_h);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("gemm failed\n");
      return CU_VIPR_MAT_ACC_ERR;
    }
    //printf("gemm finish\n");
    // get the result matrix from GPU
    stat = cublasGetMatrix (mat_w, MatAccEnv.fp32_base_mat_w,
    		                sizeof(fp32),
							MatAccEnv.fp32_gpu_out_mat, mat_w,
							result,                     mat_w);
    //printf("gemm finish a\n");
    if (stat != CUBLAS_STATUS_SUCCESS) {
        switch(stat) {
        case CUBLAS_STATUS_INVALID_VALUE:
      	  printf ("data upload failed: CUBLAS_STATUS_INVALID_VALUE\n");
      	  break;
        case CUBLAS_STATUS_MAPPING_ERROR:
      	  printf ("data upload failed: CUBLAS_STATUS_MAPPING_ERROR\n");
      	  break;
        }
        return CU_VIPR_MAT_ACC_ERR;
    }
    //printf("gemm finish 2\n");
    return CU_VIPR_MAT_ACC_OK;
}



int32 QueryDataBaseN_FP32(const int32 q_mat_h, const int32 q_mat_w,
		                  const fp32* q_mat,
						  const int32 N, const int32* data_base_N_idx, const int32 total_db_kp,
						  fp32* result)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    // allocate cuda memory for query data
    // check if query mat allocated previously suits for the current query matrix
    uint64 query_mat_size = q_mat_h*q_mat_w;
    // if the currently allocated gpu query memory buffer smaller than the required
    if  (MatAccEnv.fp32_query_mat_size<query_mat_size) {
      // check if had previously allocate gpu memory.  MatAccEnv.fp32_gpu_query_mat not equals to NULL, release the current memory
      if (MatAccEnv.fp32_gpu_query_mat != NULL) {
        cudaFree (MatAccEnv.fp32_gpu_query_mat);
      }
      //printf("Reallocate query memory\n");
      cudaStat = cudaMalloc ((void**)&MatAccEnv.fp32_gpu_query_mat, query_mat_size*sizeof(fp32));
      if (cudaStat != cudaSuccess) {
        printf ("query GPU memory allocation failed\n");
        // if allocate gpu memory failed reset the gpu memory pointer to NULL
        MatAccEnv.fp32_gpu_query_mat = NULL;
        MatAccEnv.fp32_query_mat_h = 0;
        MatAccEnv.fp32_query_mat_w = 0;
        MatAccEnv.fp32_query_mat_size = 0;
        return CU_VIPR_MAT_ACC_ERR;
      }
      // reset the query mat size parameter
      MatAccEnv.fp32_query_mat_size = query_mat_size;
    }
    MatAccEnv.fp32_query_mat_h    = q_mat_h;
    MatAccEnv.fp32_query_mat_w    = q_mat_w;

    // allocate result matrix gpu memory
    // calculate the required out matrix width dimension
    //int32 out_w=total_db_kp;
    //for (int32 i=0; i<N; i++){
    //  int32 db_idx = data_base_N_idx[i];
    //  out_w += MatAccEnv.db_each_frame_keypoint_num[db_idx];
    //}


    // check if out mat allocated previously suits for the current query matrix
    //printf ("out_w %d\n", out_w);
    uint64 out_mat_size = MatAccEnv.fp32_query_mat_w * total_db_kp;
    //printf ("%d\n",out_mat_size);
    // if the currently allocated gpu out memory buffer smaller than the required
    if (MatAccEnv.fp32_out_mat_size<out_mat_size) {
      // check if had previously allocate gpu memory.  MatAccEnv.fp32_gpu_out_mat not equals to NULL, release the current memory
      if (MatAccEnv.fp32_gpu_out_mat != NULL) {
        cudaFree (MatAccEnv.fp32_gpu_out_mat);
      }

      cudaStat = cudaMalloc ((void**)&MatAccEnv.fp32_gpu_out_mat, out_mat_size*sizeof(fp32));
      if (cudaStat != cudaSuccess) {
        printf ("result GPU memory allocation failed\n");
        // if allocate gpu memory failed reset the gpu memory pointer to NULL
        MatAccEnv.fp32_gpu_out_mat = NULL;
        MatAccEnv.fp32_out_mat_h = 0;
        MatAccEnv.fp32_out_mat_w = 0;
        MatAccEnv.fp32_out_mat_size = 0;
        return CU_VIPR_MAT_ACC_ERR;
      }
      // reset the out mat size parameter
      MatAccEnv.fp32_out_mat_size = out_mat_size;
    }
    MatAccEnv.fp32_out_mat_w    = total_db_kp;
    MatAccEnv.fp32_out_mat_h    = MatAccEnv.fp32_query_mat_w;


    // set the query mat in gpu with data
    stat = cublasSetMatrix (q_mat_h, q_mat_w, sizeof(fp32), q_mat, q_mat_h, MatAccEnv.fp32_gpu_query_mat, q_mat_h);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      switch(stat) {
      case CUBLAS_STATUS_INVALID_VALUE:
    	  printf ("data download failed: CUBLAS_STATUS_INVALID_VALUE");
    	  break;
      case CUBLAS_STATUS_MAPPING_ERROR:
    	  printf ("data download failed: CUBLAS_STATUS_MAPPING_ERROR");
    	  break;
      }

      return CU_VIPR_MAT_ACC_ERR;
    }
    // do the matmul in gpu
    float alpha = 1.0;
    float beta = 0.0;
    fp32* base_mat_start = NULL;
    fp32* out_mat_start  = MatAccEnv.fp32_gpu_out_mat;
    for (int32 i=0; i<N; i++) {
      int32 db_idx = data_base_N_idx[i];
      int32 db_frame_mat_w = MatAccEnv.db_each_frame_keypoint_num[db_idx];
      //printf ("%d, %d, %d\n",db_mat_w, q_mat_w, MatAccEnv.fp32_base_mat_h);
      //printf ("%d\n", q_mat_h);
      //printf ("%d\n", MatAccEnv.fp32_base_mat_h);
      //printf ("%d\n\n",out_w);
      //printf ("%d, %d\n", MatAccEnv.fp32_base_mat_h, MatAccEnv.db_frame_base_mat_start_col[db_idx]);


      base_mat_start = MatAccEnv.fp32_gpu_base_mat+MatAccEnv.fp32_base_mat_h*MatAccEnv.db_frame_base_mat_start_col[db_idx];

      //printf ("%d, %d, %d\n",base_mat_start, q_mat, out_mat_start);

      stat = cublasSgemm(MatAccEnv.blas_handle,
        		         CUBLAS_OP_T, CUBLAS_OP_N,
						 MatAccEnv.fp32_query_mat_w, db_frame_mat_w, MatAccEnv.fp32_query_mat_h,//db_mat_w, q_mat_w, MatAccEnv.fp32_base_mat_h,
                         &alpha,
						 MatAccEnv.fp32_gpu_query_mat, q_mat_h,
						 base_mat_start, MatAccEnv.fp32_base_mat_h,
						 &beta,
						 out_mat_start, MatAccEnv.fp32_out_mat_h //db_mat_w
						);
      if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("gemm failed\n");
        switch(stat) {
        case CUBLAS_STATUS_NOT_INITIALIZED:
      	  printf ("the library was not initialized: CUBLAS_STATUS_NOT_INITIALIZED\n");
      	  break;
        case CUBLAS_STATUS_INVALID_VALUE:
      	  printf ("CUBLAS_STATUS_INVALID_VALUE\n");
      	  break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
          printf ("the device does not support math in half precision.\n");
        case CUBLAS_STATUS_EXECUTION_FAILED:
          printf ("the function failed to launch on the GPU.\n");
        }
        return CU_VIPR_MAT_ACC_ERR;
      }
      out_mat_start  += db_frame_mat_w*MatAccEnv.fp32_query_mat_w;
    }

    // get the result matrix from GPU
    //printf ("%d, %d\n", MatAccEnv.fp32_query_mat_h, MatAccEnv.fp32_query_mat_w);
    //printf ("Get the result %d, %d--%d\n",MatAccEnv.fp32_out_mat_h, MatAccEnv.fp32_out_mat_w, MatAccEnv.fp32_gpu_out_mat);
    stat = cublasGetMatrix (MatAccEnv.fp32_out_mat_h, MatAccEnv.fp32_out_mat_w,
    		                sizeof(fp32),
							MatAccEnv.fp32_gpu_out_mat, MatAccEnv.fp32_out_mat_h,
							result,                     MatAccEnv.fp32_out_mat_h);
    //printf("gemm finish a\n");
    if (stat != CUBLAS_STATUS_SUCCESS) {
        switch(stat) {
        case CUBLAS_STATUS_INVALID_VALUE:
      	  printf ("data upload failed: CUBLAS_STATUS_INVALID_VALUE\n");
      	  break;
        case CUBLAS_STATUS_MAPPING_ERROR:
      	  printf ("data upload failed: CUBLAS_STATUS_MAPPING_ERROR\n");
      	  break;
        }
        return CU_VIPR_MAT_ACC_ERR;
    }
    //printf("gemm finish 2\n");
    return CU_VIPR_MAT_ACC_OK;
}

int32 NormalizeBatchVectorFP32(const int32 vect_dim, const int32 vect_num, const fp32* data, fp32* data_normlized)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;


    fp32* batch_norm_mem=NULL;
    uint64 vects_size =  vect_dim*vect_num;
    cudaStat = cudaMalloc ((void**)&batch_norm_mem, vects_size*sizeof(fp32));
    if (cudaStat != cudaSuccess) {
       printf ("Normalization vectors GPU memory allocation failed\n");
       // if allocate gpu memory failed reset the gpu memory pointer to NULL
       //MatAccEnv.fp32_norm_vects = NULL;
       return CU_VIPR_MAT_ACC_ERR;
    }
    //MatAccEnv.fp32_vect_dim = vect_dim;
    //MatAccEnv.fp32_vect_num = vect_num;

    stat = cublasSetMatrix (vect_dim, vect_num, sizeof(fp32), data, vect_dim, batch_norm_mem, vect_dim);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      switch(stat) {
      case CUBLAS_STATUS_INVALID_VALUE:
    	  printf ("data download failed: CUBLAS_STATUS_INVALID_VALUE\n");
    	  break;
      case CUBLAS_STATUS_MAPPING_ERROR:
    	  printf ("data download failed: CUBLAS_STATUS_MAPPING_ERROR\n");
    	  break;
      }
      cudaFree(batch_norm_mem);
      return CU_VIPR_MAT_ACC_ERR;
    }

    // calculate the norm2 of each vector
    fp32 norm[vect_num];
    for (uint32 i=0; i<vect_num; i++) {
        stat = cublasSnrm2(
                     MatAccEnv.blas_handle,
					 vect_dim,             // vector dimension
					 batch_norm_mem+(i*vect_dim),    //
					 1,                    // stride
					 norm+i                // out norm value in host
               );
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("Calculate norm2 failed\n");
            cudaFree(batch_norm_mem);
            return CU_VIPR_MAT_ACC_ERR;
        }
        norm[i] = 1.0/norm[i];
    }
    // normalize the batch vector
    for (uint32 i=0; i<vect_num; i++) {
    	stat = cublasSscal(
    			MatAccEnv.blas_handle,
				vect_dim,
				&norm[i],
				batch_norm_mem+(i*vect_dim),
				1);
    }
    stat = cublasGetMatrix (vect_dim, vect_num,
        		            sizeof(fp32),
							batch_norm_mem, vect_dim,
							data_normlized,  vect_dim);
    //printf("gemm finish a\n");
    if (stat != CUBLAS_STATUS_SUCCESS) {
        switch(stat) {
        case CUBLAS_STATUS_INVALID_VALUE:
          	printf ("data upload failed: CUBLAS_STATUS_INVALID_VALUE\n");
          	break;
        case CUBLAS_STATUS_MAPPING_ERROR:
          	printf ("data upload failed: CUBLAS_STATUS_MAPPING_ERROR\n");
          	break;
        }
        return CU_VIPR_MAT_ACC_ERR;
    }
    cudaFree(batch_norm_mem);
    return CU_VIPR_MAT_ACC_OK;


}



void CuVPRAccMat::SetMatrixDtypeCount(const cudaDataType cuDtype, const DtypeCount_t bytecount)
{
	this->bytecount = bytecount;
	this->cuDtype = cuDtype;
}
DtypeCount_t CuVPRAccMat::GetMatrixDtypeCount(void)
{
	return this->bytecount;
}
uint8 CuVPRAccMat::allocMatGpuMem(const MatDim_t h, const MatDim_t w)
{
	uint32 bytes = h*w*this->bytecount;
	//uint8 reminder = bytes%16;
	//if (reminder!=0)
	//  bytes+=16-reminder;
	if (cudaMalloc(reinterpret_cast<void**>(&(this->mat)), bytes) == cudaSuccess) {
		//printf ("Successfully allocate DB GPU memory.\n");

	    this->mat_size = h*w;
	    this->mem_entry_size = this->mat_size;
	    this->mat_next_fill_line = 0;

	    this->mat_h = h;
		this->mat_w = w;
	    return CU_VPR_ACC_SUCCESS;
	} else{
		printf("Failed to allocate GPU memory.\n");
		return CU_VPR_ACC_FAILED;
	}
}

uint8 CuVPRAccMat::ReleaseMatGpuMem(void)
{
    // check if the matrix gpu memory is NULL
	if (this->mat == (void*)0) {
		this->mat_next_fill_line = 0;
		this->mat_h = 0;
		this->mat_w = 0;
		this->mat_size = 0;
		this->mem_entry_size = 0;
		return CU_VPR_ACC_SUCCESS;
	}
	if (cudaFree(this->mat)==cudaSuccess) {
		this->mat_next_fill_line = 0;
		this->mat_h = 0;
		this->mat_w = 0;
		this->mat_size = 0;
		this->mem_entry_size = 0;
		this->mat = (void*)0;
		return CU_VPR_ACC_SUCCESS;
	} else {
		printf("Unable to release GPU memory.\n");
		return CU_VPR_ACC_FAILED;
	}
}
uint8 CuVPRAccMat::DynamicFillMatGpuData(const MatDim_t h, const MatDim_t w, const void* data)
{
    if (h*w!=this->mem_entry_size) {
    	this->ReleaseMatGpuMem();       // release GPU memory
    	this->allocMatGpuMem(h,w);      // allocate new GPU memory
    	this->SetGpuMatDesc(h, w);      // alter the GPU mat desc
    }
    // fill data
    mat_next_fill_line = 0; // mat reset next fill line
	if (mat_data_order==ROW_MAJOR) this->AttachDataToMat(h, data);
	else this->AttachDataToMat(w, data);
    /*else {
    	this->mat_w = w;
    	this->mat_h = h;
    	this->mat_size = w*h;
    	this->AssignMatData(h, data);
    	this->SetGpuMatDesc(h, w);
    }*/

    return CU_VPR_ACC_SUCCESS;
}
uint8 CuVPRAccMat::DynamicReshapeMatGpu(const MatDim_t h, const MatDim_t w)
{
    if (h*w!=this->mem_entry_size) {
    	this->ReleaseMatGpuMem();       // release GPU memory
    	this->allocMatGpuMem(h,w);      // allocate new GPU memory
    	this->SetGpuMatDesc(h, w);      // alter the GPU mat desc
    }
    /*else {
    	this->mat_w = w;
    	this->mat_h = h;
    	this->mat_size = w*h;
    	this->SetGpuMatDesc(h, w);
    }*/

    return CU_VPR_ACC_SUCCESS;
}
uint8 CuVPRAccMat::allocGpuMatDesc(void)
{
	if (cublasLtMatrixLayoutCreate(&(this->mat_desc), this->cuDtype, 0, 0, 0)==CUBLAS_STATUS_ALLOC_FAILED) {
		printf("Failed to create cublasLtMatrixLayout_t by function cublasLtMatrixLayoutCreate.\n");
		return CU_VPR_ACC_FAILED;
	}

	__CheckStatus(cublasLtMatrixLayoutInit(this->mat_desc, this->cuDtype, 0,0,0), "CuVPRAccMat::allocGpuMatDesc");
    mat_data_order = ROW_MAJOR;
	cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
	if (cublasLtMatrixLayoutSetAttribute(this->mat_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order))==CUBLAS_STATUS_SUCCESS) {
	    return CU_VPR_ACC_SUCCESS;
	} else {
		printf("Failed in cublasLtMatrixLayoutSetAttribute.\n");
		return CU_VPR_ACC_FAILED;
	}
}

uint8 CuVPRAccMat::SetGpuMatDescLayoutOrder(MatDataOrder layout_order)
{
	uint8 state=0;
	int32_t order;
	if (layout_order == ROW_MAJOR)
	    order = CUBLASLT_ORDER_ROW;
	else
	    order = CUBLASLT_ORDER_COL;
	mat_data_order = layout_order;
	state+=__CheckStatus(cublasLtMatrixLayoutSetAttribute(this->mat_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)), "CuVPRAccMat::SetGpuMatDescLayoutOrder");
	if (state!=0)
	  return CU_VPR_ACC_FAILED;
	else
	  return CU_VPR_ACC_SUCCESS;
}
uint8 CuVPRAccMat::SetGpuMatDesc(const MatDim_t h, const MatDim_t w)
{
	uint8 state=0;
	MatDim_t ld = GetMatLD();
	state+=__CheckStatus(cublasLtMatrixLayoutSetAttribute(this->mat_desc, CUBLASLT_MATRIX_LAYOUT_ROWS, &h, sizeof(h)),   "CuVPRAccMat::SetGpuMatDesc");
	state+=__CheckStatus(cublasLtMatrixLayoutSetAttribute(this->mat_desc, CUBLASLT_MATRIX_LAYOUT_COLS, &w, sizeof(w)),   "CuVPRAccMat::SetGpuMatDesc");
	state+=__CheckStatus(cublasLtMatrixLayoutSetAttribute(this->mat_desc, CUBLASLT_MATRIX_LAYOUT_LD,   &ld, sizeof(ld)),   "CuVPRAccMat::SetGpuMatDesc");
	if (state!=0)
	  return CU_VPR_ACC_FAILED;
	else
	  return CU_VPR_ACC_SUCCESS;
}
uint8 CuVPRAccMat::allocMatCpuMem(const MatDim_t h, const MatDim_t w)
{
	this->mat = reinterpret_cast<void*>(malloc(this->bytecount*h*w));
	if (this->mat==NULL) {
	  return CU_VPR_ACC_FAILED;
	}
	this->mat_h = h;
	this->mat_w = w;
	this->mat_size = h*w;
	return CU_VPR_ACC_SUCCESS;
}


uint8 CuVPRAccMat::AttachDataToMat(const MatDim_t dim, const void* mat_data)
{
	MatDim_t ld = GetMatLD();
	// check if rest of the GPU memory able to contain the entire mat_data
	if ((mat_next_fill_line+dim)*ld>mem_entry_size) {
		printf("DB matrix attachment overrun! ld:%d,  dim:%d,  mat_next_fill_line:%d,  mem_entry_size:%d\n", ld, dim, mat_next_fill_line, mem_entry_size);
		return CU_VPR_ACC_FAILED;
	}

	void *destPtr = reinterpret_cast<void*>(reinterpret_cast<uint8*>(mat)+ (mat_next_fill_line*ld)*bytecount);
	uint64 cpy_size = dim*ld*bytecount;
	if (cudaMemcpy(destPtr, mat_data, cpy_size, cudaMemcpyHostToDevice)==cudaSuccess) {
		mat_next_fill_line += dim;
		return CU_VPR_ACC_SUCCESS;
	} else {
		printf("Transfer data to GPU memory failed.\n");
		return CU_VPR_ACC_FAILED;
	}
}

template <typename T> T CuVPRAccMat::Get(MatDim_t H, MatDim_t W)
{
	return *(reinterpret_cast<T*>(this->mat)+W+H*this->mat_w);
}
template <typename T> void CuVPRAccMat::Set(const MatDim_t h, const MatDim_t w, T value)
{
    T* ptr = (T*)(this->mat)+w+h*this->mat_w;
    *ptr = value;
}
MatDim_t CuVPRAccMat::GetMatHeight(void)
{
	return this->mat_h;
}
MatDim_t CuVPRAccMat::GetMatWidth(void)
{
	return this->mat_w;
}
MatDim_t CuVPRAccMat::GetMatLD(void)
{
	if (mat_data_order==ROW_MAJOR) return mat_w;
	else return mat_h;
}
void* CuVPRAccMat::GetMatPtr(void)
{
	return this->mat;
}
uint8 CuVPRAccMat::GetGpuData(void *data)
{
	uint64 cpy_size = this->mem_entry_size*this->bytecount;
	cudaError_t err;
	err = cudaMemcpy(data, this->mat, cpy_size, cudaMemcpyDeviceToHost);
	switch(err) {
	case cudaSuccess:
		return CU_VPR_ACC_SUCCESS;
		break;
	case cudaErrorInvalidValue:
        printf ("cudaErrorInvalidValue. \n");
        return CU_VPR_ACC_FAILED;
        break;
	case cudaErrorInvalidMemcpyDirection:
    	printf("cudaErrorInvalidMemcpyDirection.\n");
		return CU_VPR_ACC_FAILED;
		break;
	default:
		printf("cudaMemcpy unknow erro code %d\n", err);
		return CU_VPR_ACC_FAILED;
	}
}

uint8 CuVPRAccMat::GetMatGpuDataParitalRow(void* data, const MatDim_t partial_row_length)
{
	{
		uint64 cpy_size = this->mem_entry_size*this->bytecount;
		uint64 row_size_bytes = partial_row_length*this->bytecount;
		cudaError_t err;
		if (this->mat_data_order == COL_MAJOR)
		  err = cudaMemcpy2D(data, row_size_bytes,
					       this->mat, this->mat_h*this->bytecount,
						   row_size_bytes, this->mat_w, cudaMemcpyDeviceToHost);
		else
		  err = cudaMemcpy2D(data, row_size_bytes,
				           this->mat, this->mat_w*this->bytecount,
						   row_size_bytes, this->mat_h, cudaMemcpyDeviceToHost);
		switch(err) {
		case cudaSuccess:
			return CU_VPR_ACC_SUCCESS;
			break;
		case cudaErrorInvalidValue:
	        printf ("cudaErrorInvalidValue. \n");
	        return CU_VPR_ACC_FAILED;
	        break;
		case cudaErrorInvalidMemcpyDirection:
	    	printf("cudaErrorInvalidMemcpyDirection.\n");
			return CU_VPR_ACC_FAILED;
			break;
		default:
			printf("cudaMemcpy unknow erro code %d\n", err);
			return CU_VPR_ACC_FAILED;
		}
	}
}

void CuVPRAccMat::dbg_PrintMember(void)
{
	uint64_t rows, cols, ld;
	cublasLtOrder_t storage_order;
	cudaDataType_t dtp;
	size_t ret;
	//CUBLASLT_MATRIX_LAYOUT_COLS
	//CUBLASLT_MATRIX_LAYOUT_LD
	__CheckStatus(cublasLtMatrixLayoutGetAttribute(mat_desc, CUBLASLT_MATRIX_LAYOUT_ROWS, &rows, sizeof(rows), &ret),"CuVPRAccMat::dbg_PrintMember");
	__CheckStatus(cublasLtMatrixLayoutGetAttribute(mat_desc, CUBLASLT_MATRIX_LAYOUT_COLS, &cols, sizeof(cols), &ret),"CuVPRAccMat::dbg_PrintMember");
	__CheckStatus(cublasLtMatrixLayoutGetAttribute(mat_desc, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld), &ret),"CuVPRAccMat::dbg_PrintMember");
	__CheckStatus(cublasLtMatrixLayoutGetAttribute(mat_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &storage_order, sizeof(storage_order), &ret),"CuVPRAccMat::dbg_PrintMember");
	__CheckStatus(cublasLtMatrixLayoutGetAttribute(mat_desc, CUBLASLT_MATRIX_LAYOUT_TYPE, &dtp, sizeof(dtp), &ret),"CuVPRAccMat::dbg_PrintMember");
	printf("mat: %x, mat_h:%d, mat_w:%d, mem_entry_size:%d \n",mat, mat_h, mat_w, mem_entry_size);
	printf("desc: %x, desc h:%d,  desc w:%d  desc ld:%d  \n",mat_desc, rows, cols, ld);
	char *s;
	switch(storage_order) {
	case CUBLASLT_ORDER_COL:
	   s="CUBLASLT_ORDER_COL";
	   break;
	case CUBLASLT_ORDER_ROW:
	   s="CUBLASLT_ORDER_ROW";
	   break;
	case CUBLASLT_ORDER_COL32:
	   s="CUBLASLT_ORDER_COL32";
	   break;
	case CUBLASLT_ORDER_COL4_4R2_8C:
	   s="CUBLASLT_ORDER_COL4_4R2_8C";
	   break;
	case CUBLASLT_ORDER_COL32_2R_4R4:
	   s="CUBLASLT_ORDER_COL32_2R_4R4";
	   break;

	}
	printf("mat storage order: %s \n",s);


	switch(dtp) {
	case CUDA_R_32F:
		s = "CUDA_R_32F";
		break;
	case CUDA_R_8F_E4M3:
		s = "CUDA_R_8F_E4M3";
		break;
	case CUDA_R_8F_E5M2:
		s = "CUDA_R_8F_E5M2";
		break;
	default:
		s="not listed datatype";
	}
	printf("mat data type: %s \n",s);

	printf ("\n");
}
template <typename T> void CuVPRAccMat::dbg_PrintMat(uint64 top)
{
	uint64 size = mem_entry_size;
	T data[size];
	if (GetGpuData((void*)data)==CU_VPR_ACC_FAILED) {
		printf("dbg_PrintMat: Failed to read matrix data from GPU memory.\n");
	    return;
	}
	if (top>size)
		top = size;
	for (uint64 i=0; i<top; i++) {
	  if ((i%8)==0)
		  printf ("\n");
	  std::cout<<data[i]<<",";
	}
	printf ("\n");
}

cuVPRMatAccEnv::cuVPRMatAccEnv(const cudaDataType cuDtype, const DtypeCount_t bytecount)
{

	if (cublasLtCreate(&(this->ltHandle))==CUBLAS_STATUS_SUCCESS) {
		printf("Successfully create cublasLt handle.\n");
	} else while(1);
	if (cublasLtMatmulDescCreate(&(this->operationDesc), CUBLAS_COMPUTE_32F, CUDA_R_32F)==CUBLAS_STATUS_SUCCESS) {
	    printf("Successfully create operationDesc.\n");
	} else while(1);
	TRANSPOSE    = CUBLAS_OP_T;
	NO_TRANSPOSE = CUBLAS_OP_N;
    db_frame_num = 0;

	this->DB_hol_mat.SetMatrixDtypeCount(CUDA_R_32F, FP32_BYTE_COUNT);
	this->DB_hol_mat.allocGpuMatDesc();

	this->query_hol_mat.SetMatrixDtypeCount(CUDA_R_32F, FP32_BYTE_COUNT);
	this->query_hol_mat.allocGpuMatDesc();

	this->hol_query_result_mat.SetMatrixDtypeCount(CUDA_R_32F, FP32_BYTE_COUNT);
	this->hol_query_result_mat.allocGpuMatDesc();

	this->DB_local_feature_mat.SetMatrixDtypeCount(cuDtype, bytecount);
	this->DB_local_feature_mat.allocGpuMatDesc();
	this->DB_local_feature_mat.SetGpuMatDescLayoutOrder(COL_MAJOR);

	this->query_local_feature_mat.SetMatrixDtypeCount(cuDtype, bytecount);
	this->query_local_feature_mat.allocGpuMatDesc();
	this->query_local_feature_mat.SetGpuMatDescLayoutOrder(COL_MAJOR);

	this->local_feature_query_result_mat.SetMatrixDtypeCount(CUDA_R_32F, FP32_BYTE_COUNT);
	this->local_feature_query_result_mat.allocGpuMatDesc();
	this->local_feature_query_result_mat.SetGpuMatDescLayoutOrder(COL_MAJOR);

	this->db_each_frame_keypoint_num_mat.SetMatrixDtypeCount(CUDA_R_32I, UINT32_BYTE_COUNT);
	this->db_frame_base_mat_start_col_mat.SetMatrixDtypeCount(CUDA_R_32I, UINT32_BYTE_COUNT);

	/*** Initialization the matmul operation ***/
    __CheckStatus(cublasLtMatmulDescCreate(&HolFeatureQueryDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "Create hol matmul desc");
    __CheckStatus(cublasLtMatmulDescSetAttribute(HolFeatureQueryDesc, CUBLASLT_MATMUL_DESC_TRANSA, &NO_TRANSPOSE, sizeof(NO_TRANSPOSE)), "hol matmul desc set attribute");
    __CheckStatus(cublasLtMatmulDescSetAttribute(HolFeatureQueryDesc, CUBLASLT_MATMUL_DESC_TRANSB, &TRANSPOSE,    sizeof(TRANSPOSE)), "hol matmul desc set attribute");
    __CheckStatus(cublasLtMatmulPreferenceCreate(&HolFeatureQueryMatmul_preference), "Create hol matmul preference");
    //__CheckStatus(cublasLtMatmulPreferenceSetAttribute(HolFeatureQueryMatmul_preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &holMatmulWS_size, sizeof(holMatmulWS_size)), "hol matmul preference set attribute");



	switch (cuDtype) {
	case CUDA_R_8F_E4M3:
	case CUDA_R_8F_E5M2:
	case CUDA_R_32F:
		__CheckStatus(cublasLtMatmulDescCreate(&LocalFeatureQueryDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F),"Create loc feature query mat desc");
		break;
	case CUDA_R_8I:
		__CheckStatus(cublasLtMatmulDescCreate(&LocalFeatureQueryDesc, CUBLAS_COMPUTE_32I, cuDtype),"Create loc feature query mat desc");
		break;
	}
	__CheckStatus(cublasLtMatmulDescSetAttribute(LocalFeatureQueryDesc, CUBLASLT_MATMUL_DESC_TRANSA, &TRANSPOSE, sizeof(TRANSPOSE)),"Set LocalFeatureQueryDesc CUBLASLT_MATMUL_DESC_TRANSA");
	__CheckStatus(cublasLtMatmulDescSetAttribute(LocalFeatureQueryDesc, CUBLASLT_MATMUL_DESC_TRANSB, &NO_TRANSPOSE,sizeof(NO_TRANSPOSE)),"Set LocalFeatureQueryDesc CUBLASLT_MATMUL_DESC_TRANSB");
	__CheckStatus(cublasLtMatmulPreferenceCreate(&LocalFeatureQueryMatmul_preference),"Create LocalFeatureQueryMatmul_preference");
	int32 A_align = 64;
	__CheckStatus(cublasLtMatmulPreferenceSetAttribute(LocalFeatureQueryMatmul_preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &A_align, sizeof(A_align)),"Create LocalFeatureQueryMatmul_preference");

}

void cuVPRMatAccEnv::GetPtrAttrbute(void* ptr)
{
	cudaPointerAttributes attributes;
	cudaPointerGetAttributes ( &attributes, ptr );
	printf("Ptr=%x --->",ptr);
	switch (attributes.type) {
	case cudaMemoryTypeUnregistered:
		printf("cudaMemoryTypeUnregistered\n");
	break;
	case cudaMemoryTypeHost:
		printf("cudaMemoryTypeHost\n");
	break;
	case cudaMemoryTypeDevice:
		printf("cudaMemoryTypeDevice\n");
	break;
	case cudaMemoryTypeManaged:
		printf("cudaMemoryTypeManaged\n");
	break;
    }
}
int32 cuVPRMatAccEnv::AllocateDB_LocalFeature(const MatDim_t mat_h,  const MatDim_t mat_w, const uint32 db_frame_num)
{
    /* allocate cpu memory to hole the number of keypoint for each frame in db */
    this->db_frame_num = db_frame_num;
    if (this->db_each_frame_keypoint_num_mat.allocMatCpuMem(1, this->db_frame_num)==CU_VPR_ACC_FAILED) {
        printf("Allocate CPU memory for db_each_frame_keypoint_num failed!\n");
        return CU_VPR_ACC_FAILED;
    }
    printf ("Successfully allocate memory for db_each_frame_keypoint_num_mat.\n");
    if (this->db_frame_base_mat_start_col_mat.allocMatCpuMem(1, this->db_frame_num)==CU_VPR_ACC_FAILED) {
        printf("Allocate CPU memory for db_frame_base_mat_start_col failed!\n");
        return CU_VPR_ACC_FAILED;
    }
    printf ("Successfully allocate memory for db_frame_base_mat_start_col_mat.\n");

    /* allocate DB local feature GPU memory */
    this->DB_local_feature_mat.ReleaseMatGpuMem();   // first try to release the GPU memory
    if (this->DB_local_feature_mat.allocMatGpuMem(mat_h, mat_w)==CU_VPR_ACC_FAILED) {
    	printf("Allocate GPU memory for DB_local_feature_mat failed!\n");
        return CU_VPR_ACC_FAILED;
    }
    //DB_local_feature_mat.dbg_PrintMember();
    // set DB local feature matrix layout desc
	if (this->DB_local_feature_mat.SetGpuMatDesc(mat_h, mat_w)==CU_VPR_ACC_FAILED) {
		printf("Failed to Init DB local feature matrix desc!\n");
		return CU_VPR_ACC_FAILED;
	}
	printf ("Successfully allocate memory for DB_local_feature_mat.\n");

    return CU_VPR_ACC_SUCCESS;
}
int32 cuVPRMatAccEnv::AllocateDB_HolisticFeature(const MatDim_t feature_num, const MatDim_t feature_dim)
{
	// allocate GPU memory for the DB holistic feature
	if (this->DB_hol_mat.allocMatGpuMem(feature_num, feature_dim) == CU_VPR_ACC_FAILED) {
		printf("Allocate GPU memory failed for Holistic feature! \n");
		return CU_VPR_ACC_FAILED;
	}
	if (this->DB_hol_mat.SetGpuMatDesc(feature_num, feature_dim)==CU_VPR_ACC_FAILED) {
		printf("Failed to Init DB holistic feature matrix desc!\n");
		return CU_VPR_ACC_FAILED;
	}
    return CU_VPR_ACC_SUCCESS;
}
int32 cuVPRMatAccEnv::SetDBInfo(const int32* db_frame_keypoint_num, const int32* db_frame_base_mat_start_col)
{
    for (uint32 i=0; i<this->db_frame_num; i++) {
    	this->db_each_frame_keypoint_num_mat.Set<int32>(0, i, *(db_frame_keypoint_num+i));
    	this->db_frame_base_mat_start_col_mat.Set<int32>(0,i, *(db_frame_base_mat_start_col+i));
    }
    return CU_VPR_ACC_SUCCESS;
}

int32 cuVPRMatAccEnv::AttachLocalFeatureToDB(const MatDim_t kp_num,  const MatDim_t kp_dim,  const void* data,
		                     const int32 batch_db_frame_num, const int32* batch_db_frame_keypoint_num)
{
    return this->DB_local_feature_mat.AttachDataToMat(kp_num, data);
}

int32 cuVPRMatAccEnv::AttachHolFeatureToDB(const MatDim_t hol_f_num,  const MatDim_t hol_f_dim,  const void* data)
{
	return DB_hol_mat.AttachDataToMat(hol_f_num, data);
}
int32 cuVPRMatAccEnv::HolQuery(const MatDim_t hol_f_num,  const MatDim_t hol_f_dim,  const void* data, fp32* result)
{
	/*** fill the query_hol_mat with the query frame holistic feature ***/
	if (this->query_hol_mat.DynamicFillMatGpuData(hol_f_num, hol_f_dim, data)==CU_VPR_ACC_FAILED) {
		printf("failed to fill hol query matrix!\n");
		return CU_VPR_ACC_FAILED;
	}
	/** Reshape the hol_query_result_mat**/
	if (this->hol_query_result_mat.DynamicReshapeMatGpu(hol_f_num, this->DB_hol_mat.GetMatHeight())==CU_VPR_ACC_FAILED) {
		printf("failed to reshape hol_query_result_mat!\n");
		return CU_VPR_ACC_FAILED;
	}

	/****** perform holistic feature query *******/
	// get the matmul heuristic
	query_hol_mat.dbg_PrintMember();
	DB_hol_mat.dbg_PrintMember();
	hol_query_result_mat.dbg_PrintMember();


	cublasLtMatmulHeuristicResult_t HolFeatureQueryMatmul_heuristicResult = {};
	int32 returnedResults = 0;
	cublasLtMatmulAlgoGetHeuristic(this->ltHandle, this->HolFeatureQueryDesc,
			                       this->query_hol_mat.GetCuMatDesc(),
								   this->DB_hol_mat.GetCuMatDesc(),
								   this->hol_query_result_mat.GetCuMatDesc(),
								   this->hol_query_result_mat.GetCuMatDesc(),
								   this->HolFeatureQueryMatmul_preference, 1, &HolFeatureQueryMatmul_heuristicResult,
								   &returnedResults);
	if (returnedResults==0) {
		printf("Unable to get Matmul algo heuristic!\n");
		return CU_VPR_ACC_FAILED;
	}
	// perform matmul
	cublasStatus_t state;
	fp32 alpha = 1.0;
	fp32 beta  = 0.0;
	state = cublasLtMatmul(this->ltHandle, this->HolFeatureQueryDesc,
			               &alpha,
						   this->query_hol_mat.GetMatPtr(), this->query_hol_mat.GetCuMatDesc(),    // mat A
						   this->DB_hol_mat.GetMatPtr(),    this->DB_hol_mat.GetCuMatDesc(),       // mat B

						   &beta,
						   nullptr, this->hol_query_result_mat.GetCuMatDesc(),  // mat C
						   this->hol_query_result_mat.GetMatPtr(), this->hol_query_result_mat.GetCuMatDesc(),  // mat D

						   &HolFeatureQueryMatmul_heuristicResult.algo,

						   nullptr, 0,//holMatmulWS, holMatmulWS_size,
						   0);
    if (state != CUBLAS_STATUS_SUCCESS) {
      printf ("gemm failed\n");
      switch(state) {
      case CUBLAS_STATUS_NOT_INITIALIZED:
    	  printf ("the library was not initialized: CUBLAS_STATUS_NOT_INITIALIZED\n");
    	  break;
      case CUBLAS_STATUS_INVALID_VALUE:
    	  printf ("CUBLAS_STATUS_INVALID_VALUE\n");
    	  break;
      case CUBLAS_STATUS_ARCH_MISMATCH:
          printf ("the device does not support math in half precision.\n");
      case CUBLAS_STATUS_EXECUTION_FAILED:
          printf ("the function failed to launch on the GPU.\n");
      }
      return CU_VPR_ACC_FAILED;
    }
    /*** Get the result back to cpu memory***/
    hol_query_result_mat.GetGpuData((void*)result);
	return CU_VPR_ACC_SUCCESS;
}
int32 cuVPRMatAccEnv::Query(const MatDim_t loc_feature_dim, const MatDim_t loc_feature_num, const void* data, fp32* result)
{
	/*** fill the query_local_feature_mat with the query frame local feature ***/
	// only works when loc_feature_num_rectified%4 == 0
	uint8 reminder = loc_feature_num%4;
	MatDim_t loc_feature_num_rectified = loc_feature_num;
	if (reminder!=0)
	  loc_feature_num_rectified += (4-reminder);
	if (this->query_local_feature_mat.DynamicFillMatGpuData(loc_feature_dim, loc_feature_num_rectified, data)==CU_VPR_ACC_FAILED) {
		printf("failed to fill local feature query matrix!\n");
		return CU_VPR_ACC_FAILED;
	}
	/** Reshape the local_feature_query_result_mat**/
	if (this->local_feature_query_result_mat.DynamicReshapeMatGpu(loc_feature_num_rectified, this->DB_local_feature_mat.GetMatWidth())==CU_VPR_ACC_FAILED) {
		printf("failed to reshape hol_query_result_mat!\n");
		return CU_VPR_ACC_FAILED;
	}
	/****** perform local feature query *******/
	// get the matmul heuristic
	//query_local_feature_mat.dbg_PrintMember();
	//DB_local_feature_mat.dbg_PrintMember();
	//local_feature_query_result_mat.dbg_PrintMember();
	cublasLtMatmulHeuristicResult_t LocalFeatureQueryMatmul_heuristicResult = {};
	int32 returnedResults = 0;
	cublasStatus_t state;
	state = cublasLtMatmulAlgoGetHeuristic(this->ltHandle, this->LocalFeatureQueryDesc,

								   this->query_local_feature_mat.GetCuMatDesc(),
								   this->DB_local_feature_mat.GetCuMatDesc(),
								   this->local_feature_query_result_mat.GetCuMatDesc(),
								   this->local_feature_query_result_mat.GetCuMatDesc(),
								   this->LocalFeatureQueryMatmul_preference, 1, &LocalFeatureQueryMatmul_heuristicResult,
								   &returnedResults);
	//printf("%d\n", state);
	if (returnedResults==0) {
		printf("Unable to get Matmul algo heuristic!\n");
		return CU_VPR_ACC_FAILED;
	}
	// perform matmul
	fp32 alpha = 1.0;
	fp32 beta  = 0.0;
	state = cublasLtMatmul(this->ltHandle, this->LocalFeatureQueryDesc,
			               &alpha,
						   this->query_local_feature_mat.GetMatPtr(),    this->query_local_feature_mat.GetCuMatDesc(),       // mat A
						   this->DB_local_feature_mat.GetMatPtr(), this->DB_local_feature_mat.GetCuMatDesc(),    // mat B

						   &beta,
						   nullptr, this->local_feature_query_result_mat.GetCuMatDesc(),  // mat C
						   this->local_feature_query_result_mat.GetMatPtr(), this->local_feature_query_result_mat.GetCuMatDesc(),  // mat D

						   &LocalFeatureQueryMatmul_heuristicResult.algo,

						   nullptr, 0,
						   0);
    if (state != CUBLAS_STATUS_SUCCESS) {
      printf ("gemm failed\n");
      switch(state) {
      case CUBLAS_STATUS_NOT_INITIALIZED:
    	  printf ("the library was not initialized: CUBLAS_STATUS_NOT_INITIALIZED\n");
    	  break;
      case CUBLAS_STATUS_INVALID_VALUE:
    	  printf ("CUBLAS_STATUS_INVALID_VALUE\n");
    	  break;
      case CUBLAS_STATUS_ARCH_MISMATCH:
          printf ("the device does not support math in half precision.\n");
      case CUBLAS_STATUS_EXECUTION_FAILED:
          printf ("the function failed to launch on the GPU.\n");
      }
      return CU_VPR_ACC_FAILED;
    }
    /*** Get the result back to cpu memory***/
    //local_feature_query_result_mat.GetGpuData((void*)result);
    local_feature_query_result_mat.GetMatGpuDataParitalRow((void*)result, loc_feature_num);
	return CU_VPR_ACC_SUCCESS;
}
int32 cuVPRMatAccEnv::QueryN(const MatDim_t q_mat_h, const MatDim_t q_mat_w,
                            const void* q_mat,
	                        const int32 N, const int32* data_base_N_idx, const int32 total_db_kp,
	                        fp32* result)
{
    // Check if the query data can fit in query_local_feature_mat

}

uint8 __CheckStatus(cublasStatus_t state, const char* str)
{
	if (state==CUBLAS_STATUS_SUCCESS) {
		//printf("%s: cublasStatus_t success. \n",str);
		return 0;
	} else {
		printf("*********************%s: cublasStatus_t error code:%d**************************\n", str,state);
		return 1;
	}
}


static cuVPRMatAccEnv *hVprAccEnv;
/*APIs that use the CublasLt to acc the VPR */
void InitMatAccExEnv(const uint32 feature_datatype)
{
	cudaDataType cuDtype;
	DtypeCount_t bytecount;
	switch (feature_datatype) {
    case DTYP_FP8_E4M3:
    	cuDtype   = CUDA_R_8F_E4M3;
    	bytecount = 1;
    	break;
    case DTYP_FP8_E5M2:
    	cuDtype   = CUDA_R_8F_E5M2;
    	bytecount = 1;
    	break;
    case DTYP_FP32:
    	cuDtype   = CUDA_R_32F;
    	bytecount = 4;
    	break;
    case DTYP_INT8:
    	cuDtype   = CUDA_R_8I;
    	bytecount = 1;
		break;
	default:
		printf("Undefined data type!\n");
		break;
	}
	hVprAccEnv = new cuVPRMatAccEnv(cuDtype, bytecount);
	printf ("MatAccExEnv successfully initialized!\n");
	int attr_value;
	cudaDeviceGetAttribute ( &attr_value, cudaDevAttrComputeMode, 0 );
	printf("GPU running in %d Mode\n",attr_value);
}
int32 malloDataBaseMat(const int32 mat_h,  const int32 mat_w, const int32 db_frame_num)
{
	return hVprAccEnv->AllocateDB_LocalFeature(mat_w, mat_h, db_frame_num);
}
int32 SetDBInfo(const int32 *db_frame_keypoint_num, const int32 *db_frame_base_mat_start_col)
{
	return hVprAccEnv->SetDBInfo(db_frame_keypoint_num, db_frame_base_mat_start_col);
}
int32 BatchAttachDataBaseMatrix(const MatDim_t mat_h,  const MatDim_t mat_w,  const uint8 *data,
		                        const int32 batch_db_frame_num, const int32* batch_db_frame_keypoint_num)
{
    return hVprAccEnv->AttachLocalFeatureToDB(mat_h, mat_w, (const void*)data, batch_db_frame_num, batch_db_frame_keypoint_num);
}
int32 BatchAttachHolFeature(const MatDim_t hol_f_num,  const MatDim_t hol_f_dim,  const uint8 *data)
{
    if (hVprAccEnv->AllocateDB_HolisticFeature(hol_f_num, hol_f_dim)==CU_VPR_ACC_FAILED) {
    	return CU_VPR_ACC_FAILED;
    }
	return hVprAccEnv->AttachHolFeatureToDB(hol_f_num, hol_f_dim, (const void*)data);
}

int32 BatchQueryHolisticFeature(const int32 mat_h,  const int32 mat_w,  const uint8* data, fp32* result)
{
	return hVprAccEnv->HolQuery(mat_h, mat_w, (const void*)data, result);
}

int32 BatchQueryDataBase(const int32 mat_h,  const int32 mat_w,  const uint8* data, fp32* result)
{
    return hVprAccEnv->Query(mat_w, mat_h, (void*)data, result);
}


int32 CVT_MatFP32ToFP8E4M3(const fp32 *matFP32, const uint32 size, const fp32 scale, uint8 *matFP8e4m3)
{
  uint32 i;
  for (i=0; i<size;i++) {
    *(matFP8e4m3+i) = __nv_cvt_float_to_fp8((*(matFP32+i))*scale, __NV_SATFINITE, __NV_E4M3);
  }
  return 0;
}
int32 CVT_MatFP8E4M3ToFP32(const uint8 *matFP8e4m3, const uint32 size, const fp32 scale, fp32 *matFP32)
{
  uint32 i;
  uint8 data;

  for (i=0; i<size;i++) {
	data = *(matFP8e4m3+i);
    // calculate the sign
    int32 sign = (((uint32)data)<<24)&0x80000000;
    int32 E    = (((uint32)data)>>3)&0x0F;

    int32 M    = (((uint32)data)>>0)&0x07;
    E = (E+127-7)<<23;
    M = M<<20;
    *((uint32*)(matFP32+i)) = sign|E|M ;
  }
  return 0;
}
int32 CVT_MatFP32ToFP8E5M2(const fp32 *matFP32, const uint32 size, const fp32 scale, uint8 *matFP8e5m2)
{
  uint32 i;
  for (i=0; i<size;i++) {
    *(matFP8e5m2+i) = __nv_cvt_float_to_fp8((*(matFP32+i))*scale, __NV_SATFINITE, __NV_E5M2);
  }
  return 0;
}
