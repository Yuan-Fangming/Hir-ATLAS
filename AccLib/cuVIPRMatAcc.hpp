/*
 * cuVIPRMatAcc.hpp
 *
 *  Created on: Jun 28, 2023
 *      Author: fangming
 */

#ifndef CUVIPRMATACC_HPP_
#define CUVIPRMATACC_HPP_
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cublasLt.h>

typedef unsigned char            uint8;
typedef signed char              int8;
typedef int                      int32;
typedef unsigned int             uint32;
typedef long long int            int64;
typedef unsigned long long int   uint64;
typedef float                    fp32;

#define CU_VIPR_MAT_ACC_OK  0
#define CU_VIPR_MAT_ACC_ERR  1

typedef struct {
    cublasHandle_t blas_handle;
    // database holistic feature variables
    fp32*  fp32_gpu_db_hol_mat;
    uint32 fp32_db_hol_mat_h;
    uint32 fp32_db_hol_mat_w;

    // query holistic feature memory
    fp32*  fp32_gpu_q_hol_mat;
    uint32 fp32_q_hol_mat_h;
    uint32 fp32_q_hol_mat_w;
    uint64 fp32_q_hol_mat_size;
    // query out holistic feature memory
    fp32*  fp32_gpu_out_hol_mat;
    uint32 fp32_out_hol_mat_h;
    uint32 fp32_out_hol_mat_w;
    uint64 fp32_out_hol_mat_size;



    // database keypoint feature variables
    fp32*  fp32_gpu_base_mat;
    uint32 fp32_base_mat_h;     // matrix hight
    uint32 fp32_base_mat_w;     // matrix width
    uint64 fp32_base_mat_size;  // fp32_base_mat_h*fp32_base_mat_w
    // following variable to achieve dynamic base matrix assignment
    uint64 fp32_base_mat_end_col_pos; // the next free column in base matrix (entry index not the byte index)
    int32 db_frame_num;
    int32 *db_each_frame_keypoint_num;  // array in size db_frame_num. It sequentially store the number of keypoint number for each of the database frame
    int32 *db_frame_base_mat_start_col; // array in size db_frame_num. It sequentially store the column idx of the first local feature ofeach frame in the base mat
    int32 attached_db_frame_num;  // number of frame that had already attached.

    // Query keypoint feature variables
    fp32*  fp32_gpu_query_mat;
    uint32 fp32_query_mat_h;
    uint32 fp32_query_mat_w;
    uint64 fp32_query_mat_size;

    // Query and database cross similarity results variables
    fp32*  fp32_gpu_out_mat;
    uint32 fp32_out_mat_h;
    uint32 fp32_out_mat_w;
    uint64 fp32_out_mat_size;

    // for operation to normalize vectors
    fp32*   fp32_norm_vects;
    uint32  fp32_vect_dim;
    uint32  fp32_vect_num;

    fp32*   fp32_vects_norm2;

}cuVIPRMatAccEnv;

extern "C" {
extern int32 Init_cuVIPRMatAcc(void);


/*
 * Allocate the base matrix gpu memory in fp32 data type
 */
extern int32 malloDataBaseMatFP32(const int32 mat_h,  const int32 mat_w, const int32 db_frame_num);

/*
 *  Set the database Holistic feature matrix in the GPU memory
 *  For a database with 55 places and 4090-dim holistic feature, the row-major holistic feature matrix will in numpy mat shape=[55,4096].
 *  When set this matrix into the gpu memory using this function, the matrix is stored in column-major with shape [4096,55], where the leading dimension is 1024
 *  mat_h/w: Height and width of the holistic feature matrix in column-major storage format
 */
extern int32 SetDBHolDataFP32(const int32 mat_h,  const int32 mat_w, const fp32* data);
/**
 * Set a matrix in GPU memory in data type float point 32-bit. The matrix is in column-major storage format.
 * mat_h/w: hight and width of the matrix
 * data:    the matrix data in column-major storage format.
 */
extern int32 SetDataBaseMatFP32(const int32 mat_h,  const int32 mat_w,  const fp32* data);

/*
 *Set The Database Feature matrix information
 *db_frame_keypoint_num:        in shape [db_frame_num]. It holds number of local feature for each framein the database
 *db_frame_base_mat_start_col:  in shape [db_frame_num]. It holds column of database feature mat for the first local feature pf each database frame
 */
extern int32 SetDatabaseFeatureMatInfo(const int32* db_frame_keypoint_num, const int32* db_frame_base_mat_start_col);
/**Dynamic base matrix assignment
 * Attach a new base matrix in the row direction to the already set base matrix in GPU memory
 *
 */
extern int32 BatchAttachDataBaseMatrixFP32(const int32 mat_h,  const int32 mat_w,  const fp32* data,
		                                   const int32 batch_db_frame_num, const int32* batch_db_frame_keypoint_num);


/*
 * Batch query holistic feature in all the database holistic feature
 */
extern int32 BatchQueryHolisticFeatureFP32(const int32 mat_h,  const int32 mat_w,  const fp32* data, fp32* result);


/*
 * ----------------------mat_w-------------------------
 *|                                                    |
 *|                                                    |
 *mat_h              Base matrix                       |
 *|                                                    |
 *|                                                    |
 *|                                                    |
 * ----------------------------------------------------
 *
 *              Mul
 *
 * -------------q_mat_w----------
 *|        |             |       |
 *|   Q0   |     Q1      |   Q2  |
 *|        |             |       |
 *q_mat_h  |             |       |     Query matrix
 *|        |             |       |
 *|        |             |       |
 *|        |             |       |
 * ------------------------------
 *              |
 *             \|/
 *            mat_h ==  q_mat_h
 *  ------------q_mat_w---------------
 * |       |                |         |
 * |  Q0   |       Q1       |   Q2    |
 * |       |                |         |
 *mat_w    |    result_mat  |         |                result_mat = transpose(Base_mat)*Query_mat
 * |       |                |         |
 * |       |                |         |
 * |       |                |         |
 *  ----------------------------------
 * query the base mat by Calculate matmul of the input matrix with the base mat in gpu memory
 * Base_mat * transpose(query_mat)
 *
 * mat_h/w: hight and width of the input matrix
 * data: The input query matrix in column-major storage format.
 * result: The result of the matmul
 */
extern int32 BatchQueryDataBaseFP32(const int32 mat_h,  const int32 mat_w,  const fp32* data, fp32* result);


/* This function retrieve 1 query frames with N database candidate frame  (The local feature in query frame matmul with local feature in N database frames )
 * each query frame has different top-N database candidate. )
 *const int32 q_mat_h,  const int32 q_mat_w: width and height of the query frame feature matrix. q_mat in shape [q_mat_h, q_mat_w]
 *const fp32* q_mat:                         In column-major shape=[desc_dim, q_kp_num] for .  Pointer to the Q query feature matrix
 *const const int32 N:                       The N database frame for retrieve.
 *const int32* data_base_N_idx:              shape=[N] The N database frame idx, which the query frame to retrieve
 *const int32  total_db_kp:                  totally number of db local feature joint the retrieval
 */
extern int32 QueryDataBaseN_FP32(const int32 q_mat_h, const int32 q_mat_w,
		                                  const fp32* q_mat,
										  const int32 N, const int32* data_base_N_idx, const int32 total_db_kp,
										  fp32* result);

/*
 * Normalize batch vectors
 * vect_dim: dimension of vector
 * vect_num: number of vector in vector batch
 * data:     The vector batch data. In shape (vect_dim, vect_dim)    column-major storage
 * data_normlized:     Host memory to fetch the batch of normalized vectors.
 */
extern int32 NormalizeBatchVectorFP32(const int32 vect_dim, const int32 vect_num, const fp32* data, fp32* data_normlized);
}/*end extern "C" */




typedef unsigned char DtypeCount_t;   // data type represents the number of byte of the matrix datatype
#define FP8_BYTE_COUNT   1
#define FP16_BYTE_COUNT  2
#define FP32_BYTE_COUNT  4
#define INT8_BYTE_COUNT   1
#define UINT8_BYTE_COUNT   1
#define INT16_BYTE_COUNT  2
#define UINT16_BYTE_COUNT  2
#define INT32_BYTE_COUNT  4
#define UINT32_BYTE_COUNT  4
#define INT64_BYTE_COUNT  8
#define UINT64_BYTE_COUNT  8

typedef uint64  MatDim_t;       // matrix dimension datatype
#define CU_VPR_ACC_SUCCESS     0
#define CU_VPR_ACC_FAILED      1
enum MatDataOrder{
	ROW_MAJOR,
	COL_MAJOR
};
class CuVPRAccMat{
public:
	CuVPRAccMat()
	{
		mat_h = 0;
		mat_w = 0;
		mat_size = 0;
		mem_entry_size = 0;
		mat_next_fill_line = 0;
		bytecount = 1;
		mat=(void*)0;
	}

    /*  Set the matrix data type byte count
     * */
	void SetMatrixDtypeCount(const cudaDataType cuDtype, const DtypeCount_t bytecount);

	DtypeCount_t GetMatrixDtypeCount(void);

	/* Allocate GPU memory for the Matrix
	 * h/w: The allocated matrix size of the gpu memory, where w is the leading dimension.
	 * return: CU_VPR_ACC_SUCCESS if GPU memory allocation success
	 *         CU_VPR_ACC_FAILED if GPU memory allocation failed
	 */
    uint8 allocMatGpuMem(const MatDim_t h, const MatDim_t w);
    /* This function release the gpu memory allocated by the function "allocMatGpuMem"
     */
    uint8 ReleaseMatGpuMem(void);

    /*This function automatically fill the matrix data into the GPU memory.
     * It operates in the following three steps:
     * 1. Check if there is already enough GPU memory allocated to store the input data.
     * 2. If not enough GPU memory allocated to store the input data---> release the GPU memory and re-allocate memory to hold the incoming data,
     * 3. Fill the GPU memory with the incomig data
     *
     * h/w:  The height and width of the incoming data to fill the GPU memory for this matrix
     * data:  Void pointer point to the data to fill
     * return: CU_VPR_ACC_SUCCESS if operation success
	 *         CU_VPR_ACC_FAILED if operation failed
	 */
    uint8 DynamicFillMatGpuData(const MatDim_t h, const MatDim_t w, const void* data);

    /* Dynamically reshape the gpu matrix.
     * This function release and reallocate GPU memory if the current GPU mem buffer cannot hold the reshape size.
     * chenges the height and width value of mat_desc
     * h/w: the reshaped height and width value for the new dimension of the matrix in GPU
     * */
    uint8 DynamicReshapeMatGpu(const MatDim_t h, const MatDim_t w);
    /* Allocate Matrix GPU desc
     * */
    uint8 allocGpuMatDesc(void);

    uint8 SetGpuMatDesc(const MatDim_t h, const MatDim_t w);
    uint8 SetGpuMatDescLayoutOrder(MatDataOrder layout_order);
	/* Allocate CPU memory for the Matrix
	 * h/w: The allocated matrix size of the gpu memory, where w is the leading dimension.
	 * return: CU_VPR_ACC_SUCCESS if CPU memory allocation success
	 *         CU_VPR_ACC_FAILED if CPU memory allocation failed
	 */
    uint8 allocMatCpuMem(const MatDim_t h, const MatDim_t w);



	/*  Assign data to the Matrix in GPU memory starting from line 0
	 *  data_h: The height of the mat_data matrix in CPU memory. Here only the height of the mat_data is required because
	 *          it is assumed the mat_data has the same leading dimension as the mat variable.
	 *          The data_h variable will update the "mat_next_fill_line" variable.
	 *  mat_data: The data in cpu to assign.
	 *
     * return: CU_VPR_ACC_SUCCESS if data successfully assigned to the GPU memory
	 *         CU_VPR_ACC_FAILED if data failed assigned to the GPU memory
	 */
	//uint8 AssignMatData(const MatDim_t data_h, const void* mat_data);

	/*  Assign data to the Matrix in GPU memory starting from GPU memory location indicated by "mat_next_fill_line"
	 *  data_h: The height of the mat_data matrix in CPU memory. Here only the height of the mat_data is required because
	 *          it is assumed the mat_data has the same leading dimension as the mat variable.
	 *          The data_h variable will update the "mat_next_fill_line" variable.
	 *  mat_data: The data in cpu to assign.
	 *
     * return: CU_VPR_ACC_SUCCESS if data successfully assigned to the GPU memory
	 *         CU_VPR_ACC_FAILED if data failed assigned to the GPU memory
	 */
	uint8 AttachDataToMat(const MatDim_t data_h, const void* mat_data);


	/* Get a element from the matrix
	 * H/W: Get the mat[H][W]
	 * */
	template <typename T> T Get(MatDim_t H, MatDim_t W);
	template <typename T> void Set(const MatDim_t h, const MatDim_t w, T value);
    /* Get the height and width of the matrix
     */
	MatDim_t GetMatHeight(void);
	MatDim_t GetMatWidth(void);
    MatDim_t GetMatLD(void);   // get mat leading dimension
	/* Get the matrix GPU memory pointer
	 */
	void* GetMatPtr(void);
	/*Get the cublaslt matrix layout desc
	 * */
	cublasLtMatrixLayout_t GetCuMatDesc(void)
	{
		return mat_desc;
	}

	/* Get the matrix gpu data back to the cpu memory
	 * data: pointer in CPU memory to hold the matrix in GPU memory
	 * return: CU_VPR_ACC_SUCCESS if success
	 *         CU_VPR_ACC_FAILED if not success
	 * **/
	uint8 GetGpuData(void *data);
	/* Get matrix with partial row of gpu data back to the cpu memory
	 * data: pointer in CPU memory to hold the matrix in GPU memory
	 * partial_row_length: GPUmem represents the matrix GPU memory. The data in GPUmem(:, 0:partial_row_length) are read back
	 * return: CU_VPR_ACC_SUCCESS if success
	 *         CU_VPR_ACC_FAILED if not success
	 * **/
	uint8 GetMatGpuDataParitalRow(void* data, const MatDim_t partial_row_length);

	void dbg_PrintMember(void);
	template <typename T> void dbg_PrintMat(uint64 top);
private:
	void* mat;        // allocated CPU/GPU memory space to store the matrix
    uint64 mem_entry_size;  // number of entry can be hold by the memory
    MatDataOrder mat_data_order;
	MatDim_t mat_h;    // matrix height
	MatDim_t mat_w;    // matrix width. The leading dimension of mat
    uint64 mat_size; // Total elements in the matrix. Helper variable  mat_size=mat_h*mat_h
    MatDim_t mat_next_fill_line;  // The next free line in the mat memory.This field is used for matrix partial fill.
    DtypeCount_t bytecount; // number of byte of the data type of the matrix

    cublasLtMatrixLayout_t mat_desc;
    cudaDataType cuDtype;
};

class cuVPRMatAccEnv{
public:
	cuVPRMatAccEnv(const cudaDataType cuDtype, const DtypeCount_t bytecount);

	void GetPtrAttrbute(void* ptr);

	/* This function allocate the necessory CPU and GPU memory for the DataBase (DB)
	 *First, it allocate CPU memory for:
	 *    1. db_each_frame_keypoint_num_mat
	 *    2. db_frame_base_mat_start_col_mat
	 *Second, it allocate GPU memory for the database local features in:
	 *    DB_local_feature_mat
	 *kp_num/kp_dim:  The DB local feature matrix is in shape [kp_num, kp_dim], where kp_num and kp_dim
	 *                are determined by the input parameter kp_num and kp_dim.
	 *db_frame_num:   The total number of frames in the database.
	 *
	 *return:CU_VPR_ACC_SUCCESS if allocate success for all the necessary matrix.
	 *       CU_VPR_ACC_FAILED  if either of the memory allocation failed.
	 * */
	/*AllocateDB_LocalFeature takes DB local feature matrix as column-major dimension*/
    int32 AllocateDB_LocalFeature(const MatDim_t kp_num,  const MatDim_t kp_dim, const uint32 db_frame_num);
    int32 AllocateDB_HolisticFeature(const MatDim_t feature_num, const MatDim_t feature_dim);
    /* Set the Database Info
     * Set the db_each_frame_keypoint_num_mat and db_frame_base_mat_start_col_mat matrix
     * db_frame_keypoint_num:  Pointer to array with shape [db_frame_num] that holds the number of local features in each frame
     * db_frame_base_mat_start_col: Pointer to array with shape [db_frame_num] that holds the start column of each frame's local
     *                              feature in the DB local feature matrix.
     *
     * return:  CU_VPR_ACC_SUCCESS or CU_VPR_ACC_FAILED.
     * */
    int32 SetDBInfo(const int32* db_frame_keypoint_num, const int32* db_frame_base_mat_start_col);
    /*This function attach partial local feature matrix to the Database local feature GPU memory
     * kp_num/kp_dim: indicates the number of local feature and feature dimension (combine to define the local feature matrix shape) attach to the GPU memory.
     * data:          Pointer to the local feature data
     * dtype_count:   Indicates the number bytes for the data type of the assigned local feature matrix data
     *                used to check the data type consistency between the assigned data and the DB_local_feature_mat
     *
     * batch_db_frame_num:  The number of frame to which the attached local features belongs.
     * batch_db_frame_keypoint_num: array in shape [batch_db_frame_num]. Elements in the
     *                              array indicate the number of local features in each of the frame.
     * return:CU_VPR_ACC_SUCCESS if attachment success.
	 *        CU_VPR_ACC_FAILED  if attachment failed.
     * */
    int32 AttachLocalFeatureToDB(const MatDim_t kp_num,  const MatDim_t kp_dim,  const void* data,
    		                     const int32 batch_db_frame_num, const int32* batch_db_frame_keypoint_num);
    /* Attach the holistic feature to the query_hol_mat
     * hol_f_num:  The total number of holistic feature (The height of the holistic feature matrix)
     * hol_f_dim:  The dimension of each holistic feature (The width of the holistic feature matrix)
     * data:       Void pointer to the holistic feature matrix
     * return:CU_VPR_ACC_SUCCESS if attachment success.
	 *        CU_VPR_ACC_FAILED  if attachment failed.
     * */
    int32 AttachHolFeatureToDB(const MatDim_t hol_f_num,  const MatDim_t hol_f_dim,  const void* data);


    int32 HolQuery(const MatDim_t hol_f_num,  const MatDim_t hol_f_dim,  const void* data, fp32* result);

    /*This function takes column-major matrix input*/
    int32 Query(const MatDim_t loc_feature_dim, const MatDim_t loc_feature_num, const void* data, fp32* result);
    /*This function takes column-major matrix input*/
    int32 QueryN(const MatDim_t q_mat_h, const MatDim_t q_mat_w,
                 const void* q_mat,
			     const int32 N, const int32* data_base_N_idx, const int32 total_db_kp,
			     fp32* result);
private:
	cublasLtHandle_t ltHandle;
	cublasLtMatmulDesc_t operationDesc = NULL;
	cublasLtMatmulPreference_t preference = NULL;


	/***************** Database matrix*****************/
	/*Hol feature matrix */
	CuVPRAccMat DB_hol_mat;
	/*local feature matrix */
	CuVPRAccMat DB_local_feature_mat;

    // following variable to achieve dynamic base matrix assignment
    uint32 db_frame_num;   // total number of frame in the DB
    //int32 attached_db_frame_num;  // number of frame that had already attached.
    CuVPRAccMat db_each_frame_keypoint_num_mat;  // array in size db_frame_num. It sequentially store the number of keypoint number for each of the database frame
    CuVPRAccMat db_frame_base_mat_start_col_mat; // array in size db_frame_num. It sequentially store the column idx of the first local feature ofeach frame in the base mat


    /***************Query and result matrix****************/
    // local feature query and result mat
    CuVPRAccMat query_local_feature_mat;         // Query keypoint feature variables
    CuVPRAccMat local_feature_query_result_mat;  // Query and database cross similarity results variables
    // holistic feature query and result mat
    CuVPRAccMat query_hol_mat;          // holistic feature query mat
	CuVPRAccMat hol_query_result_mat;   // holistic feature query result mat
    /**************************  Feature normalization **************************/
    //fp32*   fp32_norm_vects;
    //uint32  fp32_vect_dim;
    //uint32  fp32_vect_num;

    //fp32*   fp32_vects_norm2;

    /*****************cuBlasLt matmul desc******************/
    cublasLtMatmulDesc_t HolFeatureQueryDesc = NULL;           // matmul for holistic feature query
    cublasLtMatmulPreference_t HolFeatureQueryMatmul_preference = NULL;
    //cublasLtMatmulHeuristicResult_t HolFeatureQueryMatmul_heuristicResult = {};

    cublasLtMatmulDesc_t LocalFeatureQueryDesc = NULL;         // matmul for local feature query in all DB frame local feature
    cublasLtMatmulPreference_t LocalFeatureQueryMatmul_preference = NULL;


    cublasOperation_t TRANSPOSE;
    cublasOperation_t NO_TRANSPOSE;



};
/* Check the cublas status and print the error code if error occur
 * stste: the cublas status returned by the cublas function
 * str:   pointer to string which will print --->str: Error messages
 *
 * return 0 if status is success 1 otherwise.
 * */
uint8 __CheckStatus(cublasStatus_t state, const char* str);

extern "C" {


/*APIs that use the CublasLt to acc the VPR */
/* Initialize the VprAccEnv create cuVPRMatAccEnv and initialize the cuVPRMatAccEnv handle.
 * feature_datatype: The data type used for the feature matrix algorithm. can be DTYP_***.
 *
 * */
#define DTYP_FP8_E4M3  0
#define DTYP_FP8_E5M2  1
//#define DTYP_FP16      2
//#define DTYP_BF16      3
#define DTYP_FP32      4
#define DTYP_INT8      5
//#define DTYP_INT16     6
extern void InitMatAccExEnv(const uint32 feature_datatype);
extern int32 malloDataBaseMat(const int32 mat_h,  const int32 mat_w, const int32 db_frame_num);
extern int32 SetDBInfo(const int32 *db_frame_keypoint_num, const int32 *db_frame_base_mat_start_col);
/* The data use uint8 data type pass the local feature matrix start address
 * all the data matrix in row-major order
 * */
extern int32 BatchAttachDataBaseMatrix(const MatDim_t mat_h,  const MatDim_t mat_w,  const uint8 *data,
		                                   const int32 batch_db_frame_num, const int32* batch_db_frame_keypoint_num);
extern int32 BatchAttachHolFeature(const MatDim_t hol_f_num,  const MatDim_t hol_f_dim,  const uint8 *data);


extern int32 BatchQueryHolisticFeature(const int32 mat_h,  const int32 mat_w,  const uint8* data, fp32* result);
extern int32 BatchQueryDataBase(const int32 mat_h,  const int32 mat_w,  const uint8* data, fp32* result);

/*************************Convert FP32 to *Datatype ***********************/
/*These two function convert a Vector or Matrix from FP32 data type into FP8 (E4M3 or E5M2)
 * This function assumes the matFP8e4m3 with length of "size" is pre-allocated in the CPU memory.
 * const fp32 *matFP32:  Input vector or matrix in FP32 data type.
 * const uint32 size:    Input vector length or matrix size (WxH) in number of entries.
 * const fp32 scale:     matFP32 scale factor. The function do the conversion FP8ExMx(matFP32*scale).
 * uint8 *matFP8e4m3:    The output converted FP8 vector or matrix
 *
 * The functions always return 0
 */
extern int32 CVT_MatFP32ToFP8E4M3(const fp32 *matFP32, const uint32 size, const fp32 scale, uint8 *matFP8e4m3);
extern int32 CVT_MatFP32ToFP8E5M2(const fp32 *matFP32, const uint32 size, const fp32 scale, uint8 *matFP8e5m2);

extern int32 CVT_MatFP8E4M3ToFP32(const uint8 *matFP8e4m3, const uint32 size, const fp32 scale, fp32 *matFP32);
}/*end extern "C" */
#endif /* CUVIPRMATACC_HPP_ */
