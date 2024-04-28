/*
 * This file is part of the Hir-ATLAS distribution (https://github.com/Yuan-Fangming/Hir-ATLAS).
 * Copyright (c) 2024 Fangming Yuan.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 2.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */


/*
 * libvipracc.hpp
 *
 *  Created on: Apr 6, 2022
 *      Author: fangming
 */

#ifndef VIPRACC_HPP_
#define VIPRACC_HPP_


extern "C" {
/*supported algorithms*/
#define PAIRWISE 0
#define DUMMY    1
#define PAIRWISE_POSGRAPH  2     // LPG
#define ONLY_PAIRWISE_MATCH 3
#define PAIRWISE_RANSAC  4
#define HOLISTIC_FEATURE_MATCH 5
#define LPG_MATCH   6
#define RSS         7  // Rapid Spatial Scoring for Patch-NetVLAD
// test algorithms
#define PAIRWISE_STARGRAPH  1000



typedef struct{
	// local feature cross similarity matrix
    const float* sm_desc;
	 int sm_desc_H;
	 int sm_desc_W;
	// local feature coordinate range
     int fm_dim_y;
     int fm_dim_x;
	// frame a local feature pos matrix
    const int* a_pos_patch;
	 int a_kp_num;    // a is query
	 int a_pos_dim;
    // frame b local feature pos matrix
    const int* b_pos_mat;
     int  b_pos_dim;
     int b_frame_num;
    const int* b_frame_kp_num;
	const int* b_frame_kp_start_idx;  //local feature idx of the first local feature in each frame
    // frame b local feature neigbor matrix
    const int *b_frames_neighbor_mat;
    const int *b_kp_in_frame_neighbor_num; // number of neighbor of each local feature for all b frame
    const int *b_frame_neighbor_start_idx; // row idx of the start of the neighbor matrix of each b frame
     int frame_neighbor_attr_count;   // column size of b_frames_neighbor_mat

    const int *valid_b_frame;    // idx of the valid b frame
     int valid_b_frame_num; // total number of valid b frame

	const float *gaussian_patch;
	 int gauss_y_size;
	 int gauss_x_size;

     int thread_num;
     int alg_type;
    float* _o_sm_row;
    // when alg_type == ONLY_PAIRWISE_MATCH the following return the matched result of keypoint in frame A to all in frame B

	// when alg_type == PAIRWISE_RANSAC the following return the matched result of keypoint in frame A to all in frame B
	// a 0 in _o_a2b_match_distance while the corresponding 1 in _o_a2b_matches means an outliner
    int dbg;

    int  * _o_a2b_matches;
    float* _o_a2b_match_distance;
} TaskParam;



typedef struct {
  int frame_num; // the number of frame to process
  int start_frame; // the map of the first frame in this task to the total number of frame
  const int *valid_b_frame_start;
  // similarity matrix patch
  const float* sm_desc; // start address of sm patch
  // sm_desc slice rectangle origin point
  int sm_slice_h_start;
  int sm_slice_w_start;
  // the dimension of entire sm_desc
  int sm_desc_H;
  int sm_desc_W;
  //
  const int* a_pos_patch;
  int a_kp_num;

  const int *a_frame_neighbor_mat;
  const int *a_frame_kp_neighbor_num;
  int frame_neighbor_attr_count;
  //
  const int* b_pos;

  const int* b_frame_kp_num; // holds number of kp in each frame in b
  //
  const float *gaussian_patch;
  int gauss_y_size;
  int gauss_x_size;
  //
  int alg_type;

  int fm_dim_x;
  int fm_dim_y;

  const int *b_frames_neighbor_mat;
  const int *b_kp_in_frame_neighbor_num;
  const int *b_frame_neighbor_start_idx;

  const int* b_frame_kp_start_idx_segment;


  float* _o_sim_seg; // the location where the similarity result store
  int  * _o_a2b_matches_seg;
  float* _o_a2b_match_distance_seg;
  float* _o_a2b_match_distance_aux_seg;
  TaskParam *pTaskParam;
} RowSegParam;




extern int C_PairwiseSimilarityRowPosGraph_MultiThread(
                                    const float* sm_desc, const int sm_desc_H, const int sm_desc_W,
                                    const int* a_pos_patch, const int a_kp_num,
                                    const int* b_pos_mat,
                                    const int* b_frame_kp_num,
                                    const int b_frame_num,
                                    const int fm_dim_y, const int fm_dim_x,
                                    const int *a_frame_neighbor_mat, const int *kp_neighbor_num, const int frame_neighbor_attr_count,
                                    const float *gaussian_patch, const int gauss_y_size, const int gauss_x_size,
                                    const int thread_num,
                                    const int alg_type,
                                    float* _o_sm_row,
                                    // when alg_type == ONLY_PAIRWISE_MATCH the following return the matched result of keypoint in frame A to all in frame B

									// when alg_type == PAIRWISE_RANSAC the following return the matched result of keypoint in frame A to all in frame B
									// a 0 in _o_a2b_match_distance while the corresponding 1 in _o_a2b_matches means an outliner
                                    int  * _o_a2b_matches,
                                    float* _o_a2b_match_distance);

/*
all "b" reprensets database. All "a" represents query
sm_desc                     : in shape [a_kp_num, sum(*b_frame_kp_num)] The cross similarity between the query frame local feature to all the database frame local feature.
a_pos_patch                 : in shape [a_kp_num, 2] The keypoint 2-D coordinate.
b_pos_mat                   : in shape [sum(*b_frame_kp_num), 2] The keypoint 2-D coordinates for each frame in the DB
b_frame_kp_num              : in shape [b_frame_num]  Each holds the number of keypoints in each DB frame
b_frame_kp_start_idx        : in shape [b_frame_num]  Each holds the keypoint idx of each DB frame  b_frame_kp_start_idx[4]*2 is the idx of the first keypoint coordinate of frame 2 in b_pos_mat
b_frame_num                 : number of DB frames
fm_dim_y,fm_dim_x           : when alg_type == LPG_MATCH: These two variable indicates the window size in which the closest neighbors for each local feature in frame a is found
b_frames_neighbor_mat       : in shape [sum(b_kp_in_frame_neighbor_num), 3] Neighbor info of each kp in DB
                              b_frames_neighbor_mat[:,0] indexing each neighbor keypoint in each DB frame
                              b_frames_neighbor_mat[:,1] The delta y distance of the each neigbor keypoint to the center nodes (neighbor_y - center_y)
                              b_frames_neighbor_mat[:,2] The delta x distance of the each neigbor keypoint to the center nodes (neighbor_x - center_y)
b_kp_in_frame_neighbor_num  : in shape [sum(b_frame_kp_num)] neighbor number for each kp in frame in DB
b_frame_frame_start_idx     : idx for start of each frame in b_frames_neighbor_mat.  For start location of frame_2 ==> b_frame_frame_start_idx[2]*frame_neighbor_attr_count
frame_neighbor_attr_count   : number of int32 for each neighbor info in b_frames_neighbor_mat
valid_d_frame               : in shape [valid_d_frame_num] is the frame idx in b to operate
valid_d_frame_num           : The total number of valid b frame

dbg                         : for debug purpose
_o_a2b_matches              : pointer to output matches of a frame kp to each valid b frame
_o_a2b_match_distance       : pointer to output match distance of a frame kp to each valid b frame
_o_a2b_match_distance_aux   : pointer to output auxiliary match distance of a frame kp to each valid b frame.
                              This output data has different meaning according to the type of alg_type
                              PAIRWISE_POSGRAPH: Output the LPG spatial context score of each kp in a.
                              PAIRWISE:          Output to indicate the if kp in a outlier or inlier.
*/
extern int C_PairwiseSimilarityVersatile_MultiThread(
                                    const float* sm_desc, const int sm_desc_H, const int sm_desc_W,
                                    const int* a_pos_patch, const int a_kp_num,    // a is query
                                    const int* b_pos_mat,                          // b is DB
                                    const int* b_frame_kp_num,
									const int* b_frame_kp_start_idx,
                                    const int b_frame_num,
                                    const int fm_dim_y, const int fm_dim_x,
                                    const int *b_frames_neighbor_mat, const int *b_kp_in_frame_neighbor_num, const int *b_frame_neighbor_start_idx,  const int frame_neighbor_attr_count,
                                    const int *valid_b_frame, const int valid_b_frame_num,
									const float *gaussian_patch, const int gauss_y_size, const int gauss_x_size,
                                    const int thread_num,
                                    const int alg_type,
                                    float* _o_sm_row,
                                    // when alg_type == ONLY_PAIRWISE_MATCH the following return the matched result of keypoint in frame A to all in frame B

									// when alg_type == PAIRWISE_RANSAC the following return the matched result of keypoint in frame A to all in frame B
									// a 0 in _o_a2b_match_distance while the corresponding 1 in _o_a2b_matches means an outliner

									const int dbg,
									int  * _o_a2b_matches,
                                    float* _o_a2b_match_distance,
									float* _o_a2b_match_distance_aux);

extern int FilterMatch(const float *match_sm,
                int match_sm_H, int match_sm_W,
                int *_o_matches,
                float *_o_distance);
}
#endif /* VIPRACC_HPP_ */
