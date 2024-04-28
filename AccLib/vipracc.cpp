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

#include <iostream>
#include <string>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/SVD>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "vipracc.hpp"
#include <time.h>

using namespace std;
using namespace Eigen;
using namespace cv;
using namespace std;

//#define VIPRACC_DEBUG
/*****
 *random sample K matched correspondence and return in Matrix with Homogeneous coordinate
 *template MAT_T matrix type
 *a_pos: pointer to position of keypont group A in normal coordinate
 *b_pos: pointer to position of keypont group B in normal coordinate
 *a2b_matches: matched distance of A in B
 *a2b_match_len: length of a2b_matches
 *sample_num:  the number of random sampled correspondences.
 *_o_A_mat: random sampled A keypoint position mat in Homogeneous coordinate
 *_o_B_mat: random sampled B keypoint position mat in Homogeneous coordinate
 * *****/

void Random_0_N_Array(const int _0_to, int*outArray, const unsigned int array_len)
{
  unsigned int main_loop_cycle = array_len/8;
  unsigned int remain_loop_cycle = array_len%8;
  unsigned int array_idx = 0;
  for (unsigned int i=0; i<main_loop_cycle; i++) {
	*(outArray+array_idx) = rand()%_0_to;
	array_idx++;
	*(outArray+array_idx) = rand()%_0_to;
	array_idx++;
	*(outArray+array_idx) = rand()%_0_to;
	array_idx++;
	*(outArray+array_idx) = rand()%_0_to;
	array_idx++;
	*(outArray+array_idx) = rand()%_0_to;
	array_idx++;
	*(outArray+array_idx) = rand()%_0_to;
	array_idx++;
	*(outArray+array_idx) = rand()%_0_to;
	array_idx++;
	*(outArray+array_idx) = rand()%_0_to;
	array_idx++;
  }
  for (unsigned int i=0; i<remain_loop_cycle; i++) {
	*(outArray+array_idx) = rand()%_0_to;
	array_idx++;
  }
}


int ProbeMatchCount(const int* match, const int a2b_match_len, unsigned int* match_idx_list)
{
  int match_pair_count =0;
  for (int i=0; i<a2b_match_len; i++) {
    if (*(match+i)>=0) {
      *(match_idx_list+match_pair_count) = i;
      match_pair_count++;
    }
  }
  return match_pair_count;
}
void FetchMatchedKp_A(const int* kp_pos,
		             const int* match, const int a2b_match_len,
					 cv::Mat &matched_kp_pos_mat)
{
  int match_pair_count =0;
  for (int i = 0; i<a2b_match_len; i++) {
	if (*(match+i)>=0) { // if there is a match
	  matched_kp_pos_mat.at<float>(match_pair_count, 0)  = *(kp_pos+(i<<1));
	  matched_kp_pos_mat.at<float>(match_pair_count, 1)  = *(kp_pos+(i<<1)+1);
	  match_pair_count++;
	}
  }
}

void FetchMatchedKp_B(const int* kp_pos,
		             const int* match, const int a2b_match_len,
					 cv::Mat &matched_kp_pos_mat)
{
  int match_pair_count =0;
  for (int i = 0; i<a2b_match_len; i++) {
	if (*(match+i)>=0) { // if there is a match
	  int in_b_idx= *(match+i);
	  matched_kp_pos_mat.at<float>(match_pair_count, 0)  = *(kp_pos+(in_b_idx<<1));
	  matched_kp_pos_mat.at<float>(match_pair_count, 1)  = *(kp_pos+(in_b_idx<<1)+1);
	  match_pair_count++;
	}
  }
}

int FetchMatchedCorrespondence(const int* a_pos, const int* b_pos,
		                       const int *a2b_matches, const int a2b_match_len,
							   MatrixXf &_o_A_mat, MatrixXf &_o_B_mat)
{
  // check each matches
  int match_pair_count =0;
  for (int i = 0; i<a2b_match_len; i++) {
    if (*(a2b_matches+i)>=0) { // if there is a match
      // fill _o_A_mat
      _o_A_mat(match_pair_count, 0) = *(a_pos+(i<<1));
      _o_A_mat(match_pair_count, 1) = *(a_pos+(i<<1)+1);
      _o_A_mat(match_pair_count, 2) = 1.0;
      // fill _o_B_mat
      int in_b_idx= *(a2b_matches+i);
      _o_B_mat(match_pair_count, 0) = *(b_pos+(in_b_idx<<1));
      _o_B_mat(match_pair_count, 1) = *(b_pos+(in_b_idx<<1)+1);
      _o_B_mat(match_pair_count, 2) = 1.0;
      //increment match_pair_count
      match_pair_count++;
    }
  }
  return match_pair_count;
}


void FillSampledKeypoint(const MatrixXf &kp_homo, const int *index, const int idx_length, MatrixXf &kp_homo_sampled)
{
  for (int i=0; i<idx_length; i++) {
    kp_homo_sampled.block<1,3>(i,0) = kp_homo.block<1,3>(*(index+i),0);
  }
}

void CalcuPointSetNormMat(const MatrixXf &kp_set, MatrixXf &kp_norm_mat)
{
  int p_num = kp_set.rows();
  ArrayXXf y_vect(p_num,1);
  ArrayXXf x_vect(p_num,1);
  ArrayXXf y_diff(p_num,1);
  ArrayXXf x_diff(p_num,1);
  ArrayXXf joint_mean_square(p_num,1);
  y_vect = kp_set.col(0);
  x_vect = kp_set.col(1);
  float mean_y = y_vect.mean();
  float mean_x = x_vect.mean();
  //ArrayXf mean_y_mat=ArrayXf::Constant(p_num, 1, mean_y);
  //ArrayXf mean_x_mat=ArrayXf::Constant(p_num, 1, mean_x);
  y_diff = y_vect-mean_y;
  x_diff = x_vect-mean_x;

  y_diff = y_diff.square();
  x_diff = x_diff.square();

  joint_mean_square = y_diff+x_diff;
  joint_mean_square = joint_mean_square/2.0;
  float joint_mean_square_sum = joint_mean_square.sum();
  joint_mean_square_sum =  joint_mean_square_sum/(p_num<<1);
  float s = 1.0/sqrt(joint_mean_square_sum);

  kp_norm_mat(2,2) = 1.0;
  kp_norm_mat(0,0) = s;
  kp_norm_mat(0,2) = -mean_y*s;
  kp_norm_mat(1,1) = s;
  kp_norm_mat(1,2) = -mean_x*s;
}


void __PrintMatches(int* matches, float *sim, const int a_num)
{
  cout<<"Print matches!!"<<endl;
  for (int i=0;i<a_num;i++) {
    cout<<i<<"->"<<*(matches+i)<<"----"<<sim[i]<<endl;
  }
}

float __idx_sm(const float* sm, int W, int h, int w)
{
  return sm[h*W+w];
}


const int* __pointer_neighbor(const int *kp_neighbor_mat, const int frame_neighbor_attr_count, const int idx)
{
  return kp_neighbor_mat + idx*frame_neighbor_attr_count;
}

int __idx_neighbor_idx(const int *kp_neighbor_mat, const int frame_neighbor_attr_count, const int idx)
{
  return *(kp_neighbor_mat + idx*frame_neighbor_attr_count + 0);
}
int __idx_neighbor_delta_y(const int *kp_neighbor_mat, const int frame_neighbor_attr_count, const int idx)
{
  return *(kp_neighbor_mat + idx*frame_neighbor_attr_count + 1);
}
int __idx_neighbor_delta_x(const int *kp_neighbor_mat, const int frame_neighbor_attr_count, const int idx)
{
  return *(kp_neighbor_mat + idx*frame_neighbor_attr_count + 2);
}



/* Mutual Match the neighbors of the two local features in frame a and frame b
 * This function conduct local feature mutual matches among the neighbor local features of two local features from frame a and frame b
 * match_sm:     In shape [a_num, b_num] The mutual similarity between the local features in frame a and frame b
 * a_num, b_num: The number of local features in frame a and frame b
 * a/b_kp_neighbor_mat: in shape [a/b_kp_neighbor_num, a/b_frame_neighbor_attr_count] The neighbor mat of the two local features in frame a and frame b
 * a/b_kp_neighbor_num: The neighbor number of the two local features in frame a and frame b
 * a/b_frame_neighbor_attr_count: Number of elements in the neighbor mat of frame a and frame b
 *
 * _o_matches,_o_distance: In shape [a_num] The output matches. All the members of _o_matches has to be all equals -1 when call this function
 *                         The matches among the neighbor local features are mapped to this frame-wise local feature matches.
 */
float __MutualMatchAmongNeighbor(const float *match_sm,
		                  const int a_num,
						  const int a_kp_idx,
		                  const int *a_kp_neighbor_mat, const int a_kp_neighbor_num, const int a_frame_neighbor_attr_count,
						  const int b_num,
						  const int b_kp_idx,
						  const int *b_kp_neighbor_mat, const int b_kp_neighbor_num, const int b_frame_neighbor_attr_count,
						  const float *gaussian_patch, const int gauss_y_size, const int gauss_x_size)
						  //int *_o_matches,
						  //float *_o_distance)
{

  int gauss_center_y = gauss_y_size>>1;
  int gauss_center_x = gauss_x_size>>1;
  int gauss_idx;
  float graph_pos_score = 0.0;
  float gauss_score=0.0;
  // declare the LPG neighbor ky similarity matrix
  float lpg_neighbor_sm[a_kp_neighbor_num*b_kp_neighbor_num];
  // construct the LPG neighbor similarity matrix
  for (int i=0; i<a_kp_neighbor_num; i++) {
	int idx_a = __idx_neighbor_idx(a_kp_neighbor_mat, a_frame_neighbor_attr_count, i);
	int lpg_neighbor_sm_line_idx = i*b_kp_neighbor_num;
	int match_sm_line_idx = idx_a*b_num;
    for (int j=0;j<b_kp_neighbor_num; j++) {
      int idx_b = __idx_neighbor_idx(b_kp_neighbor_mat, b_frame_neighbor_attr_count, j);
      lpg_neighbor_sm[lpg_neighbor_sm_line_idx+j] = match_sm[match_sm_line_idx+idx_b];
    }
  }
  // apply mutual match on the LPG neighbor similarity matrix
  /*matches based on the LPG neighbor similarity matrix*/
  int lpg_matches[a_kp_neighbor_num];
  float lpg_distance[a_kp_neighbor_num];
  FilterMatch(lpg_neighbor_sm,
		      a_kp_neighbor_num, b_kp_neighbor_num,
			  lpg_matches,
			  lpg_distance);

  // apply LPG
  for (int i=0;i<a_kp_neighbor_num;i++) {
	int match_idx_in_b = lpg_matches[i];
    if ( match_idx_in_b>=0) {
      //cout <<"A---"<<endl;
      int a_delta_y = __idx_neighbor_delta_y(a_kp_neighbor_mat, a_frame_neighbor_attr_count, i);
      int a_delta_x = __idx_neighbor_delta_x(a_kp_neighbor_mat, a_frame_neighbor_attr_count, i);
      if ((a_delta_y!=0)&&(a_delta_x!=0)) {
        //cout <<"B---"<<endl;
    	int b_delta_y = __idx_neighbor_delta_y(b_kp_neighbor_mat, b_frame_neighbor_attr_count, match_idx_in_b);
		int b_delta_x = __idx_neighbor_delta_x(b_kp_neighbor_mat, b_frame_neighbor_attr_count, match_idx_in_b);
		int diff_y = b_delta_y-a_delta_y;
		int diff_x = b_delta_x-a_delta_x;
		//cout <<"A"<<endl;
        diff_y += gauss_center_y;
        diff_x += gauss_center_x;
        // look up gaussian score
        gauss_idx = diff_y*gauss_x_size+diff_x;
        //cout<<diff_y<<","<<diff_x<<endl;

        /*
        cout <<"B*a "<<a_delta_y<<" "<<a_delta_x<<endl;
        cout <<"B*b "<<b_delta_y<<" "<<b_delta_x<<endl;
        cout <<"B   "<<diff_y<<" "<<diff_x<<endl;
        */
        gauss_score = gaussian_patch[gauss_idx];
        //match_score =
        graph_pos_score += gauss_score;
      }
    }
  }
  graph_pos_score /= a_kp_neighbor_num;
  //cout <<"exit"<<endl;
  return graph_pos_score;

/*
  // mapping the lpg matches to the frame level local feature matches
  for (int i=0; i<a_kp_neighbor_num;i++) {
	int neighbor_mat_idx_in_b = lpg_matches[i];
    if(neighbor_mat_idx_in_b>=0) {
      int idx_a = __idx_neighbor_idx(a_kp_neighbor_mat, a_frame_neighbor_attr_count, i);
      int idx_b = __idx_neighbor_idx(b_kp_neighbor_mat, b_frame_neighbor_attr_count, neighbor_mat_idx_in_b);
      _o_matches[idx_a] = idx_b;
      _o_distance[idx_a] = lpg_distance[i];
    }
  }*/
}

float C_PairwiseSimilarityLPG_match(const float *match_sm, // cross similarity matrix for local features between a and b. In shape [a_num, b_num]
		                            // a is query
		                            const int a_num,  // number of local featurein frame a
                                    const int *a_pos, // coordinate of each local feature in frame a
									const int *a_frame_neighbor_mat, const int *a_kp_neighbor_num, const int a_frame_neighbor_attr_count,
									// b is database
									const int b_num,  // number of local featurein frame b
									const int *b_pos, // coordinate of each local feature in frame b
									const int *b_frame_neighbor_mat, const int *b_kp_neighbor_num, const int b_frame_neighbor_attr_count,

                                    const float *gaussian_patch, const int gauss_y_size, const int gauss_x_size,
                                    const int alg_type)
{
/*	Verify the correctness of match_sm
 *  int matches[a_num];
	float distance[a_num];
	FilterMatch( match_sm,
			     a_num, b_num,
				 matches,
	             distance);

    float sim_sum = 0.0;
    int matches_size = a_num;
    int i;
    int match_idx;
    for (i=0;i<matches_size; i++) {
      match_idx = *(matches+i);
      if (match_idx>=0) {
        sim_sum+= *(distance+i);
      }
    }
    return sim_sum/sqrt(float(a_num*b_num));
*/

	// cout <<"New kp"<<endl;
	//  cout <<"A-X"<<	__idx_neighbor_delta_y(a_frame_neighbor_mat, a_frame_neighbor_attr_count, 0) << endl;
	//  cout <<"A-Y"<<	__idx_neighbor_delta_x(a_frame_neighbor_mat, a_frame_neighbor_attr_count, 0) << endl;
	//  cout <<"B-X"<<	__idx_neighbor_delta_y(b_frame_neighbor_mat, b_frame_neighbor_attr_count, 0) << endl;
	//  cout <<"B-Y"<<	__idx_neighbor_delta_x(b_frame_neighbor_mat, b_frame_neighbor_attr_count, 0) << endl;
  int neighbor_matches[a_num];
  float neighbor_distance[a_num];

  float lpg_score_matrix[a_num*b_num];
  // the idx for each keypoint's neighbor in frame a in a_frame_neighbor_mat (only the location use
  // multiply with a_frame_neighbor_attr_count to address the specific element)
  int a_frame_neighbor_kp_start_idx = 0;
  /*for each keypoint in frame a*/
  for (int ai=0; ai<a_num; ai++) {

	const int *a_kp_neighbor_mat = __pointer_neighbor(a_frame_neighbor_mat, a_frame_neighbor_attr_count, a_frame_neighbor_kp_start_idx);
	//cout<<"A#####----"<< ai<<endl;
	int a_number_of_kp_neighbor = a_kp_neighbor_num[ai];

    int b_frame_neighbor_kp_start_idx = 0; // same as a_frame_neighbor_kp_start_idx but for frame b neighbor
	/*associate to each keypoint in frame b*/
    for (int bi=0; bi<b_num; bi++) { //for (int bi=0; bi<b_num; bi++) {

      const int *b_kp_neighbor_mat = __pointer_neighbor(b_frame_neighbor_mat, b_frame_neighbor_attr_count, b_frame_neighbor_kp_start_idx);
      int b_number_of_kp_neighbor = b_kp_neighbor_num[bi];
      /*apply mutual matches on the neighbor local features*/

      // mutual matches among the neighbors and calculate its LPG score
      lpg_score_matrix[ai*b_num+bi] = __MutualMatchAmongNeighbor(match_sm,
    			                                a_num,
						                        ai,
						                        a_kp_neighbor_mat, a_number_of_kp_neighbor, a_frame_neighbor_attr_count,
    					                        b_num,
						                        bi,
						                        b_kp_neighbor_mat, b_number_of_kp_neighbor, b_frame_neighbor_attr_count,
						                        gaussian_patch, gauss_y_size, gauss_x_size
						                        );
      b_frame_neighbor_kp_start_idx += 	b_number_of_kp_neighbor;
    }
    a_frame_neighbor_kp_start_idx += a_number_of_kp_neighbor;
  }

  /*match according to LPG score*/
  FilterMatch(lpg_score_matrix,
		      a_num, b_num,
			  neighbor_matches,
			  neighbor_distance);
  float sim_sum = 0.0;
  int i;
  int match_idx;
  for (i=0;i<a_num; i++) {

    match_idx = *(neighbor_matches+i);
    if (match_idx>=0) {
      //cout <<i<<endl;
      sim_sum+= (*(neighbor_distance+i)) * match_sm[i*b_num+match_idx];
    }
  }
  //cout <<"exit"<<endl;
  return sim_sum/sqrt(float(a_num*b_num));
}

/*
matches:  shape=[a_num]
distance: shape=[a_num]  when in PAIRWISE_RANSAC mode, the distance pointed field will changed. A 0.0 in *(distance+n) and a 1 in *(matches+n) indicates an outliner match.
a_pos:    shape[a_num, 2]   a_pos[:,0] are y. a_pos[:,1] are x
b_pos:    shape[b_num, 2]   b_pos[:,0] are y. b_pos[:,1] are x
a_frame_neighbor_mat:   shape [K,attr].   attr= frame_neighbor_attr_count.
                        a_frame_neighbor_mat[:,0]   int indexing each neighbor keypoints' position in a_pos
                        a_frame_neighbor_mat[:,1]   int The delta y distance of the each neigbor keypoint to the center nodes (neighbor_y - center_y)
                        a_frame_neighbor_mat[:,2]   int The delta x distance of the each neigbor keypoint to the center nodes (neighbor_x - center_x)
kp_neighbor_num: shape[a_num]:  records the number of neighbors for each of keypoint in kp set A
frame_neighbor_attr_count: number of atribute in a_frame_neighbor_mat (in this case = 3)

gaussian_patch: shape[Y,X]
gauss_y_size: Y
gauss_x_size: X

alg_type: PAIRWISE  PAIRWISE_POS  PAIRWISE_POSGRAPH PAIRWISE_RANSAC LPG_MATCH
*/

float C_PairwiseSimilarityPosGraph(const int a_num, const int b_num,
                                     int *matches,
                                   float *distance,
                                   const int *a_pos,
                                   const int *b_pos,
                                   const int dim_y, const int dim_x,
                                   const int *a_frame_neighbor_mat, const int *kp_neighbor_num, const int frame_neighbor_attr_count,   //a_frame_neighbor_mat in shape[K,3],  [K,0] is neighbor idx [K,1:2] is pos delta
                                   const float *gaussian_patch, const int gauss_y_size, const int gauss_x_size,
                                   const int alg_type,
								   float* _o_similarity_aux=(float*)0)
{
  if ((a_num==0)||(b_num==0)){
    return 0.0;
  }
  if(alg_type == PAIRWISE_RANSAC) {

    unsigned int match_idx_list[a_num];
	// probe the number of matched keypoints
	int match_count = ProbeMatchCount(matches, a_num, match_idx_list);
	//cout<<"matched point:"<<match_count<<endl;
    // fecth the matched kpoint pos from A and B
	// construct the keypoint correspondence matrix
	Mat matched_A_kp_pos_mat=Mat::ones(match_count,2,CV_32F);
	Mat matched_B_kp_pos_mat=Mat::ones(match_count,2,CV_32F);
	//cout<<"2"<<endl;
	// fill the keypoint correspondence matrix
	FetchMatchedKp_A(a_pos, matches, a_num, matched_A_kp_pos_mat);
	FetchMatchedKp_B(b_pos, matches, a_num, matched_B_kp_pos_mat);
	//cout<<"3"<<endl;
    // Run RANSACto find the fundamental matrix
	Mat mask;
	Mat C = cv::findFundamentalMat(matched_A_kp_pos_mat, matched_B_kp_pos_mat, FM_RANSAC, 7, 0.99f, mask);
	//cout<<"4"<<endl;
    // set the similarity of outlinered match to 0.0
	int inliner_count=0;
	for (int i=0;i<mask.rows; i++) {
      if (mask.at<uchar>(i)==0) {
		unsigned int outliner_distance_idx = match_idx_list[i];
		*(distance+outliner_distance_idx) = 0.0;
	  } else {
		inliner_count++;
	  }
	}
	//cout<<"inliner matched point:"<<inliner_count<<endl;
	// calculate the a2b frame similarity
    float sim_sum = 0.0;
    int matches_size = a_num;
    int i;
    int match_idx;
    for (i=0;i<matches_size; i++) {
      match_idx = *(matches+i);
      if (match_idx>=0) {
        sim_sum+= *(distance+i);
      }
    }
    return sim_sum/sqrt(float(a_num*b_num));
  }





  if(alg_type == PAIRWISE_STARGRAPH) {
	int gauss_center_y = gauss_y_size>>1;
	int gauss_center_x = gauss_x_size>>1;

    // get the number of matched local features
    unsigned int match_idx_list[a_num];
    // probe the number of matched keypoints
    int match_count = ProbeMatchCount(matches, a_num, match_idx_list);
    int a_x_mean=0;
    int a_y_mean=0;
    int b_x_mean=0;
    int b_y_mean=0;
    // calculate the mean pos of each matched local feature in a and b
    for (int i=0; i<match_count; i++ ) {
    	int a_pos_idx = match_idx_list[i]*2;
    	int b_pos_idx = matches[match_idx_list[i]]*2;
    	a_y_mean += a_pos[a_pos_idx];
    	a_x_mean += a_pos[a_pos_idx+1];
    	b_y_mean += b_pos[b_pos_idx];
    	b_x_mean += b_pos[b_pos_idx+1];
    }
	a_y_mean /= match_count;
	a_x_mean /= match_count;
	b_y_mean /= match_count;
	b_x_mean /= match_count;
    // calculate the image score
	float sim_sum = 0.0;
	for (int i=0; i<match_count; i++ ) {
    	int a_pos_idx = match_idx_list[i]*2;
    	int b_pos_idx = matches[match_idx_list[i]]*2;
    	//int a_idx = match_idx_list[i];
    	//int b_idx = matches[match_idx_list[i]];
    	int y_a = a_pos[a_pos_idx]-a_y_mean;
    	int x_a = a_pos[a_pos_idx+1]-a_x_mean;
    	int y_b = b_pos[b_pos_idx]-b_y_mean;
    	int x_b = b_pos[b_pos_idx+1]-b_x_mean;
    	int diff_x =  x_a-x_b;
    	int diff_y =  y_a-y_b;
        diff_y += gauss_center_y;
        diff_x += gauss_center_x;

    	int gauss_idx = diff_y*gauss_x_size+diff_x;
    	float gauss_score = gaussian_patch[gauss_idx];
    	sim_sum += distance[match_idx_list[i]]*gauss_score;
	}
    return sim_sum/sqrt(float(a_num*b_num));
  }
    /*
	srand((unsigned)time(NULL));

	  // find all matched keypoint
	#define param_min_sampled_kp_count  10
	int param_ransac_try_count = 50;
	float threshold = 32*3;//5.0;

	//int a_num = 100;
	//int b_num = 100;
	unsigned int matched_A_kp_idx[a_num];
	int matched_count = ProbeMatchCount(matches, a_num, matched_A_kp_idx);
	param_ransac_try_count = matched_count;
	MatrixXf matched_A_homo(matched_count,3);
	MatrixXf matched_B_homo(matched_count,3);
	matched_count = FetchMatchedCorrespondence(a_pos, b_pos, matches, a_num, matched_A_homo, matched_B_homo);
	// check if matched point is smaller than 16
	if (matched_count<param_min_sampled_kp_count) {
	  // if matched point smaller than the required number of matches return frame similarity 0.0
	  return 0.0;
	}
	// random select 8 matched keypoints and calculate the fundamental matrix
	MatrixXf fundamental_mat=MatrixXf::Ones(param_ransac_try_count, 9);
	MatrixXf coeff_mat(param_min_sampled_kp_count, 8);
	MatrixXf sampled_8point_A_homo(param_min_sampled_kp_count,3);
	MatrixXf sampled_8point_B_homo(param_min_sampled_kp_count,3);
	MatrixXf sampled_8point_A_homo_norm(param_min_sampled_kp_count,3);
	MatrixXf sampled_8point_B_homo_norm(param_min_sampled_kp_count,3);
	//point set normalization matrix
	MatrixXf Ha=MatrixXf::Zero(3,3);
	MatrixXf Hb=MatrixXf::Zero(3,3);

	// some matrix and variable
	MatrixXf equation_result_mat=MatrixXf::Constant(param_min_sampled_kp_count,1,-1.0);
	int random_kp_idx[param_min_sampled_kp_count];

	// variables for calculate the diszance to epline
	MatrixXf s(3,3);
	MatrixXf f_mat(3,3);
	ArrayXXf epline_B(matched_count,3);
	ArrayXXf distance_norm(matched_count,1);
	ArrayXXf A2epline_distance(matched_count,3);
	ArrayXXf b2epline_distance(matched_count,param_ransac_try_count);

	for (int i=0; i<param_ransac_try_count; i++) {
#ifdef VIPRACC_DEBUG
	  cout<<"#######---------------------------------------------------------------------------------------########"<<endl;
#endif
      //generate random N(param_min_sampled_kp_count) number from 0 to total matched keypoints(matched_count)
	  Random_0_N_Array(matched_count ,random_kp_idx, param_min_sampled_kp_count);
	  //fill the sampled_8point_A/B_homo with the sampled point
	  FillSampledKeypoint(matched_A_homo, random_kp_idx, param_min_sampled_kp_count, sampled_8point_A_homo);
	  FillSampledKeypoint(matched_B_homo, random_kp_idx, param_min_sampled_kp_count, sampled_8point_B_homo);
	  // calculate the point set normalization matrix
	  CalcuPointSetNormMat(sampled_8point_A_homo, Ha);
	  CalcuPointSetNormMat(sampled_8point_B_homo, Hb);
	  // Normalize the point set for A/B
	  sampled_8point_A_homo_norm = sampled_8point_A_homo*Ha.transpose();
	  sampled_8point_B_homo_norm = sampled_8point_B_homo*Hb.transpose();
#ifdef VIPRACC_DEBUG
	  cout<<sampled_8point_A_homo<<endl;
	  cout<<"######"<<endl;
	  cout<<sampled_8point_A_homo_norm<<endl;
	  cout<<Ha<<endl;
#endif
	  // construct coefficient matrix

	  coeff_mat.block<param_min_sampled_kp_count,1>(0,0) = sampled_8point_A_homo_norm.array().block<param_min_sampled_kp_count,1>(0,1)*sampled_8point_B_homo_norm.array().block<param_min_sampled_kp_count,1>(0,1);
	  coeff_mat.block<param_min_sampled_kp_count,1>(0,1) = sampled_8point_A_homo_norm.array().block<param_min_sampled_kp_count,1>(0,1)*sampled_8point_B_homo_norm.array().block<param_min_sampled_kp_count,1>(0,0);
	  coeff_mat.block<param_min_sampled_kp_count,1>(0,2) = sampled_8point_A_homo_norm.block<param_min_sampled_kp_count,1>(0,1);
	  coeff_mat.block<param_min_sampled_kp_count,1>(0,3) = sampled_8point_A_homo_norm.array().block<param_min_sampled_kp_count,1>(0,0)*sampled_8point_B_homo_norm.array().block<param_min_sampled_kp_count,1>(0,1);
	  coeff_mat.block<param_min_sampled_kp_count,1>(0,4) = sampled_8point_A_homo_norm.array().block<param_min_sampled_kp_count,1>(0,0)*sampled_8point_B_homo_norm.array().block<param_min_sampled_kp_count,1>(0,0);
	  coeff_mat.block<param_min_sampled_kp_count,1>(0,5) = sampled_8point_A_homo_norm.block<param_min_sampled_kp_count,1>(0,0);
	  coeff_mat.block<param_min_sampled_kp_count,1>(0,6) =                                                                             sampled_8point_B_homo_norm.block<param_min_sampled_kp_count,1>(0,1);
	  coeff_mat.block<param_min_sampled_kp_count,1>(0,7) =                                                                             sampled_8point_B_homo_norm.block<param_min_sampled_kp_count,1>(0,0);

	  // solve least square
	  fundamental_mat.block<1,8>(i,0)=(coeff_mat.transpose() * coeff_mat).ldlt().solve(coeff_mat.transpose() * equation_result_mat).transpose();
	  // rank-2 enforce of fundamental matrix
	  BDCSVD<MatrixXf> svd(Ha.transpose()*fundamental_mat.block<1,9>(i,0).reshaped(3,3)*Hb, ComputeThinU | ComputeThinV);
	  s = svd.singularValues().asDiagonal();
#ifdef VIPRACC_DEBUG
	  cout<<"#######singular value########"<<endl;
	  cout<<s<<endl;
#endif
	  s(2,2) = 0.0;
	  f_mat = svd.matrixU()*s*svd.matrixV().transpose();
	  // denomalze the fundamental matrix
	  //f_mat = Ha.transpose()*f_mat*Hb;
	  // calculate the epipplar-line in frame B
	  epline_B = matched_A_homo*f_mat;
#ifdef VIPRACC_DEBUG
	  cout<<"#######epline_B########"<<endl;
	  cout<<epline_B<<endl;
#endif
	  // calculate point B to the epline in frame B
	  distance_norm = epline_B.col(0).square()+epline_B.col(1).square();
	  distance_norm = distance_norm.sqrt();
	  A2epline_distance = epline_B*matched_B_homo.array();
	  b2epline_distance.col(i) = abs(A2epline_distance.rowwise().sum());
	  b2epline_distance.col(i) = b2epline_distance.col(i)/distance_norm;
#ifdef VIPRACC_DEBUG
	  cout<<"#######B to epline_B distance########"<<endl;
	  cout<<b2epline_distance.col(i)<<endl;
#endif
	}
	// select the best case
	MatrixXi thresholed_distance = (b2epline_distance<threshold).cast<int>();
	MatrixXi score = thresholed_distance.colwise().sum();
	// find the index of with maximum number of matches
	int max_idx = 0;
	int max_value = -1;
	for (int i=0; i<score.cols();i++) {
	  if (score(i)>max_value) {
	    max_idx = i;
	    max_value = score(i);
	  }
	}
	MatrixXi selected_matches = thresholed_distance.col(max_idx);
	for (int i=0; i<selected_matches.rows(); i++) {
	  if (selected_matches(i)==0) {
	    int a2bmatch_idx = matched_A_kp_idx[i];
	    matches[a2bmatch_idx] = -1;
      }
	}


	float sim_sum = 0.0;
	int matches_size = a_num;
	int i;
	int match_idx;
	for (i=0;i<matches_size; i++) {
	  match_idx = *(matches+i);
	  if (match_idx>=0) {
	    sim_sum+= *(distance+i);
	  }
	}
	//cout<<sim_sum/sqrt(float(a_num*b_num))<<endl;
	return sim_sum/sqrt(float(a_num*b_num));

  }*/

    // continue to calculate the a2b frame similarity GOTO THE FOLLOWING CODE DIRECTLY TO CALCULATE.
  //########################## for pairwise ######################################
  else if (alg_type==PAIRWISE) {
    float sim_sum = 0.0;
    int matches_size = a_num;
    int i;
    int match_idx;
    for (i=0;i<matches_size; i++) {
      match_idx = *(matches+i);
      if (match_idx>=0) {
        sim_sum+= *(distance+i);
      }
    }
    return sim_sum/sqrt(float(a_num*b_num));

  }


  else if (alg_type==RSS) {
    double sim_sum = 0.0;
    int i;
    int match_idx;
    int diff_x[a_num];
    int diff_y[a_num];
    float mean_diff_x = 0;
    float mean_diff_y = 0;
    int match_count = 0;

    // step 1: Calculate the diff and diff_mean
    for (i=0;i<a_num; i++) {
      match_idx = *(matches+i);
      if (match_idx>=0) { // found one correspondence
        //printf("%d  %d ---  (%d,%d).  (%d,%d)\n",i, match_idx ,a_pos[i*2], a_pos[i*2+1], b_pos[match_idx*2], b_pos[match_idx*2+1]);

    	match_count++;
    	diff_x[i] = a_pos[i*2+1]-b_pos[match_idx*2+1];   diff_y[i] = a_pos[i*2]-b_pos[match_idx*2];  // calculate the diff
    	mean_diff_x += diff_x[i]; mean_diff_y += diff_y[i];    // calculate the diff mean
    	//printf("%d, %d\n", diff_x[i], diff_y[i]);
      }
    }
    if (match_count==0)
      return 0; // if has no mutual matches return 0
    // calculate the mean diff
    mean_diff_x/=match_count;
    mean_diff_y/=match_count;
    //printf ("Mean %f, %f", mean_diff_x,mean_diff_y);

    // absolute the diff and find the maximum diff
    /*
    int max_diff_x = 100;
    int max_diff_y = 100;
    for (i=0;i<a_num; i++) {
      match_idx = *(matches+i);
      if (match_idx>=0) { // found one correspondence
    	//diff_x[i] = abs(diff_x[i]);  diff_y[i] = abs(diff_y[i]);
    	if (max_diff_x < abs(diff_x[i])) max_diff_x = abs(diff_x[i]);
    	if (max_diff_y < abs(diff_y[i])) max_diff_y = abs(diff_y[i]);
      }
    } */
    int max_diff_x = 0;
    int max_diff_y = 0;
    for (i=0;i<a_num; i++) {
      int x = a_pos[i*2+1];
      int y = a_pos[i*2];
      if (max_diff_x < x) max_diff_x = x;
      if (max_diff_y < y) max_diff_y = y;
    }
    for (i=0;i<b_num; i++) {
      int x = b_pos[i*2+1];
      int y = b_pos[i*2];
      if (max_diff_x < x) max_diff_x = x;
      if (max_diff_y < y) max_diff_y = y;
    }
    //printf ("%d  %d\n", max_diff_x,max_diff_y);
    //
    for (i=0;i<a_num; i++) {
      match_idx = *(matches+i);
      if (match_idx>=0) { // found one correspondence
        double score_x = pow(max_diff_x - abs(diff_x[i]-mean_diff_x), 2.0);
        double score_y = pow(max_diff_y - abs(diff_y[i]-mean_diff_y), 2.0);
        sim_sum += (score_x+score_y);
      }
    }
    //printf("%f\n", sim_sum/b_num);
    return float(sim_sum/b_num);

  }

  //######################## for pairwise pose graph #############################

  else if(alg_type==PAIRWISE_POSGRAPH) {
    //cout<<gauss_y_size<<" "<<gauss_x_size<<endl;
    //cout<<"New frame"<<endl;
    //__PrintMatches(matches,a_num);
    float sim_sum = 0.0;

    int gauss_center_y = gauss_y_size>>1;
    int gauss_center_x = gauss_x_size>>1;


    int correspondec_idx;

    int delta_a_y;
    int delta_a_x;

    int b_pos_idx;
    int b_y;
    int b_x;

    int delta_b_y;
    int delta_b_x;

    int diff_y;
    int diff_x;

    int gauss_idx;
    float gauss_score=0.0;

    // idx for a_frame_neighbor_mat
    int neighbor_idx   = 0;
    int delta_y_idx    = 1;
    int delta_x_idx    = 2;


    int matches_size = a_num;
    int match_in_b_idx;
    for (int i=0;i<matches_size; i++) { // i idx throuth each kp in a
      match_in_b_idx = *(matches+i);
      //cout<<i<<" "<<match_in_b_idx<<endl;
      // get the number of neighbor kp
      int neighbor_num = *(kp_neighbor_num+i);

      if (match_in_b_idx>=0) { // if has match
        // coordinate of coorespondent in B
        int kp_b_loc_idx = match_in_b_idx*2;
        int kp_b_loc_y = b_pos[kp_b_loc_idx];
        int kp_b_loc_x = b_pos[kp_b_loc_idx+1];
        // graph score and matched kp count in graph
        float graph_pos_score = 0.0;
        int matched_neighbor_count=0;
        int neighbor_ID=-2;
        //cout<<"neighbor_num "<<neighbor_num<<endl;
        // location for each attribute
        // iterate each neighbor
        for (int j=0;j<neighbor_num; j++) {
          // check if the neighbor is valid
          neighbor_ID = a_frame_neighbor_mat[neighbor_idx];
          //_o_similarity_aux
          if (neighbor_ID!=-1) {
            // check if the neighbor has matched

            correspondec_idx = matches[neighbor_ID];
            //cout <<"###"<<correspondec_idx<<"   "<<neighbor_ID<<endl;
            if (correspondec_idx>=0) {
              // get delta vector
              delta_a_y = a_frame_neighbor_mat[delta_y_idx];
              delta_a_x = a_frame_neighbor_mat[delta_x_idx];

              // get x y of correspondence
              b_pos_idx = correspondec_idx*2;
              b_y = b_pos[b_pos_idx];
              b_x = b_pos[b_pos_idx+1];
              //cout<<i<<"**"<<j<<endl;
              //cout<<correspondec_idx<<"&"<<b_y<<endl;
              // calculate pos delta of coorespondence
              delta_b_y = b_y - kp_b_loc_y;
              delta_b_x = b_x - kp_b_loc_x;
              //cout<<b_x<<"#"<<kp_b_loc_x<<endl;
              // difference of two delta
              diff_y = delta_b_y - delta_a_y;
              diff_x = delta_b_x - delta_a_x;
              //cout<<diff_y<<","<<diff_x<<"  "<<gauss_score<<endl;
              //cout<<delta_a_x<<" "<<delta_b_x<<endl;
              // translate the delta to gaussian center
              diff_y += gauss_center_y;
              diff_x += gauss_center_x;
              // look up gaussian score
              gauss_idx = diff_y*gauss_x_size+diff_x;
              //cout<<diff_y<<","<<diff_x<<endl;
              gauss_score = gaussian_patch[gauss_idx];


              //if ((delta_a_y==0)&&(delta_a_x==0)) gauss_score=1.0; // debug

              /*graph_pos_score calcuation with match score*/
              //float match_score = distance[neighbor_ID];
              //graph_pos_score += (gauss_score*match_score);
              /****/

              /*graph_pos_score calcuation without match score*/
              graph_pos_score += gauss_score;
              /****/


              //cout<<"   "<<j<<"  ("<<delta_a_y<<", "<<delta_a_x<<") ==>"<< gauss_score<<endl;
              matched_neighbor_count++;

              if (gauss_score<0.00) {

                cout<<delta_a_y<<"####"<<delta_a_x<<endl;
                cout<<delta_b_y<<"####"<<delta_b_x<<endl;
                cout<<b_y<<"####"<<b_x<<endl;
                cout<<kp_b_loc_y<<"####"<<kp_b_loc_x<<endl;
                cout<<gauss_center_y<<"####"<<gauss_center_x<<endl;
                cout<<diff_y<<", "<<diff_x<<"  "<<gauss_score<<endl;
              }

            }
          }
          neighbor_idx += frame_neighbor_attr_count;
          delta_y_idx  += frame_neighbor_attr_count;
          delta_x_idx  += frame_neighbor_attr_count;
        }
        //
        if (matched_neighbor_count>0) {
          graph_pos_score /= matched_neighbor_count;
          _o_similarity_aux[match_in_b_idx]=graph_pos_score;
          //graph_pos_score /= neighbor_num;
          //cout<<"#############==>"<<graph_pos_score<<endl;

        } else {
          //cout<<"No matched neighbor found "<<neighbor_num<<" "<<matched_neighbor_count<<" "<<correspondec_idx<<endl;
        }
        // pairwise match similarity with LPG score
        sim_sum         += distance[i]*graph_pos_score;
        // LPG score only
        //sim_sum         += graph_pos_score;
      } else { // if (match_in_b_idx>=0)
        // if no matches adjust the idx for a_frame_neighbor_mat
        int a_frame_neighbor_mat_skip_num = neighbor_num*frame_neighbor_attr_count;
        neighbor_idx   += a_frame_neighbor_mat_skip_num;
        delta_y_idx    += a_frame_neighbor_mat_skip_num;
        delta_x_idx    += a_frame_neighbor_mat_skip_num;

      }
    }
    //cout<<sim_sum<<endl;
    return sim_sum/sqrt(float(a_num*b_num));
  }
  return 0.0;
}

/* Clip a slice similarity matrix out from "match_sm" start from point (slice_H_start,slice_W_start) with size (slice_H, slice_W)
 * match_sm:               The similarity matrix with shape [match_sm_H, match_sm_W]
 * match_sm_H/match_sm_W:  The size of similarity matrix

 * slice_H_start,slice_W_start: The start point of the slice sm
 * slice_H, slice_W:            The size of slice sm
 *
 * _o_match_sm_slice:     The slice sm is returned by this pointer _o_match_sm_slice[h*slice_W + w] addressing the similarity of point (h,w)
 */
void __ClipMatchSlice(const float *match_sm, const int match_sm_W, const int match_sm_H,
                    const int slice_H_start, const int slice_W_start, const int slice_H, const int slice_W,
					float *_o_match_sm_slice)
{
  int slice_mem_idx = 0;
  int slice_H_mem_start_idx = slice_H_start*match_sm_W + slice_W_start;
  for (int i = 0; i<slice_H; i++) {
    for (int j=0; j< slice_W; j++) {
      _o_match_sm_slice[slice_mem_idx] = match_sm[slice_H_mem_start_idx+j];
      slice_mem_idx++;
    }
    slice_H_mem_start_idx += match_sm_W;
  }
}

// the match_sm slice version of FilterMatch
// match_sm_row_len/match_sm_col_len: This parameter indicate the row/col length of match_sm in float
int FilterMatchSlice(const float *match_sm, const int match_sm_row_len, const int match_sm_col_len,
                const int match_sm_H_start, const int match_sm_W_start, const int match_sm_H, const int match_sm_W,  //the patch in match_sm the FilterMatch done
                int *_o_matches,
                float *_o_distance)
{
  int row_max_idx[match_sm_H];

  float col_max[match_sm_W];
  int col_max_idx[match_sm_W];
  // initialize col_max
  int i;
  int j;
  int k;
  for (i=0; i<match_sm_W;i++) {
    col_max[i]=-20.0;
  }
  // scan the each row find maxima for both row and col
  float row_max;
  k = (match_sm_H_start*match_sm_row_len) + match_sm_W_start;
  int k_jmp_len = match_sm_row_len-match_sm_W;
  float value_at_k;
  for (i=0; i<match_sm_H; i++) {
    row_max =-20.0;
    for (j=0; j<match_sm_W; j++) {
      value_at_k = match_sm[k];
      // compare with row max value
      if (row_max<value_at_k) {
        row_max = value_at_k;
        row_max_idx[i] = j;
      }
      // compare with col max value
      if (col_max[j]<value_at_k) {
        col_max[j]=value_at_k;
        col_max_idx[j]=i;
      }
      //cout << match_sm[k]<<"  "<<k<<endl;
      k++;
    }
    k+=k_jmp_len;
  }
  // go through row max
  int idx_in_col_max;
  for (k=0;k<match_sm_H;k++) {
    // check match
    idx_in_col_max = row_max_idx[k];
    if (col_max_idx[idx_in_col_max]==k) { // if match
      _o_matches[k] = idx_in_col_max;
      _o_distance[k] = col_max[idx_in_col_max];
      //cout<<k <<"  "<<col_max[idx_in_col_max] << endl;
    }else{
      _o_matches[k] = -1;
      _o_distance[k] = 0.0;
    }
  }
  return 0;
}

// the match_sm slice version of FilterMatch
// match_sm: local feature cross similarity in column-major storage format
// match_sm_W/H: This parameter indicate the width/hight of the entire match_sm
// match_sm_slice_W/H: Slice similarity width/Hight
int FilterMatchSliceEfficient(const float *match_sm, const int match_sm_W, const int match_sm_H,
                const int slice_sm_H_start, const int slice_sm_W_start, const int match_sm_slice_H, const int match_sm_slice_W,  //the patch in match_sm the FilterMatch done
                int *_o_matches,
                float *_o_distance)
{
  int row_max_idx[match_sm_slice_H];
  float row_max[match_sm_slice_H];

  int col_max_idx[match_sm_slice_W];
  // initialize col_max
  int i;
  int j;
  int k;
  for (i=0; i<match_sm_slice_H;i++) {
	  row_max[i]=-20000.0;
  }
  // scan each row find maxima for both row and col
  float col_max;
  k = slice_sm_W_start*match_sm_H + slice_sm_H_start;
  int k_jmp_len = match_sm_H-match_sm_slice_H;
  float value_at_k;
  for (i=0; i<match_sm_slice_W; i++) {
	col_max =-20000.0;
    for (j=0; j<match_sm_slice_H; j++) {
      value_at_k = match_sm[k];
      // compare with row max value
      if (row_max[j]<value_at_k) {

        row_max[j] = value_at_k;
        row_max_idx[j] = i;

      }
      // compare with col max value
      if (col_max<value_at_k) {
        col_max=value_at_k;
        col_max_idx[i]=j;
      }
      //cout << match_sm[k]<<"  "<<k<<endl;
      k++;
    }
    k+=k_jmp_len;
  }

  // go through row max
  int idx_in_row_max;
  for (k=0;k<match_sm_slice_H;k++) {
    // check match
    idx_in_row_max = row_max_idx[k];
    if (col_max_idx[idx_in_row_max]==k) { // if match
      _o_matches[k] = idx_in_row_max;
      _o_distance[k] = row_max[k];

      //cout<<k <<"  "<<col_max[idx_in_col_max] << endl;
    }else{
      _o_matches[k] = -1;
      _o_distance[k] = 0.0;
    }

    //printf("%d,",_o_matches[k]);
  }
  //printf("\n");
  return 0;
}






// multithread function
// this functin is been assigned with several frame in B
void* __RowSeg(void* args)
{
   //cout<<"Start thread"<<endl;
   RowSegParam *param = (RowSegParam*)args;
   int frame_num = param->frame_num;
   int i;
   //cout<<"#"<<endl;
   int match_sm_row_len = param->sm_desc_W;
   int match_sm_col_len = param->sm_desc_H;

   int mach_slice_h_start = param->sm_slice_h_start;
   int mach_slice_w_start = param->sm_slice_w_start;

   int a_kp_num = param->a_kp_num;
   // create matches and distance array
   int* o_matches=param->_o_a2b_matches_seg;//int matches[a_kp_num];
   float* o_distance=param->_o_a2b_match_distance_seg;//float distance[a_kp_num];
   //cout<<"$"<<endl;
   /*global used variable*/
   const int* a_pos_patch = param->a_pos_patch;

   const float* gauss_patch = param->gaussian_patch;
   int    gauss_patch_y_size = param->gauss_y_size;
   int    gauss_patch_x_size = param->gauss_x_size;

   int alg_type = param->alg_type;
   //cout<<frame_num<<endl;
   int fm_dim_y = param->fm_dim_y;
   int fm_dim_x = param->fm_dim_x;

   const int *a_frame_neighbor_mat = param->a_frame_neighbor_mat;
   const int *kp_neighbor_num = param->a_frame_kp_neighbor_num;
   int frame_neighbor_attr_count = param->frame_neighbor_attr_count;
   /* index for kp in b*/
   int frame_b_kp_idx_start = 0;
   /* iterate through each frame in b */
   for (i=0; i<frame_num; i++) {
     //cout<<"Processing frame "<<i<<endl;
     /* calculate mutual match */
     // number of kp in current b frame
     int b_kp_num = param->b_frame_kp_num[i];


     // slice height and width in sm_desc
     // a_kp_num  b_kp_num
     // find mutual match
     FilterMatchSlice(param->sm_desc, match_sm_row_len, match_sm_col_len,
                mach_slice_h_start, mach_slice_w_start, a_kp_num, b_kp_num,  //the patch in match_sm the FilterMatch done
                o_matches+i*a_kp_num, //&matches[0],
                o_distance+i*a_kp_num); //&distance[0]);
     //cout<<"frame kp match finished "<<i<<endl;

     if (alg_type != ONLY_PAIRWISE_MATCH) { // include only when alg_type not equal ONLY_PAIRWISE_MATCH
       /*Calculate similarity of two frame based on the kp mutual match*/
       const int* b_pos_start_pointer = param->b_pos + (frame_b_kp_idx_start<<1);
       float similarity = C_PairwiseSimilarityPosGraph(a_kp_num, b_kp_num,
                                 o_matches+i*a_kp_num, //&matches[0],
                                 o_distance+i*a_kp_num, //&distance[0],
                                 a_pos_patch,
                                 b_pos_start_pointer,
                                 fm_dim_y, fm_dim_x,
                                 a_frame_neighbor_mat, kp_neighbor_num, frame_neighbor_attr_count,   //a_frame_neighbor_mat in shape[K,3],  [K,0] is neighbor idx [K,1:2] is pos delta
                                 gauss_patch, gauss_patch_y_size, gauss_patch_x_size,
                                 alg_type);
       // similarity of frame write back
       //cout << "simililarity for frame:"<< i <<" is " << similarity << endl;
       *((param->_o_sim_seg)+i) = similarity;
       /* incremantal by b_kp_num(b frame kp number) */
       frame_b_kp_idx_start += b_kp_num;
     }
     // increment start position in sm_desc slice
     mach_slice_w_start += b_kp_num;
     //cout<<b_kp_num<<endl;
   }
   pthread_exit(NULL);
}

/*
 * Get the b frame pos mat start column position in b_pos_mat with the given b frame idx in valid_b_frame
 */
const int* __Get_b_Frame_pos_mat_start_Pointer(TaskParam *p, const int frame_idx)
{
  return p->b_pos_mat+(*(p->b_frame_kp_start_idx+frame_idx))*p->b_pos_dim;
}
const int* __Get_b_Frame_neighbor_mat_start_Pointer(TaskParam *p, const int frame_idx)
{
  return p->b_frames_neighbor_mat+(*(p->b_frame_neighbor_start_idx+frame_idx))*p->frame_neighbor_attr_count;
}
const int* __Get_b_kp_in_frame_neighbor_num_start_Pointer(TaskParam *p, const int frame_idx)
{
  return p->b_kp_in_frame_neighbor_num+(*(p->b_frame_kp_start_idx+frame_idx));
}
const int __Get_sm_desc_b_frame_start_col(TaskParam *p, const int frame_idx)
{
  return *(p->b_frame_kp_start_idx+frame_idx);
}
/*The function for 1 query multi-database PoseGraph*/
// multithread function
// this functin is been assigned with several frame in B
void* __RowSeg2(void* args)
{
   //cout<<"Start thread"<<endl;
   RowSegParam *param = (RowSegParam*)args;
   TaskParam *pTasparam = param->pTaskParam;
   int frame_num = param->frame_num;
   int i;
   //cout<<"#"<<endl;
   int match_sm_row_len = param->sm_desc_W;
   int match_sm_col_len = param->sm_desc_H;

   int match_slice_h_start = param->sm_slice_h_start;
   int match_slice_w_start; // = param->sm_slice_w_start;
   int a_kp_num = param->a_kp_num;
   // create matches and distance array
   int* o_matches=param->_o_a2b_matches_seg;//int matches[a_kp_num];
   float* o_distance=param->_o_a2b_match_distance_seg;//float distance[a_kp_num];
   float* o_distance_aux = param->_o_a2b_match_distance_aux_seg; //distance_aux[a_kp_num];
   //cout<<"$"<<endl;
   /*global used variable*/
   const int* a_pos_patch = param->a_pos_patch;

   const float* gauss_patch = param->gaussian_patch;
   int    gauss_patch_y_size = param->gauss_y_size;
   int    gauss_patch_x_size = param->gauss_x_size;
   int alg_type = param->alg_type;
   //cout<<frame_num<<endl;
   int fm_dim_y = param->fm_dim_y;
   int fm_dim_x = param->fm_dim_x;

   const int *a_frame_neighbor_mat = param->a_frame_neighbor_mat;
   const int *a_kp_neighbor_num = param->a_frame_kp_neighbor_num;
   const int *b_kp_neighbor_num_pointer = param->b_kp_in_frame_neighbor_num;  // point to array holds the number of neighbor for each keypoint
   int frame_neighbor_attr_count = param->frame_neighbor_attr_count;
   /* index for kp in b*/
   int frame_b_kp_idx_start = 0;
   int frame_step_neighbor_count = 0;

   const int *valid_b_frame_start = param->valid_b_frame_start;
   /* iterate through each frame in b */
   for (i=0; i<frame_num; i++) {
	 //if (i==2) pthread_exit(NULL);  ///

     // ####printf("%d\n",i);
     //cout<<"Processing frame "<<i<<endl;
     /* calculate mutual match */
     //
	 int current_b_frame_idx =  *(valid_b_frame_start+i);

	 //printf("%d\n",current_b_frame_idx);
	 //if ((pTasparam->dbg == 831)&&(current_b_frame_idx==1876)) continue;

	 // number of kp in current b frame
     int b_kp_num = param->b_frame_kp_num[current_b_frame_idx];
     const int *current_b_kp_neighbor_num_pointer = __Get_b_kp_in_frame_neighbor_num_start_Pointer(pTasparam, current_b_frame_idx);
     //cout<<"frame kp match finished "<<i<<endl;
     const int *b_pos_start_pointer = __Get_b_Frame_pos_mat_start_Pointer(pTasparam, current_b_frame_idx);//param->b_pos + (frame_b_kp_idx_start<<1);
     // update the in frame neighbor start position
     const int *b_frame_neighbor_mat = __Get_b_Frame_neighbor_mat_start_Pointer(pTasparam, current_b_frame_idx);//param->b_frames_neighbor_mat + (frame_step_neighbor_count*frame_neighbor_attr_count);
     //cout<<"frame_step_neighbor_count "<<frame_step_neighbor_count<<"+"<< frame_step_neighbor_count*frame_neighbor_attr_count<<endl;
     match_slice_w_start = param->sm_slice_w_start + frame_b_kp_idx_start; //  __Get_sm_desc_b_frame_start_col(pTasparam, current_b_frame_idx);
     /*Only for PoseGraph to invert the match patch matrix*/
     if (alg_type == PAIRWISE_POSGRAPH) {

       // slice height and width in sm_desc
       // a_kp_num  b_kp_num
       // find mutual match
       FilterMatchSliceEfficient(param->sm_desc, match_sm_row_len, match_sm_col_len,
                  match_slice_h_start, match_slice_w_start, a_kp_num, b_kp_num,  //the patch in match_sm the FilterMatch done
                  o_matches+i*a_kp_num, //&matches[0],
                  o_distance+i*a_kp_num); //&distance[0]);

       //__PrintMatches(o_matches+i*a_kp_num, o_distance+i*a_kp_num,a_kp_num);
       //printf("%d,  %d", a_kp_num, b_kp_num);
       // get match of b in a 
       int   b2a_matches[b_kp_num];
       float b2a_distance[b_kp_num];
       int b_idx;
       
       int *a2b_match_start = o_matches+i*a_kp_num;
       float *a2b_distance_start = o_distance+i*a_kp_num;
       // initialize b2a_matches
       for (int kk=0; kk<b_kp_num; kk++) {
         b2a_matches[kk] = -1;
         b2a_distance[kk] = 0.0;
       }
       for (int kk=0; kk<a_kp_num; kk++) { // for all entry in A
         b_idx = *(a2b_match_start+kk); // get a matched b index
         if (b_idx!=-1){
           b2a_matches[b_idx] = kk;
           b2a_distance[b_idx] = *(a2b_distance_start+kk);
         }
       }
       //cout << "reverse map keypoint similarity patch finished"<< endl;
       // calculate PoseGraph

       //
       //cout<<"b_frame_neighbor_mat "<<b_frame_neighbor_mat<<endl;
       //cout<<"b_kp_num "<<b_kp_num<<endl;
       float lpg_graph_sim [b_kp_num];
       float similarity = C_PairwiseSimilarityPosGraph(b_kp_num, a_kp_num,
                                 b2a_matches, //&matches[0],
                                 b2a_distance, //&distance[0],
                                 b_pos_start_pointer,
                                 a_pos_patch,
                                 fm_dim_y, fm_dim_x,
                                 b_frame_neighbor_mat, current_b_kp_neighbor_num_pointer, frame_neighbor_attr_count,   //a_frame_neighbor_mat in shape[K,3],  [K,0] is neighbor idx [K,1:2] is pos delta
                                 gauss_patch, gauss_patch_y_size, gauss_patch_x_size,
                                 alg_type,
								 o_distance_aux+i*a_kp_num);
       //printf ("LPG similarity is %f \n", similarity);
       // similarity of frame write back
       //cout << "simililarity for frame:"<< i <<" is " << similarity << endl;
       *((param->_o_sim_seg)+i) = similarity;

       // update the total number of neighbors covered so far
       //for (int kp_in_frame=0; kp_in_frame<b_kp_num; kp_in_frame++){
       //  frame_step_neighbor_count += *(b_kp_neighbor_num_pointer+frame_b_kp_idx_start+kp_in_frame);
       //  //cout<<"kp in frame neighbor num"<< *(b_kp_neighbor_num_pointer+frame_b_kp_idx_start+kp_in_frame) <<endl;
       //}

       /* incremantal by b_kp_num(b frame kp number) */
       frame_b_kp_idx_start += b_kp_num;




     /*For LPG match algorithm*/
     } else if (alg_type == LPG_MATCH) {
    	 //cout<<"Processing "<<endl;
    	 float sm_slice[a_kp_num*b_kp_num];
    	 __ClipMatchSlice(param->sm_desc, match_sm_row_len, match_sm_col_len,
    			          match_slice_h_start, match_slice_w_start, a_kp_num, b_kp_num,
						  &sm_slice[0]);

         float similarity = C_PairwiseSimilarityLPG_match( &sm_slice[0],
        		                   a_kp_num,
								   a_pos_patch,
								   a_frame_neighbor_mat, a_kp_neighbor_num, frame_neighbor_attr_count,

								   b_kp_num,
                                   b_pos_start_pointer,
                                   b_frame_neighbor_mat, b_kp_neighbor_num_pointer+frame_b_kp_idx_start, frame_neighbor_attr_count,
                                   gauss_patch, gauss_patch_y_size, gauss_patch_x_size,
                                   alg_type);

         // similarity of frame write back
         //cout << "simililarity for frame:"<< i <<" is " << similarity << endl;
         *((param->_o_sim_seg)+i) = similarity;
         // update the total number of neighbors covered so far
         for (int kp_in_frame=0; kp_in_frame<b_kp_num; kp_in_frame++){
           frame_step_neighbor_count += *(b_kp_neighbor_num_pointer+frame_b_kp_idx_start+kp_in_frame);
           //cout<<"kp in frame neighbor num"<< *(b_kp_neighbor_num_pointer+frame_b_kp_idx_start+kp_in_frame) <<endl;
         }
         /* incremantal by b_kp_num(b frame kp number) */
         frame_b_kp_idx_start += b_kp_num;
     }




     else { //if ((alg_type != ONLY_PAIRWISE_MATCH)&&(alg_type != PAIRWISE_POSGRAPH)) { // include only when alg_type not equal ONLY_PAIRWISE_MATCH or PAIRWISE_POSGRAPH
       /*Calculate similarity of two frame based on the kp mutual match*/
       // slice height and width in sm_desc
       // a_kp_num  b_kp_num
       // find mutual match
       FilterMatchSliceEfficient(param->sm_desc, match_sm_row_len, match_sm_col_len,
                    match_slice_h_start, match_slice_w_start, a_kp_num, b_kp_num,  //the patch in match_sm the FilterMatch done
                    o_matches+i*a_kp_num, //&matches[0],
                    o_distance+i*a_kp_num); //&distance[0]);
       //__PrintMatches(o_matches+i*a_kp_num, o_distance+i*a_kp_num,a_kp_num);
       //printf("%d,  %d", a_kp_num, b_kp_num);
       //const int* b_pos_start_pointer = param->b_pos + (frame_b_kp_idx_start<<1);
       //cout << "other algs"<< endl;
       float similarity = C_PairwiseSimilarityPosGraph(a_kp_num, b_kp_num,
                                 o_matches+i*a_kp_num, //&matches[0],
                                 o_distance+i*a_kp_num, //&distance[0],
                                 a_pos_patch,
                                 b_pos_start_pointer,
                                 fm_dim_y, fm_dim_x,
								 (const int *)0, (const int *)0, 0,
								 //b_frame_neighbor_mat, b_kp_neighbor_num_pointer+frame_b_kp_idx_start, frame_neighbor_attr_count,   //a_frame_neighbor_mat in shape[K,3],  [K,0] is neighbor idx [K,1:2] is pos delta
								 gauss_patch, gauss_patch_y_size, gauss_patch_x_size,
								 //gauss_patch, gauss_patch_y_size, gauss_patch_x_size,
                                 alg_type);
       //printf ("Here 2\n\n");
       //printf ("PW similarity is %f \n", similarity);
       // similarity of frame write back
       //cout << "simililarity for frame:"<< i <<" is " << similarity << endl;
       *((param->_o_sim_seg)+i) = similarity;
       /* incremantal by b_kp_num(b frame kp number) */
       frame_b_kp_idx_start += b_kp_num;
     }
     // increment start position in sm_desc slice
     //match_slice_w_start += b_kp_num;
     //cout<<b_kp_num<<endl;

     //break;
   }
   pthread_exit(NULL);
}




extern "C" {

int FilterMatch(const float *match_sm,
                int match_sm_H, int match_sm_W,
                int *_o_matches,
                float *_o_distance)
{
  int row_max_idx[match_sm_H];

  float col_max[match_sm_W];
  int col_max_idx[match_sm_W];
  // initialize col_max
  int i;
  int j;
  int k;
  for (i=0; i<match_sm_W;i++) {
    col_max[i]=-20.0;
  }
  // scan the each row find maxima for both row and col
  float row_max;
  k =0;
  float value_at_k;
  for (i=0; i<match_sm_H; i++) {
    row_max =-20.0;
    for (j=0; j<match_sm_W; j++) {
      value_at_k = match_sm[k];
      // compare with row max value
      if (row_max<value_at_k) {
        row_max = value_at_k;
        row_max_idx[i] = j;
      }
      // compare with col max value
      if (col_max[j]<value_at_k) {
        col_max[j]=value_at_k;
        col_max_idx[j]=i;
      }
      //cout << match_sm[k]<<"  "<<k<<endl;
      k++;
    }
  }
  // go through row max
  int idx_in_col_max;
  for (k=0;k<match_sm_H;k++) {
    // check match
    idx_in_col_max = row_max_idx[k];
    if (col_max_idx[idx_in_col_max]==k) { // if match
      _o_matches[k] = idx_in_col_max;
      _o_distance[k] = col_max[idx_in_col_max];
      //cout<<k <<"  "<<col_max[idx_in_col_max] << endl;
    }else{
      _o_matches[k] = -1;
      _o_distance[k] = 0.0;
    }
  }
  return 0;
}




int C_PairwiseSimilarityRowPosGraph_MultiThread(
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
                                    float* _o_a2b_match_distance)
{

  int i;
  /* assign b frame interval for each thread */
  int frame_base_num = b_frame_num/thread_num;
  int reminder_frame_num = b_frame_num%thread_num;
  int thread_frame_num_array[thread_num];
  /*assign each thread number of frame to process*/
  for (i=0; i<thread_num; i++) {
    thread_frame_num_array[i] = frame_base_num;
    if (reminder_frame_num>0) {
      thread_frame_num_array[i]++;
      reminder_frame_num--;
    }
  }
  /* In case frame number smaller than required thread number*/
  int true_thread_num=0;
  for (i=0; i<thread_num; i++) {
    if (thread_frame_num_array[i]==0) {
      break;
    }
    true_thread_num++;
  }

  //cout<<"Creating thread "<<true_thread_num << endl;

  pthread_t tids[true_thread_num];

  RowSegParam param[true_thread_num]={};
  int thread_start_frame_idx = 0;
  int thread_end_frame_idx = 0;
  long b_all_kp_idx=0;
  for(int i = 0; i < true_thread_num; i++){
    thread_start_frame_idx = thread_end_frame_idx;
    thread_end_frame_idx  = thread_start_frame_idx+thread_frame_num_array[i];
    /* assign param for each Thread */
    param[i].frame_num = thread_frame_num_array[i]; // the number of frame to process
    // similarity matrix patch
    param[i].sm_desc = sm_desc; // start address of sm patch
    // sm_desc slice rectangle
    param[i].sm_slice_h_start = 0;  // sm patch slice height
    param[i].sm_slice_w_start = b_all_kp_idx;  // sm patch slice width

    // the dimension of entire sm_desc
    param[i].sm_desc_H = sm_desc_H;
    param[i].sm_desc_W = sm_desc_W;
    //
    param[i].a_pos_patch               = a_pos_patch;
    param[i].a_kp_num                  = a_kp_num;
    param[i].a_frame_neighbor_mat      = a_frame_neighbor_mat;
    param[i].a_frame_kp_neighbor_num           = kp_neighbor_num;
    param[i].frame_neighbor_attr_count = frame_neighbor_attr_count;
    //
    param[i].b_pos          = b_pos_mat+(b_all_kp_idx<<1);
    param[i].b_frame_kp_num = b_frame_kp_num + thread_start_frame_idx; // holds number of kp in each frame in b
    //
    param[i].gaussian_patch = gaussian_patch;
    param[i].gauss_y_size   = gauss_y_size;
    param[i].gauss_x_size   = gauss_x_size;
    //
    param[i].alg_type = alg_type;

    param[i].fm_dim_x = fm_dim_x;
    param[i].fm_dim_y = fm_dim_y;
    //
    param[i]._o_sim_seg = _o_sm_row+thread_start_frame_idx;

    param[i]._o_a2b_matches_seg = _o_a2b_matches+(thread_start_frame_idx*a_kp_num);
    param[i]._o_a2b_match_distance_seg = _o_a2b_match_distance+(thread_start_frame_idx*a_kp_num);

    for (int j=thread_start_frame_idx; j<thread_end_frame_idx; j++) {
      b_all_kp_idx += *(b_frame_kp_num+j);
    }
    int ret = pthread_create(&tids[i], NULL, __RowSeg, (void*)(&param[i]));
    if (ret != 0){
           cout << "pthread_create error: error_code=" << ret << endl;
    }
  }


  // wait thread finish
  for(i = 0; i < true_thread_num; i++)
    pthread_join(tids[i], NULL);

  return 0;
}



void __FindFrameSpatialClosestNeighbor(const int* frame_pos_patch, const int frame_kp_num,
		                        const int neighbor_win_y, const int neighbor_win_x,
								const int frame_neighbor_attr_count,
							    int *_o_frames_neighbor_mat,  int *_o_kp_in_frame_neighbor_num)
{




  // iterate each keypoint to find its closest neighbor

  int win_y_radius = neighbor_win_y>>1;
  int win_x_radius = neighbor_win_x>>1;
  //cout << "Start to Find closest neighbor for A"<< win_y_radius<< win_x_radius << endl;
  int  coordinate_size = 2; //each coordinate in frame_pos_patch is composed by 2 integers
  int delta_y, delta_x, delta_y_abs, delta_x_abs; // kp coordinate delta from neighbor to the center node: neighbor_xy - center_xy
  int center_node_y, center_node_x; // center node coordinate
  int frames_neighbor_mat_idx = 0;  // pointed to each neighbor in _o_frames_neighbor_mat
  int kp_neighbor_num = 0;  // the instant number of neighbor for each center node
  for (int i = 0; i<frame_kp_num; i++) {
	// start a new center node
	center_node_y = *(frame_pos_patch + i*coordinate_size);   // get center node y coordinate
	center_node_x = *(frame_pos_patch + i*coordinate_size+1); // get center node x coordinate
	kp_neighbor_num = 0; // reset the neighbor number of this center node
    for (int j = 0; j<frame_kp_num; j++) {
      delta_y = *(frame_pos_patch + j*coordinate_size)  -center_node_y;
      delta_x = *(frame_pos_patch + j*coordinate_size+1)-center_node_x;

      delta_y_abs = delta_y;
      delta_x_abs = delta_x;
      if (delta_y_abs<0) delta_y_abs=-delta_y_abs; // absolute value for delta_y
      if (delta_x_abs<0) delta_x_abs=-delta_x_abs; // absolute value for delta_x
      // if found spatial closest neighbor
      if ((delta_y_abs<win_y_radius) && (delta_x_abs<win_x_radius) ) {
    	int frames_neighbor_mat_base_idx = frames_neighbor_mat_idx*frame_neighbor_attr_count;
        *(_o_frames_neighbor_mat+frames_neighbor_mat_base_idx+0) = j;
        *(_o_frames_neighbor_mat+frames_neighbor_mat_base_idx+1) = delta_y;
        *(_o_frames_neighbor_mat+frames_neighbor_mat_base_idx+2) = delta_x;

        //cout << "Center node:" << center_node_y << " "<< center_node_x << endl;
        //cout << "Leaf node:" << *(frame_pos_patch + j*coordinate_size)<<" "<< *(frame_pos_patch + j*coordinate_size+1)<< endl;
        //cout << "Find new neighbor"<< endl;
        //cout << delta_y<< endl;
        //cout << delta_x<< endl<<endl;

        kp_neighbor_num++;
        frames_neighbor_mat_idx++;
      }
    }
    _o_kp_in_frame_neighbor_num[i] = kp_neighbor_num;
  }
  return;
}




int GetFrameKpNum(const int * b_frame_kp_num, const int frame_idx)
{
  return b_frame_kp_num[frame_idx];
}

/*
b_frames_neighbor_mat       : neighbor info of each kp in DB
b_kp_in_frame_neighbor_num  : neighbor number for each kp in DB
b_frame_frame_start_idx     : idx for start of each frame in b_frames_neighbor_mat.  For start location of frame_2 ==> b_frame_frame_start_idx[2]*frame_neighbor_attr_count
frame_neighbor_attr_count   : number of int32 for each neighbor info in b_frames_neighbor_mat
*/
int C_PairwiseSimilarityVersatile_MultiThread(
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
									float* _o_a2b_match_distance_aux)
{

  int i;
  /* assign b frame interval for each thread */
  int frame_base_num = valid_b_frame_num/thread_num;
  int reminder_frame_num = valid_b_frame_num%thread_num;
  int thread_frame_num_array[thread_num];
  /*assign each thread number of frame to process*/
  for (i=0; i<thread_num; i++) {
    thread_frame_num_array[i] = frame_base_num;
    if (reminder_frame_num>0) {
      thread_frame_num_array[i]++;
      reminder_frame_num--;
    }
  }
  /* In case frame number smaller than required thread number*/
  int true_thread_num=0;
  for (i=0; i<thread_num; i++) {
    if (thread_frame_num_array[i]==0) {
      break;
    }
    true_thread_num++;
  }

  int* a_frames_neighbor_mat      = (int*)0;
  int* a_kp_in_frame_neighbor_num = (int*)0;

  if (alg_type==LPG_MATCH) {
	/*Find the closest neighbors for each local feature in a*/
	/*allocate memory*/
	// assume each keypoint associate all the keypoints in the frame as its Spatial closest neighbor
	a_frames_neighbor_mat      = (int*)malloc(a_kp_num*a_kp_num*frame_neighbor_attr_count*sizeof(int));
	a_kp_in_frame_neighbor_num = (int*)malloc(a_kp_num*sizeof(int));
    __FindFrameSpatialClosestNeighbor(a_pos_patch, a_kp_num,
    		                          fm_dim_y, fm_dim_x,
									  frame_neighbor_attr_count,
									  a_frames_neighbor_mat, a_kp_in_frame_neighbor_num);
  }
  // fill the task paramere
  TaskParam taskparam;
  taskparam.dbg = dbg;
  taskparam.sm_desc   = sm_desc;
  taskparam.sm_desc_H =sm_desc_H;
  taskparam.sm_desc_W =sm_desc_W;
	// local feature coordinate range
  taskparam.fm_dim_y =fm_dim_y;
  taskparam.fm_dim_x =fm_dim_x;
	// frame a local feature pos matrix
  taskparam.a_pos_patch =a_pos_patch;
  taskparam.a_kp_num    =a_kp_num;    // a is query
  taskparam.a_pos_dim    =2;
  // frame b local feature pos matrix
  taskparam.b_pos_mat            =b_pos_mat;
  taskparam.b_pos_dim    =2;
  taskparam.b_frame_num          =b_frame_num;
  taskparam.b_frame_kp_num       =b_frame_kp_num;
  taskparam.b_frame_kp_start_idx =b_frame_kp_start_idx;  //local feature idx of the first local feature in each frame
  // frame b local feature neigbor matrix
  taskparam.b_frames_neighbor_mat          =b_frames_neighbor_mat;
  taskparam.b_kp_in_frame_neighbor_num     =b_kp_in_frame_neighbor_num; // number of neighbor of each local feature for all b frame
  taskparam.b_frame_neighbor_start_idx     =b_frame_neighbor_start_idx; // row idx of the start of the neighbor matrix of each b frame
  taskparam.frame_neighbor_attr_count      =frame_neighbor_attr_count;   // column size of b_frames_neighbor_mat

  taskparam.valid_b_frame     =valid_b_frame;    // idx of the valid b frame
  taskparam.valid_b_frame_num =valid_b_frame_num; // total number of valid b frame

  taskparam.gaussian_patch =gaussian_patch;
  taskparam.gauss_y_size   =gauss_y_size;
  taskparam.gauss_x_size   =gauss_x_size;

  taskparam.thread_num =thread_num;
  taskparam.alg_type   =alg_type;
  taskparam._o_sm_row  =_o_sm_row;
  // when alg_type == ONLY_PAIRWISE_MATCH the following return the matched result of keypoint in frame A to all in frame B

	// when alg_type == PAIRWISE_RANSAC the following return the matched result of keypoint in frame A to all in frame B
	// a 0 in _o_a2b_match_distance while the corresponding 1 in _o_a2b_matches means an outliner
  taskparam._o_a2b_matches        =_o_a2b_matches;
  taskparam._o_a2b_match_distance =_o_a2b_match_distance;



  pthread_t tids[true_thread_num];

  RowSegParam param[true_thread_num]={};
  int thread_start_frame_idx = 0;
  int thread_end_frame_idx = 0;
  long b_all_kp_idx=0;
  
  //int b_kp_in_frame_neighbor_num_start = 0;
  for(int i = 0; i < true_thread_num; i++){
    thread_start_frame_idx = thread_end_frame_idx;
    thread_end_frame_idx  = thread_start_frame_idx+thread_frame_num_array[i];
    /* assign param for each Thread */
    param[i].frame_num = thread_frame_num_array[i]; // the number of frame to process
    param[i].start_frame = 0;
    param[i].valid_b_frame_start =valid_b_frame+thread_start_frame_idx;
    // similarity matrix patch
    param[i].sm_desc = sm_desc; // start address of sm patch
    // sm_desc slice rectangle
    param[i].sm_slice_h_start = 0;  // sm patch slice height
    param[i].sm_slice_w_start = b_all_kp_idx;  // sm patch slice width

    // the dimension of entire sm_desc
    param[i].sm_desc_H = sm_desc_H;
    param[i].sm_desc_W = sm_desc_W;
    //
    param[i].a_pos_patch               = a_pos_patch;
    param[i].a_kp_num                  = a_kp_num;

    param[i].a_frame_neighbor_mat      = a_frames_neighbor_mat;
    param[i].a_frame_kp_neighbor_num   = a_kp_in_frame_neighbor_num; // only for LPG-MATCH

    param[i].frame_neighbor_attr_count = frame_neighbor_attr_count;
    //
    param[i].b_pos          = b_pos_mat;//b_pos_mat+(b_all_kp_idx<<1);
    param[i].b_frame_kp_num = b_frame_kp_num;//b_frame_kp_num + thread_start_frame_idx; // holds number of kp in each frame in b

    param[i].b_frames_neighbor_mat       = b_frames_neighbor_mat;//b_frames_neighbor_mat+(*(b_frame_neighbor_start_idx+thread_start_frame_idx))*frame_neighbor_attr_count;
    param[i].b_kp_in_frame_neighbor_num  = b_kp_in_frame_neighbor_num;//b_kp_in_frame_neighbor_num+b_all_kp_idx;
    //cout<<"b_kp_in_frame_neighbor_num "<<b_kp_in_frame_neighbor_num<<endl;
    param[i].b_frame_neighbor_start_idx  = b_frame_neighbor_start_idx;  //b_frame_neighbor_start_idx+thread_start_frame_idx;
    param[i].b_frame_kp_start_idx_segment = b_frame_kp_start_idx+thread_start_frame_idx;
    //
    param[i].gaussian_patch = gaussian_patch;
    param[i].gauss_y_size   = gauss_y_size;
    param[i].gauss_x_size   = gauss_x_size;
    //
    param[i].alg_type = alg_type;

    param[i].fm_dim_x = fm_dim_x;
    param[i].fm_dim_y = fm_dim_y;
    //

    //
    param[i]._o_sim_seg = _o_sm_row+thread_start_frame_idx;

    param[i]._o_a2b_matches_seg = _o_a2b_matches+(thread_start_frame_idx*a_kp_num);
    param[i]._o_a2b_match_distance_seg = _o_a2b_match_distance+(thread_start_frame_idx*a_kp_num);
    param[i]._o_a2b_match_distance_aux_seg = _o_a2b_match_distance_aux+(thread_start_frame_idx*a_kp_num);
    param[i].pTaskParam = &taskparam;

    for (int j=thread_start_frame_idx; j<thread_end_frame_idx; j++) {
      b_all_kp_idx += *(b_frame_kp_num+(*(valid_b_frame+j)));
    }
    int ret = pthread_create(&tids[i], NULL, __RowSeg2, (void*)(&param[i]));
    if (ret != 0){
           cout << "pthread_create error: error_code=" << ret << endl;
    }
  }


  // wait thread finish
  for(i = 0; i < true_thread_num; i++)
    pthread_join(tids[i], NULL);

  if (alg_type==LPG_MATCH) {
	free((void*)a_frames_neighbor_mat);
	free((void*)a_kp_in_frame_neighbor_num);
  }
  return 0;
}



}//end of extern C




/******
 * Fetch the matched correspondence into the output matrix.
 *a_pos: pointer to position of keypont group A in normal coordinate
 *b_pos: pointer to position of keypont group B in normal coordinate
 *a2b_matches: matched distance of A in B
 *a2b_match_len: length of a2b_matches
 *_o_A/B_mat: reference to Output matched correspondences in homogeneous coordinate. This matrix must in size (a2b_match_len,3) and in type float
 *****/

/*
int main(int argc, char **argv) {
  srand((unsigned)time(NULL));

  ArrayXXf Point3D_array=ArrayXXf::Random(100, 4); // x,y,z
  Point3D_array.col(2)=ArrayXXf::Random(100,1)*2.0+5.0;
  //Point3D_array = Point3D_array;
  //Point3D_array = Point3D_array;

  MatrixXf Point3D_mat0 = Point3D_array.matrix();

  Point3D_mat0.block<100,1>(0,3)=MatrixXf::Ones(100,1);
  MatrixXf trans_mat(4,4);
  float deg = 3.141593*(23.0/180.0);
  trans_mat<<   cos(deg),     0,   sin(deg),    0,
		        0,            1,   0,           0,
				-sin(deg),    0,   cos(deg),    0,
			    0,            0,   0,           1;
  MatrixXf Point3D_mat1 = Point3D_mat0*trans_mat.transpose();

  cout << Point3D_mat0 <<endl<<"####"<<endl;
  cout << Point3D_mat1<<endl;

  int a_pos[200];
  int b_pos[200];
  float z0;
  float z1;
  int a2b_matches[100];
  for (int i=0; i<200; i+=2) {
    z0 = Point3D_mat0(i/2, 2);
    z1 = Point3D_mat1(i/2, 2);
	a_pos[i]   = (Point3D_mat0(i/2, 1)/z0)*500;
	a_pos[i+1] = (Point3D_mat0(i/2, 0)/z0)*500;
	b_pos[i]   = (Point3D_mat1(i/2, 1)/z1)*500;
	b_pos[i+1]   = (Point3D_mat1(i/2, 0)/z1)*500;
	a2b_matches[i/2]=i/2;

  }
  int outliner_idx[30];
  for (int i=0; i<10;i++){
    int idx =i;//rand()%100;
    a_pos[idx<<1]=rand()%100;
    a_pos[(idx<<1)+1]=rand()%100;
    outliner_idx[i]=idx;
    //a2b_matches[idx]=-1;
  }
  a_pos[20<<1]=rand()%100;
  a_pos[(20<<1)+1]=rand()%100;
  a_pos[30<<1]=rand()%100;
  a_pos[(30<<1)+1]=rand()%100;
  a_pos[40<<1]=rand()%100;
  a_pos[(40<<1)+1]=rand()%100;


  // initialize random number generator
  srand((unsigned)time(NULL));

  // find all matched keypoint
#define param_min_sampled_kp_count  10
  int param_ransac_try_count = 50;
  float threshold = 10.0;

  int a_num = 100;
  int b_num = 100;
  unsigned int matched_A_kp_idx[100];
  int matched_count = ProbeMatchCount(a2b_matches, a_num, matched_A_kp_idx);
  MatrixXf matched_A_homo(matched_count,3);
  MatrixXf matched_B_homo(matched_count,3);
  matched_count = FetchMatchedCorrespondence(a_pos, b_pos, a2b_matches, a_num, matched_A_homo, matched_B_homo);
  // check if matched point is smaller than 16
  if (matched_count<param_min_sampled_kp_count) {
	// if matched point smaller than the required number of matches return frame similarity 0.0
    return 0.0;
  }
  // random select 8 matched keypoints and calculate the fundamental matrix
  MatrixXf fundamental_mat=MatrixXf::Ones(param_ransac_try_count, 9);
  MatrixXf coeff_mat(param_min_sampled_kp_count, 8);
  MatrixXf sampled_8point_A_homo(param_min_sampled_kp_count,3);
  MatrixXf sampled_8point_B_homo(param_min_sampled_kp_count,3);
  MatrixXf sampled_8point_A_homo_norm(param_min_sampled_kp_count,3);
  MatrixXf sampled_8point_B_homo_norm(param_min_sampled_kp_count,3);
  //point set normalization matrix
  MatrixXf Ha=MatrixXf::Zero(3,3);
  MatrixXf Hb=MatrixXf::Zero(3,3);

  // some matrix and variable
  MatrixXf equation_result_mat=MatrixXf::Constant(param_min_sampled_kp_count,1,-1.0);
  int random_kp_idx[param_min_sampled_kp_count];

  // variables for calculate the diszance to epline
  MatrixXf s(3,3);
  MatrixXf f_mat(3,3);
  ArrayXXf epline_B(matched_count,3);
  ArrayXXf distance_norm(matched_count,1);
  ArrayXXf A2epline_distance(matched_count,3);
  ArrayXXf distance(matched_count,param_ransac_try_count);

  for (int i=0; i<param_ransac_try_count; i++) {
	//generate random N(param_min_sampled_kp_count) number from 0 to total matched keypoints(matched_count)
    Random_0_N_Array(matched_count ,random_kp_idx, param_min_sampled_kp_count);
    //fill the sampled_8point_A/B_homo with the sampled point
    FillSampledKeypoint(matched_A_homo, random_kp_idx, param_min_sampled_kp_count, sampled_8point_A_homo);
    FillSampledKeypoint(matched_B_homo, random_kp_idx, param_min_sampled_kp_count, sampled_8point_B_homo);
    // calculate the point set normalization matrix
    CalcuPointSetNormMat(sampled_8point_A_homo, Ha);
    CalcuPointSetNormMat(sampled_8point_B_homo, Hb);
    // Normalize the point set for A/B
    sampled_8point_A_homo_norm = sampled_8point_A_homo*Ha.transpose();
    sampled_8point_B_homo_norm = sampled_8point_B_homo*Hb.transpose();
    //cout<<sampled_8point_A_homo_norm<<endl;
    //cout<<Ha<<endl;
    // construct coefficient matrix

    coeff_mat.block<param_min_sampled_kp_count,1>(0,0) = sampled_8point_A_homo_norm.array().block<param_min_sampled_kp_count,1>(0,1)*sampled_8point_B_homo_norm.array().block<param_min_sampled_kp_count,1>(0,1);
    coeff_mat.block<param_min_sampled_kp_count,1>(0,1) = sampled_8point_A_homo_norm.array().block<param_min_sampled_kp_count,1>(0,1)*sampled_8point_B_homo_norm.array().block<param_min_sampled_kp_count,1>(0,0);
    coeff_mat.block<param_min_sampled_kp_count,1>(0,2) = sampled_8point_A_homo_norm.block<param_min_sampled_kp_count,1>(0,1);
    coeff_mat.block<param_min_sampled_kp_count,1>(0,3) = sampled_8point_A_homo_norm.array().block<param_min_sampled_kp_count,1>(0,0)*sampled_8point_B_homo_norm.array().block<param_min_sampled_kp_count,1>(0,1);
    coeff_mat.block<param_min_sampled_kp_count,1>(0,4) = sampled_8point_A_homo_norm.array().block<param_min_sampled_kp_count,1>(0,0)*sampled_8point_B_homo_norm.array().block<param_min_sampled_kp_count,1>(0,0);
    coeff_mat.block<param_min_sampled_kp_count,1>(0,5) = sampled_8point_A_homo_norm.block<param_min_sampled_kp_count,1>(0,0);
    coeff_mat.block<param_min_sampled_kp_count,1>(0,6) =                                                                             sampled_8point_B_homo_norm.block<param_min_sampled_kp_count,1>(0,1);
    coeff_mat.block<param_min_sampled_kp_count,1>(0,7) =                                                                             sampled_8point_B_homo_norm.block<param_min_sampled_kp_count,1>(0,0);

    // solve least square
    fundamental_mat.block<1,8>(i,0)=(coeff_mat.transpose() * coeff_mat).ldlt().solve(coeff_mat.transpose() * equation_result_mat).transpose();
    // rank-2 enforce of fundamental matrix
    BDCSVD<MatrixXf> svd(Ha.transpose()*fundamental_mat.block<1,9>(i,0).reshaped(3,3)*Hb, ComputeThinU | ComputeThinV);
    s = svd.singularValues().asDiagonal();
    cout<<s<<endl;
    s(2,2) = 0.0;
    f_mat = svd.matrixU()*s*svd.matrixV().transpose();
    // denomalze the fundamental matrix
    //f_mat = Ha.transpose()*f_mat*Hb;
    // calculate the epipplar-line in frame B
    epline_B = matched_A_homo*f_mat;
    cout<<epline_B<<endl;
    // calculate point B to the epline in frame B
    distance_norm = epline_B.col(0).square()+epline_B.col(1).square();
    distance_norm = distance_norm.sqrt();
    A2epline_distance = epline_B*matched_B_homo.array();
    distance.col(i) = abs(A2epline_distance.rowwise().sum());
    distance.col(i) = distance.col(i)/distance_norm;

    cout<<distance.col(i)<<endl;

  }
  // select the best case
  MatrixXi thresholed_distance = (distance<threshold).cast<int>();
  MatrixXi score = thresholed_distance.colwise().sum();
  // find the index of with maximum number of matches
  int max_idx = 0;
  int max_value = -1;
  for (int i=0; i<score.cols();i++) {
    if (score(i)>max_value) {
      max_idx = i;
      max_value = score(i);
    }
  }
  MatrixXi selected_matches = thresholed_distance.col(max_idx);
  for (int i=0; i<selected_matches.rows(); i++) {
	if (selected_matches(i)==0) {
      int a2bmatch_idx = matched_A_kp_idx[i];
      a2b_matches[a2bmatch_idx] = -1;
	}

  }
  cout<<thresholed_distance<<endl;
  return 0;
}*/

