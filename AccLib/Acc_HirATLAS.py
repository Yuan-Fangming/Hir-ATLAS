import numpy as np
from VIPRDLpy3 import *
import time
from sklearn.decomposition import PCA 
import pickle as pk
import gc
import random
from threading import Thread
import copy
from sklearn.decomposition import IncrementalPCA 
import pickle as pk

# for C++ acc codes
import ctypes as ct
from numpy.ctypeslib import ndpointer
import time
from scipy.spatial.distance import cdist

# convert the algorithm type string to integer
# algorithm can be:
# 'PW', 'LPG','RANSAC','HOL'
def Alg2Digit(alg_str):
  dummy = __CAccLib(acc_file=None)
  if alg_str=="PW":
    return dummy.ALG_TYPE_PAIRWISE
  if alg_str=="LPG":
    return dummy.ALG_TYPE_PAIRWISE_POSGRAPH
  if alg_str=="RANSAC":
    return dummy.ALG_TYPE_PAIRWISE_RANSAC
  if alg_str=="HOL":
    return dummy.ALG_TYPE_HOLISTIC_FEATURE_MATCH
  if alg_str=="LPG-MATCH":
    return dummy.ALG_TYPE_LPG_MATCH    
  if alg_str=="STARGRAPH":
    return dummy.PAIRWISE_STARGRAPH  
  if alg_str=="RSS":
    return dummy.ALG_TYPE_RSS




class __CAccLib:
  def __init__(self, acc_file='FeatureMatchAcc.so'):
    self.ALG_TYPE_PAIRWISE               = 0 # retrieval by pairwise     
    self.ALG_TYPE_PAIRWISE_POSGRAPH      = 2 # retrieval by pairwise match+LPG
    self.ALG_TYPE_ONLY_PAIRWISE_MATCH    = 3
    self.ALG_TYPE_PAIRWISE_RANSAC        = 4 # retrieval by pairwise match+RANSAC
    self.ALG_TYPE_HOLISTIC_FEATURE_MATCH = 5 # retrieval by holistic feature match
    self.ALG_TYPE_LPG_MATCH              = 6 # LPG match
    self.ALG_TYPE_RSS                    = 7 # Rapid Spatial Scoreof Patch-NetVLAD
    # test experiment
    self.PAIRWISE_STARGRAPH=1000
    if acc_file == None:
      return
    
    acc_lib = ct.CDLL(acc_file)#'.lib/FeatureMatchAcc.so'
      
    self.CAcc_FilterMatch = acc_lib.FilterMatch
    self.CAcc_FilterMatch.restype = ct.c_int
    self.CAcc_FilterMatch.argtypes = [\
                       ndpointer(ct.c_float, ndim=1, flags="C_CONTIGUOUS"),\
                       ct.c_int32 , ct.c_int32,\
                       ndpointer(ct.c_int32, ndim=1, flags="C_CONTIGUOUS"),\
                       ndpointer(ct.c_float, ndim=1, flags="C_CONTIGUOUS")\
                                 ]               
    
    
    
    self.CAcc_PairwiseSimilarityRowPosGraph_MultiThread = acc_lib.C_PairwiseSimilarityRowPosGraph_MultiThread
    self.CAcc_PairwiseSimilarityRowPosGraph_MultiThread.restype=ct.c_int
    self.CAcc_PairwiseSimilarityRowPosGraph_MultiThread.argtypes = [\
                  ndpointer(ct.c_float, ndim=1, flags="C_CONTIGUOUS"), ct.c_int32 , ct.c_int32,\
                  ndpointer(ct.c_int32, ndim=1, flags="C_CONTIGUOUS"), ct.c_int32 ,\
                  ndpointer(ct.c_int32, ndim=1, flags="C_CONTIGUOUS"), \
                  ndpointer(ct.c_int32, ndim=1, flags="C_CONTIGUOUS"),\
                  ct.c_int32 ,\
                  ct.c_int32 , ct.c_int32,\
                  ndpointer(ct.c_int, flags="C_CONTIGUOUS"), ndpointer(ct.c_int, flags="C_CONTIGUOUS"), ct.c_int,\
                  ndpointer(ct.c_float, flags="C_CONTIGUOUS"), ct.c_int, ct.c_int,\
                  ct.c_int,\
                  ct.c_int,\
                  ndpointer(ct.c_float, flags="C_CONTIGUOUS"),\
                  ndpointer(ct.c_int32, flags="C_CONTIGUOUS"),\
                  ndpointer(ct.c_float, flags="C_CONTIGUOUS")\
               ]
    
    self.CAcc_PairwiseSimilarityVersatile_MultiThread = acc_lib.C_PairwiseSimilarityVersatile_MultiThread
    self.CAcc_PairwiseSimilarityVersatile_MultiThread.restype=ct.c_int
    self.CAcc_PairwiseSimilarityVersatile_MultiThread.argtypes = [\
                  ndpointer(ct.c_float, ndim=1, flags="C_CONTIGUOUS"), ct.c_int32 , ct.c_int32,\
                  ndpointer(ct.c_int32, ndim=1, flags="C_CONTIGUOUS"), ct.c_int32 ,\
                  ndpointer(ct.c_int32, ndim=1, flags="C_CONTIGUOUS"), \
                  ndpointer(ct.c_int32, ndim=1, flags="C_CONTIGUOUS"),\
                  ndpointer(ct.c_int32, ndim=1, flags="C_CONTIGUOUS"),\
                  ct.c_int32 ,\
                  ct.c_int32 , ct.c_int32,\
                  ndpointer(ct.c_int32, flags="C_CONTIGUOUS"), ndpointer(ct.c_int32, flags="C_CONTIGUOUS"), ndpointer(ct.c_int32, flags="C_CONTIGUOUS"), ct.c_int,\
                  ndpointer(ct.c_int32, flags="C_CONTIGUOUS"), ct.c_int,\
                  ndpointer(ct.c_float, flags="C_CONTIGUOUS"), ct.c_int, ct.c_int,\
                  ct.c_int,\
                  ct.c_int,\
                  ndpointer(ct.c_float, flags="C_CONTIGUOUS"),\
                  ct.c_int,\
                  ndpointer(ct.c_int32, flags="C_CONTIGUOUS"),\
                  ndpointer(ct.c_float, flags="C_CONTIGUOUS"),\
                  ndpointer(ct.c_float, flags="C_CONTIGUOUS")\
               ]    
    
class CAccFeatureMatch(__CAccLib):
  def __init__(self, acc_file='FeatureMatchAcc.so'):
    super(CAccFeatureMatch, self).__init__(acc_file)
    self.time_analysis = 0
     
     
                      
  def Generate2DGaussian(self,Y,X, padding_y, padding_x, sigma=1.0):
    x_size=X
    y_size=Y
    x, y = np.meshgrid(np.linspace(-10,10,x_size), np.linspace(-10,10,y_size))
    dst = np.sqrt(x*x+y*y)
 
    # Initializing sigma and muu
    #sigma = 1
    muu = 0.000
 
    # Calculating Gaussian array
    gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )  
    gauss2d_y_size = Y
    gauss2d_x_size = X
    gauss_patch_x_size = padding_x
    gauss_patch_y_size = padding_y
    
    gauss_patch =np.zeros((gauss_patch_y_size,gauss_patch_x_size), dtype=np.float32)
    gauss_patch[int(gauss_patch_y_size/2)-int(gauss2d_y_size/2): int(gauss_patch_y_size/2)+ gauss2d_y_size-int(gauss2d_y_size/2),   int(gauss_patch_x_size/2)-int(gauss2d_x_size/2): int(gauss_patch_x_size/2)+ gauss2d_x_size-int(gauss2d_x_size/2)] = gauss

    return gauss_patch 
    
  def PartitionedMatMul(self,A,B,part_size):
    A_size = A.shape[0]
    B_size = B.shape[0]
    loopA = int(A_size/part_size)
    loopB = int(B_size/part_size)
    reminA = A_size%part_size
    reminB = B_size%part_size
    if reminA!=0:
      loopA+=1
    if reminB!=0:
      loopB+=1
    
    #print (C.shape)
    with tf.device('/gpu:0'): 
      C = tf.Variable(tf.zeros((A_size,B_size), dtype=tf.float16))  
      stepA=0
      endA = 0
      for i in range(0, loopA):
        if A_size-stepA<part_size:
          endA += reminA
        else:
          endA += part_size
        mat_A = A[stepA:endA,:]
        stepB=0
        endB=0
        for j in range(0, loopB):
          if B_size-stepB<part_size:
            endB += reminB
            #print B_size-stepB, reminB, endB
          else:
            endB += part_size
          mat_B = B[stepB:endB,:]
        
          W = tf.matmul(mat_A,mat_B, transpose_b = True)
          #print (stepA,endA, stepB,endB)
          C[stepA:endA, stepB:endB] = W
          stepB += part_size 
        stepA+=part_size
        
    return C    

  def Array2Dto1D(self, mat, dtype):
    if mat is None:
      mat_h = 0
      mat_w = 0
      mat_in_row = np.ascontiguousarray(np.zeros(1), dtype=dtype)  
    else:
      mat_h = mat.shape[0]
      mat_w = mat.shape[1]
      mat_in_row = np.reshape(mat, (mat_h*mat_w))
      mat_in_row = np.ascontiguousarray(mat_in_row, dtype=dtype)        
    return mat_in_row, mat_h, mat_w  


  def CAcc_FilterMatchWorpper(self, sm_patch):
    ###############take time 1 unit################
    #sm_patch_h = sm_patch.shape[0]
    #sm_patch_w = sm_patch.shape[1]
    #sm_patch_in_row = np.reshape(sm_patch, (sm_patch_h*sm_patch_w))
    #sm_patch_in_row = np.ascontiguousarray(sm_patch_in_row, dtype=np.float32)
    
    
    sm_patch_in_row, sm_patch_h, sm_patch_w = self.Array2Dto1D(sm_patch, dtype=np.float32)
    
    
    matches = np.zeros((sm_patch_h), dtype=np.int32)
    distance = np.zeros((sm_patch_h), dtype=np.float32)
    ###############################################

        
    ###############take time 2 unit################
    #sm_patch_in_row= np.zeros((1000),dtype=np.float32)
    self.CAcc_FilterMatch(sm_patch_in_row, sm_patch_h, sm_patch_w, matches, distance)
    ###############################################  
    return matches, distance    

  # matches in shape [N] N=match_sm.shape[0]
  # is matches[0] = 10  means match_sm[0,10] is vertical and horizential mutual maximum
  def FilterMatch(self, match_sm):
    sm_max_ver = np.zeros((match_sm.shape[0], match_sm.shape[1]),dtype=np.int8)
    sm_max_hor = np.zeros((match_sm.shape[0], match_sm.shape[1]),dtype=np.int8)
    if match_sm.shape[0] ==0 or match_sm.shape[1] ==0:
       a=0
    else:
      for i in range(0, match_sm.shape[1]):
        max_idx = np.argmax(match_sm[:,i])
        sm_max_ver[max_idx,i] =1 
      for i in range(0, match_sm.shape[0]):
        max_idx = np.argmax(match_sm[i,:])
        sm_max_hor[i,max_idx] =1 
    match_mat = np.logical_and(sm_max_ver, sm_max_hor)
    matches = np.full((match_mat.shape[0]), -1)
    distance = np.zeros((match_mat.shape[0]),np.float32)
    for i in range(match_mat.shape[0]):
      mv = match_mat[i,:]
      if True in mv:
         idx = mv.tolist().index(True)
         matches[i] = idx
         distance[i] = match_sm[i,idx]
    return matches, distance
# A tensor in shape[N_a,25088]  {vector_idx,element_in_vector}
# B tensor in shape[N_b,25088]  {vector_idx,element_in_vector}
# calculate each consin distance between A[:,:] with all vector B[:,:]
# return [N_a, N_b]   [a,b] means the consin distance between A[a,:] and B[b,:] 
  def SuerExpress_cross_cosine_distanceGPU(self, A, B):
    with tf.device('/gpu:0'): 
      #normalize A and B
      square_a = tf.square(A)
      square_b = tf.square(B)
      sum_square_a = tf.reduce_sum(square_a, axis=1)  # shape [batch,N_a]
      sum_square_a = tf.add(sum_square_a, 0.00000000001)   # add 0.000001 incase of zero vector 
      sum_square_b = tf.reduce_sum(square_b, axis=1)  # shape [batch,N_b]
      sum_square_b = tf.add(sum_square_b, 0.00000000001)

      sum_square_a = tf.reshape(sum_square_a,[sum_square_a.shape[0],1])
      sum_square_b = tf.reshape(sum_square_b,[sum_square_b.shape[0],1])
      #print sum_square_a.shape
      #A = PartitionNormDesc(A, 1280)
      #B = PartitionNormDesc(B, 1280)
      #print ("###############Test2##################",A.shape)
      C = tf.matmul(A,B,transpose_b = True)  #self.PartitionedMatMul(A,B, 2048) #640
      #print ("C has shape ", C.shape)
 
      Norm = tf.matmul(tf.sqrt(sum_square_a), tf.sqrt(sum_square_b), transpose_b = True)
      result = tf.divide(C,Norm)
      return result.numpy() #tf.nn.softmax(C, axis=1)
      
  # these two function calculate the cosine similarity between arrays or between vector and arrays
  # Use GPU to accelerate these functions
  # all the input are in type numpy arrays in cpu memory space
  # A/B: in shape e.g [N_x, 1024]
  # return: numpy array in shape[N_a, N_b] in cpu memory space
  def cross_cosine_distance_ArrayVsArray(self, A, B):
    tf_A = tf.Variable(A, dtype=tf.float16)
    tf_B = tf.Variable(B, dtype=tf.float16)
    return self.SuerExpress_cross_cosine_distanceGPU(tf_A, tf_B)
    
  def cross_cosine_distance_ArrayVsArray_batch(self, A, B, batch=16384):
    A_size = A.shape[0]
    B_size = B.shape[0]
    
    step= int(int(B_size)/int(batch))
    reminder = int(int(B_size)%int(batch))
    simMat = np.zeros((A_size, B_size))
    tf_A = tf.Variable(A, trainable=False, dtype=tf.float16)
    last_idx = 0
    for i in range(0,step):  
      tf_B = tf.Variable(B[i*batch:(i+1)*batch,:], trainable=False, dtype=tf.float16, synchronization=tf.VariableSynchronization.NONE)
      simMat[:,i*batch:(i+1)*batch] = self.SuerExpress_cross_cosine_distanceGPU(tf_A, tf_B) 
      last_idx = (i+1)*batch
    if reminder != 0:
      tf_B = tf.Variable(B[last_idx:,:], trainable=False, dtype=tf.float16)
      simMat[:,last_idx:]  = self.SuerExpress_cross_cosine_distanceGPU(tf_A, tf_B)  
    return simMat


  def cross_cosine_distance_ArrayVsArray_GPU_preload_B(self, A, B):
    A_size = A.shape[0]
    B_size = B.shape[0]
    
    simMat = np.zeros((A_size, B_size))
    tf_A = tf.Variable(A, trainable=False, dtype=tf.float16, synchronization=tf.VariableSynchronization.NONE)   
    simMat[:,:]  = self.SuerExpress_cross_cosine_distanceGPU(tf_A, B)  
    return simMat
    
  def cross_cosine_distance_VectVsArray_gpu(self, v, B):
    tf_v = tf.Variable(v.reshape((1,v.shape[0])), dtype=tf.float16)
    tf_B = tf.Variable(B, dtype=tf.float16)
    return self.SuerExpress_cross_cosine_distanceGPU(tf_v, tf_B)

  def cross_cosine_distance_VectVsArray_cpu(self, v, B):
    
    w = v.reshape(1,v.shape[0])
    #print ("v shape is here", v.shape, B.shape)
    value = cdist(w,B, metric='cosine')
    #print(value[0,:].shape)
    return -(value[0,:])

  def euclidean_distance_VectVsArray_cpu(self, v, B):
    w = v.reshape(1,v.shape[0])
    #print ("v shape is here", v.shape, B.shape)
    value = cdist(w,B, metric='euclidean')
    #print(value[0,:].shape)
    return -(value[0,:])

  def cross_cosine_distance_ArrayVsArray_cpu(self, v, B):

    #print ("v shape is here", v.shape, B.shape)
    value = cdist(v,B, metric='cosine')
    #print(value[0,:].shape)
    return -(value)

  def euclidean_distance_ArrayVsArray_cpu(self, v, B):
    #print ("v shape is here", v.shape, B.shape)
    value = cdist(v,B, metric='euclidean')
    #print(value[0,:].shape)
    return -(value)
  # pick out the matched kp and sort them according to X coordinate
  # return a dict. with given X coordinate as key. Return the location index in loc_mat_a
  def SortKpXLoc(self, matches, enable_matched=False, loc_mat_a=None):
    kp_num = loc_mat_a.shape[0]
    X = np.zeros((kp_num,2),dtype=np.int32)
    if enable_matched == True:
      bbbb=bbbb
    else:
      for i in range(loc_mat_a.shape[0]):
        X[i,0] = loc_mat_a[i,1]
        X[i,1] = i      
    X = X[X[:,0].argsort()]
    return X
  # return a dict. with given X coordinate as key. Return the location index in loc_mat_a
  import time
  def SortKpXLocGetRange(self, matches, enable_matched=False, loc_mat_a=None, x_from=0, x_end=0):

    #start_time = time.time() ######
    kp_num = loc_mat_a.shape[0]
    indic = np.arange(0,kp_num,dtype=np.int32)
    X = np.zeros((2,kp_num),dtype=np.int32 )
    X[1,:] = indic
    X = np.transpose(X)
    
    if enable_matched == True:
      bbbb=bbbb
    else:
      X[:,0] = loc_mat_a[:,1] 
    #end_time = time.time() ######    
    #print (end_time-start_time)#####
    
    
    #start_time = time.time() ######
    X = X[X[:,0].argsort()]
    #end_time = time.time() ######    
    #print (end_time-start_time)#####
    X_in_range = []
    for i in range(0,X.shape[0]):
      key_value = X[i,0]
      if key_value<=x_end and key_value>=x_from:
        #X_in_range[key_value]=X[key_value]
        X_in_range.append(X[i,:])
    return np.array(X_in_range)
  # find the closet matched neighbor of target_kp with given window size
  # target_kp:  target keypoint in shape [y0,x0]
  # win_x_radius, win_y_radius: scalar value of window radius
  # loc_mat_a:  all keypoint location in shape [N,2]  loc_mat_a[n,0]=yn. loc_mat_a[n,1]=xn
  # matches: selectable variable
  # enable_matched:  if use matches
  # return the list with each represents the closet neighbor keypont indise in loc_mat_a and the delta pos ([y-tgt_kp_y, x-tgt_kp_y]) to the target kp
  # return in shape [k] where k means k neighbor. each element is np.array([idx, delta_y, delta_x])
  def FindClosetNeighbor(self, target_kp, win_x_radius, win_y_radius, loc_mat_a, matches, enable_matched=False):
    x_start = target_kp[1]-win_x_radius
    x_end   = target_kp[1]+win_x_radius
    y_start = target_kp[0]-win_y_radius
    y_end   = target_kp[0]+win_y_radius
    X_in_range = self.SortKpXLocGetRange(matches, enable_matched, loc_mat_a, x_start, x_end)
    kp_idx = X_in_range[:,1]#  list(X_in_range.values()) 
    # select the kp which y in the range
    closet_kp_idx = []
    for i in range(0,kp_idx.shape[0]):
      idx = kp_idx[i]
      y =  loc_mat_a[idx,0]
      x = loc_mat_a[idx,1]
      if y<= y_end and y>=y_start:
        closet_kp_idx.append(np.array([int(idx), int(y-target_kp[0]),  int(x-target_kp[1])]))
    if len(closet_kp_idx)==0:
      closet_kp_idx.append(np.array([-1,-1,-1], dtype=np.int32))
    return closet_kp_idx

  def FindEveryKPClosetNeighbor(self, win_x_radius, win_y_radius, loc_mat_a, matches, enable_matched=False):
    frame_neighbor_list = []
    for kp_idx in range(0,loc_mat_a.shape[0]): 
      target_kp = loc_mat_a[kp_idx,:]
      kp_neigbor_list = self.FindClosetNeighbor(target_kp, win_x_radius, win_y_radius, loc_mat_a, matches, enable_matched=False)
      
      frame_neighbor_list.append(kp_neigbor_list)
    return frame_neighbor_list
    
  def FindEveryKPClosetNeighborInMat(self, win_x_radius, win_y_radius, loc_mat_a, matches, enable_matched=False):
    frame_neighbor_list = []
    neighbor_count_list = []
    for kp_idx in range(0,loc_mat_a.shape[0]): 
      target_kp = loc_mat_a[kp_idx,:]
      kp_neigbor_list = self.FindClosetNeighbor(target_kp, win_x_radius, win_y_radius, loc_mat_a, matches, enable_matched=False)
      kp_neigbor_array = np.array(kp_neigbor_list)
      frame_neighbor_list.append(kp_neigbor_array.astype(np.int32))
      neighbor_count_list.append(int(kp_neigbor_array.shape[0]))
    return np.concatenate(frame_neighbor_list, axis=0), np.array(neighbor_count_list, dtype=np.int32)
  def ConvertAframeNeighborMat2CAccFormat(self, a_num, a_frame_neighbor_mat):
    kp_neighbor_num = np.zeros((a_num), dtype=np.int32)
    for i in range(0,len(a_frame_neighbor_mat)):
      neighbor_list = a_frame_neighbor_mat[i]  
      kp_neighbor_num[i] = len(neighbor_list)
      a_frame_neighbor_mat[i] = np.array(neighbor_list, dtype=np.int32)
    a_frame_neighbor_mat_in_row = np.concatenate(a_frame_neighbor_mat, axis=0)
    
    a_frame_neighbor_mat_in_row,a,b=self.Array2Dto1D(a_frame_neighbor_mat_in_row, dtype=np.int32)   
    
    return  a_frame_neighbor_mat_in_row, kp_neighbor_num

    
    
  def PairwiseSimilarityPosGraph(self, a_num, b_num, matches, distances, a_pos, b_pos, dim_y, dim_x, a_frame_neighbor_mat, gaussian_patch):
    #print a_num, b_num
    sim_sum =0.0
    ##gaussian_center_x = gaussian_patch.shape[1]/2
    ##gaussian_center_y = gaussian_patch.shape[0]/2
    gaussian_center = np.array([gaussian_patch.shape[0]/2, gaussian_patch.shape[1]/2])
    
    for i in range(0, matches.shape[0]):
      if matches[i] >=0: # if there is a matche
        kp_a_loc = a_pos[i,:]
        kp_b_loc = b_pos[matches[i],:] 
        # coordinate of coorespondent in B
        #kp_b_loc_y = kp_b_loc[0]
        #kp_b_loc_x = kp_b_loc[1]
       
        
        # go through each neighbor of this kp to calculate its weighting score
        graph_pos_score = 0
        matched_neighbor_count = 0
        neighbor_list = a_frame_neighbor_mat[i]
        for j in range(len(neighbor_list)):
          A_neighbor_attr  = neighbor_list[j]
          neighbor_idx = int(A_neighbor_attr[0])

          # check if the neighbor has matched
          correspondec_idx = matches[neighbor_idx]
          if correspondec_idx>=0:  
            ##delta_y_a = A_neighbor_attr[1]
            ##delta_x_a = A_neighbor_attr[2] 
             
            delta_a = A_neighbor_attr[1:3]
            #print ("Q")
            # get x y of correspondence 
            ##y_b = b_pos[correspondec_idx, 0]
            ##x_b = b_pos[correspondec_idx, 1] 
            b = b_pos[correspondec_idx,:]
            # calculate pos delta of coorespondence
            ##delta_x_b = x_b - kp_b_loc_x
            ##delta_y_b = y_b - kp_b_loc_y
            delta_b = b-kp_b_loc
            
            # difference of two delta
            ##diff_x = delta_x_b - delta_x_a
            ##diff_y = delta_y_b - delta_y_a
            diff = delta_b-delta_a
            # translate the delta to gaussian center
            ##diff_x += gaussian_center_x
            ##diff_y += gaussian_center_y
            diff += gaussian_center
            #print ("W")
            # look up gaussian score
            ##gaussian_score = gaussian_patch[int(diff_y), int(diff_x)]
            gaussian_score = gaussian_patch[int(diff[0]), int(diff[1])]
            #print ("A")
            graph_pos_score += gaussian_score  #*distances[neighbor_idx]
            matched_neighbor_count+=1
        # normalize graph pos score
        graph_pos_score/=(matched_neighbor_count+0.00000001)
        sim_sum += distances[i]*graph_pos_score
    #print ("P")
    return sim_sum/(np.sqrt(a_num*b_num))   

   
    
  def CAcc_PairwiseSimilarityRowPosGraph__MultiThread_Warpper(self,sm_desc_row_patch, a_kp_num, a_pos_patch, b_pos_mat ,b_delimiter, dim_y, dim_x, a_neighbor_patch, gaussian_patch, \
                                                              a_pos_in_row, a_neighbor_patch_mat_contignous, a_kp_neighbor_num_mat_contignous,\
                                                              para_thread_num=1, alg_type=2):
    sm_desc_row_patch_contignous, sm_desc_row_patch_H, sm_desc_row_patch_W = self.Array2Dto1D(sm_desc_row_patch, dtype=np.float32) 

    #a_neighbor_patch_mat_contignous, a_kp_neighbor_num_mat_contignous = self.ConvertAframeNeighborMat2CAccFormat(a_kp_num, a_neighbor_patch)
    
    gauss_in_row, gauss_h, gauss_w = self.Array2Dto1D(gaussian_patch, dtype=np.float32)
       
    #a_pos_in_row, a,b = self.Array2Dto1D(a_pos_patch, dtype=np.int32)
    b_pos_in_row, a,b = self.Array2Dto1D(b_pos_mat, dtype=np.int32)
    
    o_sm_row = np.zeros((b_delimiter.shape[0]),dtype=np.float32,order='C')
    o_sm_row = np.ascontiguousarray(o_sm_row, dtype=np.float32)
    
    b_delimiter_contignous = np.ascontiguousarray(b_delimiter, dtype=np.int32)
    
    o_match_a2b = np.zeros((a_kp_num*b_delimiter.shape[0]),dtype=np.int32,order='C')
    o_distance_a2b = np.zeros((a_kp_num*b_delimiter.shape[0]),dtype=np.float32,order='C')
    # CAcc function
    a = self.CAcc_PairwiseSimilarityRowPosGraph_MultiThread(sm_desc_row_patch_contignous, sm_desc_row_patch_H, sm_desc_row_patch_W,\
                                                            a_pos_in_row,  a_kp_num,\
                                                            b_pos_in_row,\
                                                            b_delimiter_contignous,\
                                                            b_delimiter.shape[0],\
                                                            dim_y,dim_x,\
                                                            a_neighbor_patch_mat_contignous, a_kp_neighbor_num_mat_contignous, 3,\
                                                            gauss_in_row, gauss_h, gauss_w,\
                                                            para_thread_num,\
                                                            alg_type,\
                                                            o_sm_row,\
                                                            o_match_a2b,\
                                                            o_distance_a2b
                                                              )
    
    return o_sm_row
  
  # Can choice to only perform pairwise matching 
  def CAcc_PairwiseSimilarityRowPosGraph__MultiThread_WarpperEx(self,sm_desc_row_patch, a_kp_num, a_pos_patch, b_pos_mat ,b_delimiter, dim_y, dim_x, a_neighbor_patch, gaussian_patch, \
                                                              a_pos_in_row, a_neighbor_patch_mat_contignous, a_kp_neighbor_num_mat_contignous,\
                                                              para_thread_num=1, alg_type=2):
    sm_desc_row_patch_contignous, sm_desc_row_patch_H, sm_desc_row_patch_W = self.Array2Dto1D(sm_desc_row_patch, dtype=np.float32) 

    #a_neighbor_patch_mat_contignous, a_kp_neighbor_num_mat_contignous = self.ConvertAframeNeighborMat2CAccFormat(a_kp_num, a_neighbor_patch)
    gauss_in_row, gauss_h, gauss_w = self.Array2Dto1D(gaussian_patch, dtype=np.float32)
       
    a_pos_in_row, a,b = self.Array2Dto1D(a_pos_patch, dtype=np.int32)
    b_pos_in_row, a,b = self.Array2Dto1D(b_pos_mat, dtype=np.int32)
    
    o_sm_row = np.zeros((b_delimiter.shape[0]),dtype=np.float32,order='C')
    o_sm_row = np.ascontiguousarray(o_sm_row, dtype=np.float32)
    
    b_delimiter_contignous = np.ascontiguousarray(b_delimiter, dtype=np.int32)
    
    o_match_a2b = np.zeros((a_kp_num*b_delimiter.shape[0]),dtype=np.int32,order='C')
    o_distance_a2b = np.zeros((a_kp_num*b_delimiter.shape[0]),dtype=np.float32,order='C')
    
    
    
    if a_neighbor_patch_mat_contignous is None:
      a_neighbor_patch_mat_contignous = np.array([], dtype=np.int32)
    if a_kp_neighbor_num_mat_contignous is None:
      a_kp_neighbor_num_mat_contignous  = np.array([], dtype=np.int32)
    if a_pos_in_row is None:
      a_pos_in_row  = np.array([], dtype=np.int32)
    # CAcc function
    a = self.CAcc_PairwiseSimilarityRowPosGraph_MultiThread(sm_desc_row_patch_contignous, sm_desc_row_patch_H, sm_desc_row_patch_W,\
                                                            a_pos_in_row,  a_kp_num,\
                                                            b_pos_in_row,\
                                                            b_delimiter_contignous,\
                                                            b_delimiter.shape[0],\
                                                            dim_y,dim_x,\
                                                            a_neighbor_patch_mat_contignous, a_kp_neighbor_num_mat_contignous, 3,\
                                                            gauss_in_row, gauss_h, gauss_w,\
                                                            para_thread_num,\
                                                            alg_type,\
                                                            o_sm_row,\
                                                            o_match_a2b,\
                                                            o_distance_a2b
                                                              )
    
    return o_sm_row, o_match_a2b, o_distance_a2b
    
    
  def CAcc_PairwiseSimilarityVersatile__MultiThread_Warpper(self,sm_desc_row_patch, sm_desc_row_patch_h, sm_desc_row_patch_w,\
                                                              query_kp_num, query_pos_patch, \
                                                              DBtopN_pos_mat ,\
                                                              DBtopN_delimiter, \
                                                              DB_frame_kp_start_idx,\
                                                              dim_y, dim_x,\
                                                              DBtopN_KP_neighbor, DBtopN_neighbor_in_frame_delimiter, DBtopN_frame_neighbor_start_idx, \
                                                              valid_db_frame_idx,\
                                                              gaussian_patch, gauss_h, gauss_w,\
                                                              para_thread_num=1, alg_type=0, dbg=0):
                                                              
                                                              
                                                              
    #sm_desc_row_patch_contignous, sm_desc_row_patch_h, sm_desc_row_patch_w = self.Array2Dto1D(sm_desc_row_patch, dtype=np.float32) 
    sm_desc_row_patch_contignous = sm_desc_row_patch #, sm_desc_row_patch_H, sm_desc_row_patch_W = self.Array2Dto1D(sm_desc_row_patch, dtype=np.float32)
    #print(sm_desc_row_patch_h, sm_desc_row_patch_w)
    #a_neighbor_patch_mat_contignous, a_kp_neighbor_num_mat_contignous = self.ConvertAframeNeighborMat2CAccFormat(a_kp_num, a_neighbor_patch)
    
    
    #DB_frame_kp_start_idx = np.ascontiguousarray(DB_frame_kp_start_idx, dtype=np.int32)
    
    valid_b_frame_num  = valid_db_frame_idx.shape[0]
    valid_db_frame_idx = np.ascontiguousarray(valid_db_frame_idx, dtype=np.int32)
       
    query_pos_in_row, a,b = self.Array2Dto1D(query_pos_patch, dtype=np.int32)
    
    #DBtopN_pos_in_row, a,b = self.Array2Dto1D(DBtopN_pos_mat, dtype=np.int32)   ##
    DBtopN_pos_in_row = DBtopN_pos_mat
    
    #DBtopN_delimiter_contignous = np.ascontiguousarray(DBtopN_delimiter, dtype=np.int32) ##
    DBtopN_delimiter_contignous = DBtopN_delimiter ##
    
    #DBtopN_KP_neighbor_in_row,a,b = self.Array2Dto1D(DBtopN_KP_neighbor, dtype=np.int32) ##
    DBtopN_KP_neighbor_in_row = DBtopN_KP_neighbor
    
    #DBtopN_neighbor_in_frame_delimiter_contignous = np.ascontiguousarray(DBtopN_neighbor_in_frame_delimiter, dtype=np.int32) ##
    DBtopN_neighbor_in_frame_delimiter_contignous = DBtopN_neighbor_in_frame_delimiter
    
    #DBtopN_frame_neighbor_start_idx_contignous = np.ascontiguousarray(DBtopN_frame_neighbor_start_idx, dtype=np.int32) ##
    DBtopN_frame_neighbor_start_idx_contignous = DBtopN_frame_neighbor_start_idx
    
    #gauss_in_row, gauss_h, gauss_w = self.Array2Dto1D(gaussian_patch, dtype=np.float32)
    gauss_in_row = gaussian_patch
    
    o_sm_row = np.zeros((valid_b_frame_num),dtype=np.float32,order='C')
    o_sm_row = np.ascontiguousarray(o_sm_row, dtype=np.float32)
    
    o_match_query2DBtopN = np.zeros((query_kp_num*valid_b_frame_num),dtype=np.int32,order='C')
    o_distance_query2DBtopN = np.zeros((query_kp_num*valid_b_frame_num),dtype=np.float32,order='C')
    o_distance_aux_query2DBtopN = np.zeros((query_kp_num*valid_b_frame_num),dtype=np.float32,order='C')
    
    start_time = time.time()
    # CAcc function
    a = self.CAcc_PairwiseSimilarityVersatile_MultiThread(sm_desc_row_patch_contignous, sm_desc_row_patch_h, sm_desc_row_patch_w,\
                                                            query_pos_in_row,  query_kp_num,\
                                                            DBtopN_pos_in_row,\
                                                            DBtopN_delimiter_contignous,\
                                                            DB_frame_kp_start_idx,\
                                                            DBtopN_delimiter.shape[0],\
                                                            dim_y,dim_x,\
                                                            DBtopN_KP_neighbor_in_row, DBtopN_neighbor_in_frame_delimiter_contignous, DBtopN_frame_neighbor_start_idx_contignous, 3,\
                                                            valid_db_frame_idx, valid_b_frame_num,\
                                                            gauss_in_row, gauss_h, gauss_w,\
                                                            para_thread_num,\
                                                            alg_type,\
                                                            o_sm_row,\
                                                            dbg,\
                                                            o_match_query2DBtopN,\
                                                            o_distance_query2DBtopN,\
                                                            o_distance_aux_query2DBtopN)
    end_time = time.time()
    #self.time_analysis += end_time-start_time
    return o_sm_row, o_match_query2DBtopN, o_distance_query2DBtopN, o_distance_aux_query2DBtopN
       
  def CalcuPairwiseSimilarityGPUSuperExpressPosGraph_CMultiThread(self, desc_mat_a, desc_mat_b, a_delimiter, b_delimiter, desc_pos_mat_a, desc_pos_mat_b, dim_y, dim_x, neighbor_list_a, gaussian_patch, desc_a_gpu=None, desc_a_gpu_preload=False):
    a_size = desc_mat_a.shape[0]
    b_size = desc_mat_b.shape[0]
    #print (a_size, b_size, a_delimiter.shape[0], b_delimiter.shape[0])
    #desc_mat_a = np.float32(desc_mat_a)
    #desc_mat_b = np.float32(desc_mat_b)
    if desc_a_gpu_preload==False:
      desc_a_gpu = tf.Variable(desc_mat_a, dtype=tf.float16)
    desc_b_gpu = tf.Variable(desc_mat_b, dtype=tf.float16)
    sm = np.zeros((a_delimiter.shape[0], b_delimiter.shape[0]),dtype="float32")
    sm_desc = self.SuerExpress_cross_cosine_distanceGPU(desc_a_gpu, desc_b_gpu)
    A_step=0

    for i in range(0, a_delimiter.shape[0]): # for each frame in A
      #print("Processing Row %d"%(i))
      B_step=0
      a_offset = a_delimiter[i]
      
      A_step_from = A_step
      A_step_to = A_step+a_offset
      
      a_neighbor_patch = neighbor_list_a[A_step_from:A_step_to]
      a_pos_patch = desc_pos_mat_a[A_step_from:A_step_to, :]
      a_neighbor_patch_mat_contignous, a_kp_neighbor_num_mat_contignous = self.ConvertAframeNeighborMat2CAccFormat(a_offset, a_neighbor_patch)
      a_pos_in_row, a,b = self.Array2Dto1D(a_pos_patch, dtype=np.int32)
      ###
      sm_desc_row_patch = sm_desc[A_step_from:A_step_to, :]
      sm[i,:] = self.CAcc_PairwiseSimilarityRowPosGraph__MultiThread_Warpper(sm_desc_row_patch, a_offset, a_pos_patch, desc_pos_mat_b, b_delimiter, dim_y, dim_x, a_neighbor_patch, gaussian_patch, \
                                                                             a_pos_in_row, a_neighbor_patch_mat_contignous, a_kp_neighbor_num_mat_contignous,\
                                                                             8)
       
      A_step+=a_offset
    return sm    



  def CAcc_CalcuPairwiseSimilarityGPUSuperExpress_SaveMatches(self, desc_mat_a, desc_mat_b, a_delimiter, b_delimiter, desc_pos_mat_a, desc_pos_mat_b, dim_y, dim_x, neighbor_list_a, gaussian_patch):
    a_size = desc_mat_a.shape[0]
    b_size = desc_mat_b.shape[0]
    #print (a_size, b_size, a_delimiter.shape[0], b_delimiter.shape[0])
    desc_mat_a = np.float32(desc_mat_a)
    desc_mat_b = np.float32(desc_mat_b)
    desc_a_gpu = tf.Variable(desc_mat_a)
    desc_b_gpu = tf.Variable(desc_mat_b)
    sm_desc = self.SuerExpress_cross_cosine_distanceGPU(desc_mat_a, desc_mat_b)
    A_step=0
    frame_match_info_colum_list = []
    for i in range(0, a_delimiter.shape[0]): # for each frame in A
      print("Processing Row %d"%(i))
      B_step=0
      a_offset = a_delimiter[i]
      frame_match_info_row_list = []
      for j in range(0,b_delimiter.shape[0]): # for each frame in B
        b_offset = b_delimiter[j]
        A_step_from = A_step
        A_step_to = A_step+a_offset
        B_step_from = B_step
        B_step_to = B_step+b_offset
        sm_patch = sm_desc[A_step_from:A_step_to, B_step_from:B_step_to]
        
        matches, distance = self.CAcc_FilterMatchWorpper(sm_patch)
        
        frame_match_info = np.array([matches, distance]) #construct frame match info
        frame_match_info_row_list.append(frame_match_info)
        B_step+=b_offset
        
      frame_match_info_row_obj_array = np.array(frame_match_info_row_list, dtype=np.object)
      frame_match_info_colum_list.append(frame_match_info_row_obj_array)
        
      A_step+=a_offset

    return frame_match_info_colum_list    






   
  def CalcuPairwiseSimilarityGPUSuperExpress_SaveMatches(self, desc_mat_a, desc_mat_b, a_delimiter, b_delimiter, desc_pos_mat_a, desc_pos_mat_b, dim_y, dim_x, neighbor_list_a, gaussian_patch):
    a_size = desc_mat_a.shape[0]
    b_size = desc_mat_b.shape[0]
    #print (a_size, b_size, a_delimiter.shape[0], b_delimiter.shape[0])
    desc_mat_a = np.float32(desc_mat_a)
    desc_mat_b = np.float32(desc_mat_b)
    desc_a_gpu = tf.Variable(desc_mat_a)
    desc_b_gpu = tf.Variable(desc_mat_b)
    sm_desc = self.SuerExpress_cross_cosine_distanceGPU(desc_mat_a, desc_mat_b)
    A_step=0
    frame_match_info_colum_list = []
    for i in range(0, a_delimiter.shape[0]): # for each frame in A
      print("Processing Row %d"%(i))
      B_step=0
      a_offset = a_delimiter[i]
      frame_match_info_row_list = []
      for j in range(0,b_delimiter.shape[0]): # for each frame in B
        b_offset = b_delimiter[j]
        A_step_from = A_step
        A_step_to = A_step+a_offset
        B_step_from = B_step
        B_step_to = B_step+b_offset
        sm_patch = sm_desc[A_step_from:A_step_to, B_step_from:B_step_to]
        matches, distance = self.FilterMatch(sm_patch)
      
        frame_match_info = np.array([matches, distance]) #construct frame match info
        frame_match_info_row_list.append(frame_match_info)
        B_step+=b_offset
        
      frame_match_info_row_obj_array = np.array(frame_match_info_row_list, dtype=np.object)
      frame_match_info_colum_list.append(frame_match_info_row_obj_array)
        
      A_step+=a_offset

    return frame_match_info_colum_list    



  # match two set of local feature from files and save the match file 
  # kp_file_dir. the dir where the kp file are and where the match files going to save
  
  #return similarity matrin in shape [N_a,N_b]
  def GenerateBlockMatchFiles(self,a,b):
    return a[0:-4]+'-Match-'+b[0:-4]
  def Match2BatchFratureFilesPosGraph_SaveMatch(self, kp_file_dir, kp_feature_files_a, kp_feature_files_b, kp_loc_files_a, kp_loc_files_b, kp_score_files_a, kp_score_files_b, fm_dim_y, fm_dim_x, kp_neighbor_x_radius=0, kp_neighbor_y_radius=0, gaussian_patch=None):
  
    num_files_a = len(kp_feature_files_a)
    num_files_b = len(kp_feature_files_b)
    
    Afrom = 0
    sm_list = []
    for i in range(0, num_files_a):
      A       = np.load(kp_file_dir+'/'+kp_feature_files_a[i], allow_pickle=True, encoding="latin1")
      #score_A = np.load(kp_score_files_a[i], allow_pickle=True, encoding="latin1")
      loc_A   = np.load(kp_file_dir+'/'+kp_loc_files_a[i], allow_pickle=True, encoding="latin1")
      
      Alen = A.shape[0]
      # reconcatenate A
      A_desc_num = np.zeros((Alen),dtype=np.int32)
      
      for ii in range(0, Alen):
        frame_desc = A[ii]
        frame_desc_pos = loc_A[ii]
        frame_desc_num = frame_desc.shape[0]
        
        # find the closet neighbor of each desc in frame
        frame_kp_closet_neighbor = self.FindEveryKPClosetNeighbor(kp_neighbor_x_radius, kp_neighbor_y_radius, frame_desc_pos, [], enable_matched=False)
        
        A_desc_num[ii] = frame_desc_num
        if ii ==0:
          A_desc = frame_desc
          A_desc_pos = frame_desc_pos
          A_kp_neighbor = frame_kp_closet_neighbor
        else:
          A_desc = np.concatenate((A_desc,frame_desc),axis = 0)
          A_desc_pos = np.concatenate((A_desc_pos,frame_desc_pos),axis = 0)    
          A_kp_neighbor += frame_kp_closet_neighbor
      ####
      Bfrom = 0
      sm_row_list=[]
      for j in range(0, num_files_b):
        B       =  np.load(kp_file_dir+'/'+kp_feature_files_b[j], allow_pickle=True, encoding="latin1")
        #score_B =  np.load(kp_score_files_b[j], allow_pickle=True, encoding="latin1")
        loc_B   =  np.load(kp_file_dir+'/'+kp_loc_files_b[j], allow_pickle=True, encoding="latin1")
        Blen = B.shape[0]
        # reconcatenate B
        B_desc_num = np.zeros((Blen),dtype=np.int32)
        for jj in range(0, Blen):
          frame_desc = B[jj]
          frame_desc_pos = loc_B[jj]
          frame_desc_num = frame_desc.shape[0]
          B_desc_num[jj] = frame_desc_num
          if jj ==0:
            B_desc = frame_desc
            B_desc_pos = frame_desc_pos
          else:
            B_desc = np.concatenate((B_desc,frame_desc),axis = 0)
            B_desc_pos = np.concatenate((B_desc_pos,frame_desc_pos),axis = 0)
        #tmp = CalcuPairwiseSimilarityGPU(A,B)#CalcuPairwiseSimilarityMP(A,B)
        #print (A_desc.shape, B_desc.shape)
        #A_desc = A_desc[0:640,:]
        #B_desc = B_desc[0:640,:]
        print ("Processing grid(%d,%d)"%(i,j))
        block_matches = self.CAcc_CalcuPairwiseSimilarityGPUSuperExpress_SaveMatches(A_desc,B_desc, A_desc_num, B_desc_num, A_desc_pos, B_desc_pos, fm_dim_y, fm_dim_x, A_kp_neighbor, gaussian_patch)
        # saving the block cross match files
        match_file_name = kp_file_dir+'/'+self.GenerateBlockMatchFiles(kp_feature_files_a[i], kp_feature_files_b[j])
        f = open(match_file_name,"wb")
        pk.dump(block_matches, f, protocol=2)
        f.close()
        Bfrom += Blen
        del B
        del Blen
      Afrom += Alen
      del A
      del Alen    
    return 0      






  # match two set of local feature and generate similarity matrix [N_a, N_b] 
  # kp_f_mat_a/kp_loc_mat_a/kp_score_mat_a: in shape [N_a].  
  #    kp_f_mat_a[i] in shape [N,feature_dim]
  #    kp_loc_mat_a[i] in shape [N,2]
  #    kp_score_mat_a[i] in shape [N]
  #return similarity matrin in shape [N_a,N_b]
  
  def Match2BatchFraturePosGraph(self, kp_f_mat_a, kp_f_mat_b, kp_loc_mat_a, kp_loc_mat_b, kp_score_mat_a, kp_score_mat_b, fm_dim_y, fm_dim_x, batch_size=4, kp_neighbor_x_radius=0, kp_neighbor_y_radius=0, gaussian_patch=None):
    a_length = kp_f_mat_a.shape[0]
    b_length = kp_f_mat_b.shape[0]    
    a_batch_N = int(a_length/batch_size)
    b_batch_N = int(b_length/batch_size) 
    
    a_batch_reminder = int(a_length%batch_size) 
    b_batch_reminder = int(b_length%batch_size) 
    
    Afrom = 0
    sm_list = []
    for i in range(0, a_batch_N+1):
      a_start = i*batch_size
      if  i==a_batch_N:
        if a_batch_reminder == 0:
          break
        else:       
          A = kp_f_mat_a[a_start:a_start+a_batch_reminder]
          #score_A = kp_score_mat_a[a_start:a_start+a_batch_reminder]
          loc_A =  kp_loc_mat_a[a_start:a_start+a_batch_reminder]        
      else: 
        A = kp_f_mat_a[a_start:a_start+batch_size]
        #score_A = kp_score_mat_a[a_start:a_start+batch_size]
        loc_A =  kp_loc_mat_a[a_start:a_start+batch_size]
      
      Alen = A.shape[0]
      # reconcatenate A
      A_desc_num = np.zeros((Alen),dtype=np.int32)
      
      for ii in range(0, Alen):
        frame_desc = A[ii]
        frame_desc_pos = loc_A[ii]
        frame_desc_num = frame_desc.shape[0]
        
        # find the closet neighbor of each desc in frame
        frame_kp_closet_neighbor = self.FindEveryKPClosetNeighbor(kp_neighbor_x_radius, kp_neighbor_y_radius, frame_desc_pos, [], enable_matched=False)
        
        A_desc_num[ii] = frame_desc_num
        if ii ==0:
          A_desc = frame_desc
          A_desc_pos = frame_desc_pos
          A_kp_neighbor = frame_kp_closet_neighbor
        else:
          A_desc = np.concatenate((A_desc,frame_desc),axis = 0)
          A_desc_pos = np.concatenate((A_desc_pos,frame_desc_pos),axis = 0)    
          A_kp_neighbor += frame_kp_closet_neighbor
      ####
      Bfrom = 0
      sm_row_list=[]
      for j in range(0, b_batch_N+1):
        b_start = j*batch_size
        if j==b_batch_N:
          if b_batch_reminder == 0:
            break
          else:
            B = kp_f_mat_b[b_start:b_start+b_batch_reminder]
            score_B = kp_score_mat_b[b_start:b_start+b_batch_reminder]
            loc_B =  kp_loc_mat_b[b_start:b_start+b_batch_reminder]
        else:                    

          B = kp_f_mat_b[b_start:b_start+batch_size]
          score_B = kp_score_mat_b[b_start:b_start+batch_size]
          loc_B =  kp_loc_mat_b[b_start:b_start+batch_size]
        Blen = B.shape[0]
        if b_batch_reminder == 0 and i==b_batch_N:
          break
        # reconcatenate B
        B_desc_num = np.zeros((Blen),dtype=np.int32)
        for jj in range(0, Blen):
          frame_desc = B[jj]
          frame_desc_pos = loc_B[jj]
          frame_desc_num = frame_desc.shape[0]
          B_desc_num[jj] = frame_desc_num
          if jj ==0:
            B_desc = frame_desc
            B_desc_pos = frame_desc_pos
          else:
            B_desc = np.concatenate((B_desc,frame_desc),axis = 0)
            B_desc_pos = np.concatenate((B_desc_pos,frame_desc_pos),axis = 0)
        #tmp = CalcuPairwiseSimilarityGPU(A,B)#CalcuPairwiseSimilarityMP(A,B)
        #print (A_desc.shape, B_desc.shape)
        #A_desc = A_desc[0:640,:]
        #B_desc = B_desc[0:640,:]
        print ("Processing grid(%d,%d)"%(i,j))
        tmp = self.CalcuPairwiseSimilarityGPUSuperExpressPosGraph(A_desc,B_desc, A_desc_num, B_desc_num, A_desc_pos, B_desc_pos, fm_dim_y, fm_dim_x, A_kp_neighbor, gaussian_patch)
       
        #sm[Afrom:Afrom+Alen, Bfrom:Bfrom+Blen] = tmp
        sm_row_list.append(tmp)
        Bfrom += Blen
        del B
        del Blen
      sm_row_array = np.concatenate(sm_row, axis=1)
      sm_list.append(sm_row_array)
      Afrom += Alen
      del A
      del Alen
    sm = np.concatenate(sm_list, axis=0)    
    return sm      
  
  
  
  # match two set of local feature from files and generate similarity matrix [N_a, N_b] 
  # kp_f_mat_a/kp_loc_mat_a/kp_score_mat_a: in shape [N_a].  
  #    kp_f_mat_a[i] in shape [N,feature_dim]
  #    kp_loc_mat_a[i] in shape [N,2]
  #    kp_score_mat_a[i] in shape [N]
  #    sm_dim_y, sm_dim_x:  dimension of similarity matrix
  #return similarity matrin in shape [N_a,N_b]
  def __ConcateFrameKeypoint(self,batch_frame_desc_mat_list, \
                                  batch_frame_desc_pos_mat_list, \
                                  batch_frame_desc_score_mat_list,\
                                  enable_collect_frame_neighbor=False,\
                                  kp_neighbor_x_radius=0, kp_neighbor_y_radius=0 
                                 ):
    # get the number of frame in batch
    batch_frame_num = batch_frame_desc_mat_list.shape[0]
    desc_num = np.zeros((batch_frame_num),dtype=np.int32)
    
    desc     = []
    desc_pos = []
    frame_neighbor_list = []
    for ii in range(0, batch_frame_num):
      frame_desc = batch_frame_desc_mat_list[ii]
      frame_desc_pos = batch_frame_desc_pos_mat_list[ii]
      frame_desc_num = frame_desc.shape[0]
      
    
        
      # find the closet neighbor of each desc in frame
      if enable_collect_frame_neighbor:
        frame_kp_closet_neighbor = self.FindEveryKPClosetNeighbor(kp_neighbor_x_radius, kp_neighbor_y_radius, frame_desc_pos, [], enable_matched=False)
        frame_neighbor_list += frame_kp_closet_neighbor
        
      desc_num[ii] = frame_desc_num
      
      desc.append(frame_desc)
      desc_pos.append(frame_desc_pos)
      
      
    desc     = np.concatenate(desc,axis = 0)
    desc_pos = np.concatenate(desc_pos,axis = 0)    
         
    return  desc, desc_pos, desc_num, frame_neighbor_list    
    
      
  def MultiBatch_Match2BatchFratureFilesPosGraph(self, kp_feature_files_a, kp_feature_files_b, kp_loc_files_a, kp_loc_files_b, kp_score_files_a, kp_score_files_b, fm_dim_y, fm_dim_x, kp_neighbor_x_radius=0, kp_neighbor_y_radius=0, gaussian_patch=None):
    num_files_a = len(kp_feature_files_a)
    num_files_b = len(kp_feature_files_b)
    # figure out a batch list
    batch_size = 2
    divid_sie = int(num_files_a/batch_size)
    reminder_size = int(num_files_a%batch_size)
    a_range_list = divid_sie*[batch_size]
    if reminder_size!=0:
      a_range_list.append(reminder_size)
    
    Afrom = 0
    sm_list = []
    
    a_file_idx = 0
    for i in range(0, len(a_range_list)):
        
      batch_size =  a_range_list[i]
      A_desc = []
      A_desc_pos = []
      A_desc_num = []
      A_kp_neighbor = []
      Alen = 0
      for kk in range(0, batch_size):
        Aa       = np.load(kp_feature_files_a[a_file_idx+kk], allow_pickle=True, encoding="latin1")
        #score_A = np.load(kp_score_files_a[i], allow_pickle=True, encoding="latin1")
        loc_Aa   = np.load(kp_loc_files_a[a_file_idx+kk], allow_pickle=True, encoding="latin1")
        Alen += Aa.shape[0]
        
        # reconcatenate A          
        Aa_desc, Aa_desc_pos, Aa_desc_num, Aa_kp_neighbor=self.__ConcateFrameKeypoint(Aa, loc_Aa, [], enable_collect_frame_neighbor=True, kp_neighbor_x_radius=kp_neighbor_x_radius, kp_neighbor_y_radius=kp_neighbor_y_radius)
        A_desc.append(Aa_desc) 
        A_desc_pos.append(Aa_desc_pos)
        A_desc_num.append(Aa_desc_num)
        A_kp_neighbor+=Aa_kp_neighbor
      
      A_desc = np.concatenate(A_desc,axis = 0)  
      A_desc_pos = np.concatenate(A_desc_pos,axis = 0) 
      A_desc_num = np.concatenate(A_desc_num,axis = 0)  
      a_file_idx += a_range_list[i]
      
      desc_a_gpu = tf.Variable(A_desc, dtype=tf.float16)
      print ("##################Test here#################",desc_a_gpu.shape)
      ####
      Bfrom = 0
      sm_row_list=[]
      for j in range(0, num_files_b):
        B       =  np.load(kp_feature_files_b[j], allow_pickle=True, encoding="latin1")
        #score_B =  np.load(kp_score_files_b[j], allow_pickle=True, encoding="latin1")
        loc_B   =  np.load(kp_loc_files_b[j], allow_pickle=True, encoding="latin1")
        Blen = B.shape[0]
        
        B_desc, B_desc_pos, B_desc_num, []=self.__ConcateFrameKeypoint(B, loc_B, [], enable_collect_frame_neighbor=False)
        #tmp = CalcuPairwiseSimilarityGPU(A,B)#CalcuPairwiseSimilarityMP(A,B)
        #print (A_desc.shape, B_desc.shape)
        #A_desc = A_desc[0:640,:]
        #B_desc = B_desc[0:640,:]
        print ("Processing grid(%d,%d)"%(i,j))
        #tmp = self.CalcuPairwiseSimilarityGPUSuperExpressPosGraph(A_desc,B_desc, A_desc_num, B_desc_num, A_desc_pos, B_desc_pos, fm_dim_y, fm_dim_x, A_kp_neighbor, gaussian_patch)
        tmp = self.CalcuPairwiseSimilarityGPUSuperExpressPosGraph_CMultiThread(A_desc,B_desc, A_desc_num, B_desc_num, A_desc_pos, B_desc_pos, fm_dim_y, fm_dim_x, A_kp_neighbor, gaussian_patch, desc_a_gpu=desc_a_gpu, desc_a_gpu_preload=True)
        sm_row_list.append(tmp)
        Bfrom += Blen
        del B
        del loc_B
        del Blen
      sm_row_array = np.concatenate(sm_row_list, axis=1)
      sm_list.append(sm_row_array)
      Afrom += Alen
      del A_desc
      del A_desc_pos
      del A_desc_num
      del A_kp_neighbor
      del Alen 
    sm = np.concatenate(sm_list, axis=0)       
    return sm       
 
 
 
    
  def Match2BatchFratureFilesPosGraph(self, kp_feature_files_a, kp_feature_files_b, kp_loc_files_a, kp_loc_files_b, kp_score_files_a, kp_score_files_b, fm_dim_y, fm_dim_x, kp_neighbor_x_radius=0, kp_neighbor_y_radius=0, gaussian_patch=None):
  
    num_files_a = len(kp_feature_files_a)
    num_files_b = len(kp_feature_files_b)
    
    Afrom = 0
    sm_list = []
    for i in range(0, num_files_a):
      A       = np.load(kp_feature_files_a[i], allow_pickle=True, encoding="latin1")
      #score_A = np.load(kp_score_files_a[i], allow_pickle=True, encoding="latin1")
      loc_A   = np.load(kp_loc_files_a[i], allow_pickle=True, encoding="latin1")
      
      Alen = A.shape[0]
      # reconcatenate A
      A_desc_num = np.zeros((Alen),dtype=np.int32)
      
      for ii in range(0, Alen):
        frame_desc = A[ii]
        frame_desc_pos = loc_A[ii]
        frame_desc_num = frame_desc.shape[0]
        
        # find the closet neighbor of each desc in frame
        frame_kp_closet_neighbor = self.FindEveryKPClosetNeighbor(kp_neighbor_x_radius, kp_neighbor_y_radius, frame_desc_pos, [], enable_matched=False)
        
        A_desc_num[ii] = frame_desc_num
        if ii ==0:
          A_desc = frame_desc
          A_desc_pos = frame_desc_pos
          A_kp_neighbor = frame_kp_closet_neighbor
        else:
          A_desc = np.concatenate((A_desc,frame_desc),axis = 0)
          A_desc_pos = np.concatenate((A_desc_pos,frame_desc_pos),axis = 0)    
          A_kp_neighbor += frame_kp_closet_neighbor
      ####
      Bfrom = 0
      sm_row_list=[]
      for j in range(0, num_files_b):
        B       =  np.load(kp_feature_files_b[j], allow_pickle=True, encoding="latin1")
        #score_B =  np.load(kp_score_files_b[j], allow_pickle=True, encoding="latin1")
        loc_B   =  np.load(kp_loc_files_b[j], allow_pickle=True, encoding="latin1")
        Blen = B.shape[0]
        # reconcatenate B
        B_desc_num = np.zeros((Blen),dtype=np.int32)
        for jj in range(0, Blen):
          frame_desc = B[jj]
          frame_desc_pos = loc_B[jj]
          frame_desc_num = frame_desc.shape[0]
          B_desc_num[jj] = frame_desc_num
          if jj ==0:
            B_desc = frame_desc
            B_desc_pos = frame_desc_pos
          else:
            B_desc = np.concatenate((B_desc,frame_desc),axis = 0)
            B_desc_pos = np.concatenate((B_desc_pos,frame_desc_pos),axis = 0)
        #tmp = CalcuPairwiseSimilarityGPU(A,B)#CalcuPairwiseSimilarityMP(A,B)
        #print (A_desc.shape, B_desc.shape)
        #A_desc = A_desc[0:640,:]
        #B_desc = B_desc[0:640,:]
        print ("Processing grid(%d,%d)"%(i,j))
        #tmp = self.CalcuPairwiseSimilarityGPUSuperExpressPosGraph(A_desc,B_desc, A_desc_num, B_desc_num, A_desc_pos, B_desc_pos, fm_dim_y, fm_dim_x, A_kp_neighbor, gaussian_patch)
        tmp = self.CalcuPairwiseSimilarityGPUSuperExpressPosGraph_CMultiThread(A_desc,B_desc, A_desc_num, B_desc_num, A_desc_pos, B_desc_pos, fm_dim_y, fm_dim_x, A_kp_neighbor, gaussian_patch)
        sm_row_list.append(tmp)
        Bfrom += Blen
        del B
        del Blen
      sm_row_array = np.concatenate(sm_row_list, axis=1)
      sm_list.append(sm_row_array)
      Afrom += Alen
      del A
      del Alen 
    sm = np.concatenate(sm_list, axis=0)       
    return sm       


  # HDV_array in shape [100,100,1024]
  # desc in shape [N,1024]
  # kp_loc in shape [N,1024]
  def CalculateHDC_HolisticFeature(self, desc, kp_loc, HDV_array):
    kp_loc = kp_loc.astype(np.int)
    kp_num = kp_loc.shape[0]
    result_hdv = np.zeros((desc.shape[1]), dtype=np.float)
    for i in range(0,kp_num):
      y = int((kp_loc[i,0])/10.0)
      x = int((kp_loc[i,1])/10.0)
      hdv = HDV_array[y,x,:]
      result_hdv += hdv*desc[i,:]
    return result_hdv
      
  # HDV_array in shape [100,100,1024]
  def HDC_HolisticMatch2BatchFratureFilesPosGraph(self, kp_feature_files_a, kp_feature_files_b, kp_loc_files_a, kp_loc_files_b, kp_score_files_a, kp_score_files_b, HDV_array):
    HDV_DIM = 1024
  
  
    num_files_a = len(kp_feature_files_a)
    num_files_b = len(kp_feature_files_b)
    # figure out a batch list    
    
    a_file_idx = 0
    Alen = 0 # number of files in A
    sm_list = []
    for i in range(0, num_files_a):    
      A       = np.load(kp_feature_files_a[i], allow_pickle=True, encoding="latin1")
      #score_A = np.load(kp_score_files_a[i], allow_pickle=True, encoding="latin1")
      loc_A   = np.load(kp_loc_files_a[i], allow_pickle=True, encoding="latin1")
      Alen = A.shape[0]
      holistic_A = np.zeros((Alen, HDV_DIM), dtype=np.float)
      # calculate holistic feature for each frame
      for frame_id in range(0,Alen):
        holistic_A[frame_id,:] =self.CalculateHDC_HolisticFeature(A[frame_id], loc_A[frame_id], HDV_array)
      
      ############## calculate for B
      Blen = 0
      sm_row_list=[]
      for j in range(0, num_files_b):
        B       =  np.load(kp_feature_files_b[j], allow_pickle=True, encoding="latin1")
        #score_B =  np.load(kp_score_files_b[j], allow_pickle=True, encoding="latin1")
        loc_B   =  np.load(kp_loc_files_b[j], allow_pickle=True, encoding="latin1")
        Blen = B.shape[0]
        holistic_B = np.zeros((Blen, HDV_DIM), dtype=np.float)
        # calculate holistic feature for each frame
        for frame_id in range(0,Blen):
          holistic_B[frame_id,:] = self.CalculateHDC_HolisticFeature(B[frame_id], loc_B[frame_id], HDV_array)  
          
        sm_partial = self.SuerExpress_cross_cosine_distanceGPU(holistic_A, holistic_B)        
        print ("Processing grid(%d,%d)"%(i,j))
        sm_row_list.append(sm_partial)
        del B
        del loc_B
        del Blen
      sm_row_array = np.concatenate(sm_row_list, axis=1)
      sm_list.append(sm_row_array)
      del A
      del loc_A
      del Alen 
    sm = np.concatenate(sm_list, axis=0)       
    return sm       




class CAccMatrixEx:
  def __init__(self, acc_file=""):
    # matrix acc supported datatype
    self.DTYPE_FP8_E4M3 = 0
    self.DTYPE_FP8_E5M2 = 1
    self.DTYPE_FP32 = 4
    self.DTYPE_INT8 = 5
    
    if acc_file == None:
      print ("Please assign an static shared library for the CAccMatrix")
      return
    
    acc_lib = ct.CDLL(acc_file)#'.lib/FeatureMatchAcc.so'
      
    self.MatAcc_Init_cuVIPRMatAcc = acc_lib.Init_cuVIPRMatAcc  
    self.MatAcc_Init_cuVIPRMatAcc.restype = ct.c_int32
      
    self.MatAcc_malloDataBaseMatFP32 = acc_lib.malloDataBaseMatFP32
    self.MatAcc_malloDataBaseMatFP32.restype = ct.c_int32
    self.MatAcc_malloDataBaseMatFP32.argtypes = [\
                       ct.c_int32 , ct.c_int32, ct.c_int32]    
      
    self.MatAcc_SetDBHolDataFP32 = acc_lib.SetDBHolDataFP32
    self.MatAcc_SetDBHolDataFP32.restype = ct.c_int32
    self.MatAcc_SetDBHolDataFP32.argtypes = [\
                       ct.c_int32 , ct.c_int32,\
                       ndpointer(ct.c_float, ndim=1, flags="C_CONTIGUOUS")]       
      
    self.MatAcc_BatchAttachDataBaseMatrixFP32 = acc_lib.BatchAttachDataBaseMatrixFP32
    self.MatAcc_BatchAttachDataBaseMatrixFP32.restype = ct.c_int32
    self.MatAcc_BatchAttachDataBaseMatrixFP32.argtypes = [\
                       ct.c_int32 , ct.c_int32,\
                       ndpointer(ct.c_float, ndim=1, flags="C_CONTIGUOUS"),\
                       ct.c_int32 ,ndpointer(ct.c_int32, ndim=1, flags="C_CONTIGUOUS")]     
     
    self.MatAcc_SetDatabaseFeatureMatInfo = acc_lib.SetDatabaseFeatureMatInfo
    self.MatAcc_SetDatabaseFeatureMatInfo.restype = ct.c_int32
    self.MatAcc_SetDatabaseFeatureMatInfo.argtypes = [\
                       ndpointer(ct.c_int32, ndim=1, flags="C_CONTIGUOUS"),\
                       ndpointer(ct.c_int32, ndim=1, flags="C_CONTIGUOUS")] 
                       
    self.MatAcc_SetDataBaseMatFP32 = acc_lib.SetDataBaseMatFP32
    self.MatAcc_SetDataBaseMatFP32.restype = ct.c_int32
    self.MatAcc_SetDataBaseMatFP32.argtypes = [\
                       ct.c_int32 , ct.c_int32,\
                       ndpointer(ct.c_float, ndim=1, flags="C_CONTIGUOUS")]    
 
 
    self.MatAcc_BatchQueryHolisticFeatureFP32 = acc_lib.BatchQueryHolisticFeatureFP32
    self.MatAcc_BatchQueryHolisticFeatureFP32.restype = ct.c_int32
    self.MatAcc_BatchQueryHolisticFeatureFP32.argtypes = [\
                       ct.c_int32 , ct.c_int32,\
                       ndpointer(ct.c_float, ndim=1, flags="C_CONTIGUOUS"),\
                       ndpointer(ct.c_float, ndim=1, flags="C_CONTIGUOUS")] 
          
                                  
    self.MatAcc_BatchQueryDataBaseFP32 = acc_lib.BatchQueryDataBaseFP32
    self.MatAcc_BatchQueryDataBaseFP32.restype = ct.c_int32
    self.MatAcc_BatchQueryDataBaseFP32.argtypes = [\
                       ct.c_int32 , ct.c_int32,\
                       ndpointer(ct.c_float, ndim=1, flags="C_CONTIGUOUS"),\
                       ndpointer(ct.c_float, ndim=1, flags="C_CONTIGUOUS")] 
   
    self.MatAcc_QueryDataBaseN_FP32 = acc_lib.QueryDataBaseN_FP32
    self.MatAcc_QueryDataBaseN_FP32.restype = ct.c_int32
    self.MatAcc_QueryDataBaseN_FP32.argtypes = [\
                       ct.c_int32 , ct.c_int32,\
                       ndpointer(ct.c_float, ndim=1, flags="C_CONTIGUOUS"), \
                       ct.c_int32, ndpointer(ct.c_int32, ndim=1, flags="C_CONTIGUOUS"),ct.c_int32,\
                       ndpointer(ct.c_float, ndim=1, flags="C_CONTIGUOUS")]
   
                       
    self.MatAcc_NormalizeBatchVectorFP32 = acc_lib.NormalizeBatchVectorFP32
    self.MatAcc_NormalizeBatchVectorFP32.restype = ct.c_int
    self.MatAcc_NormalizeBatchVectorFP32.argtypes = [\
                       ct.c_int32 , ct.c_int32,\
                       ndpointer(ct.c_float, ndim=1, flags="C_CONTIGUOUS"),\
                       ndpointer(ct.c_float, ndim=1, flags="C_CONTIGUOUS")]  
    
    
    
    

    # MatAccEx APIs 
    self.MatAcc_InitMatAccExEnv = acc_lib.InitMatAccExEnv
    self.MatAcc_InitMatAccExEnv.argtypes = [ct.c_int32]   
    
    self.MatAcc_malloDataBaseMat = acc_lib.malloDataBaseMat
    self.MatAcc_malloDataBaseMat.restype = ct.c_int32
    self.MatAcc_malloDataBaseMat.argtypes = [\
                       ct.c_int32 , ct.c_int32, ct.c_int32] 
                       
    self.MatAcc_SetDBInfo = acc_lib.SetDBInfo
    self.MatAcc_SetDBInfo.restype = ct.c_int32
    self.MatAcc_SetDBInfo.argtypes = [\
                       ndpointer(ct.c_int32, ndim=1, flags="C_CONTIGUOUS"),\
                       ndpointer(ct.c_int32, ndim=1, flags="C_CONTIGUOUS")]  
                       
    self.MatAcc_BatchAttachDataBaseMatrix = acc_lib.BatchAttachDataBaseMatrix
    self.MatAcc_BatchAttachDataBaseMatrix.restype = ct.c_int32
    self.MatAcc_BatchAttachDataBaseMatrix.argtypes = [\
                       ct.c_int32 , ct.c_int32,\
                       ct.c_char_p,\
                       ct.c_int32 ,ndpointer(ct.c_int32, ndim=1, flags="C_CONTIGUOUS")]        
    
    self.MatAcc_BatchAttachHolFeature = acc_lib.BatchAttachHolFeature
    self.MatAcc_BatchAttachHolFeature.restype = ct.c_int32
    self.MatAcc_BatchAttachHolFeature.argtypes = [\
                       ct.c_int32 , ct.c_int32,\
                       ct.c_char_p]        
               
    self.MatAcc_BatchQueryHolisticFeature = acc_lib.BatchQueryHolisticFeature
    self.MatAcc_BatchQueryHolisticFeature.restype = ct.c_int32
    self.MatAcc_BatchQueryHolisticFeature.argtypes = [\
                       ct.c_int32 , ct.c_int32,\
                       ct.c_char_p,\
                       ndpointer(ct.c_float, ndim=1, flags="C_CONTIGUOUS")]            
     
    self.MatAcc_BatchQueryDataBase = acc_lib.BatchQueryDataBase
    self.MatAcc_BatchQueryDataBase.restype = ct.c_int32
    self.MatAcc_BatchQueryDataBase.argtypes = [\
                       ct.c_int32 , ct.c_int32,\
                       ct.c_char_p,\
                       ndpointer(ct.c_float, ndim=1, flags="C_CONTIGUOUS")]  
               
                      
    ### FP32 to FP8 conversion ###
    self.MatAcc_CVT_MatFP32ToFP8E4M3 = acc_lib.CVT_MatFP32ToFP8E4M3
    self.MatAcc_CVT_MatFP32ToFP8E4M3.restype = ct.c_int
    self.MatAcc_CVT_MatFP32ToFP8E4M3.argtypes = [\
                       ndpointer(ct.c_float, ndim=1, flags="C_CONTIGUOUS"), ct.c_uint32,\
                       ct.c_float, \
                       ndpointer(ct.c_ubyte, ndim=1, flags="C_CONTIGUOUS")]     
        
    self.MatAcc_CVT_MatFP32ToFP8E5M2 = acc_lib.CVT_MatFP32ToFP8E5M2
    self.MatAcc_CVT_MatFP32ToFP8E5M2.restype = ct.c_int
    self.MatAcc_CVT_MatFP32ToFP8E5M2.argtypes = [\
                       ndpointer(ct.c_float, ndim=1, flags="C_CONTIGUOUS"), ct.c_uint32,\
                       ct.c_float, \
                       ndpointer(ct.c_ubyte, ndim=1, flags="C_CONTIGUOUS")]   
    ### FP8 to FP32 conversion ###
    self.MatAcc_CVT_MatFP8E4M3ToFP32 = acc_lib.CVT_MatFP8E4M3ToFP32
    self.MatAcc_CVT_MatFP8E4M3ToFP32.restype = ct.c_int
    self.MatAcc_CVT_MatFP8E4M3ToFP32.argtypes = [\
                       ndpointer(ct.c_ubyte, ndim=1, flags="C_CONTIGUOUS"), ct.c_uint32,\
                       ct.c_float, \
                       ndpointer(ct.c_float, ndim=1, flags="C_CONTIGUOUS")]    
    # call the initialization function
    
    self.base_mat_h = 0
    self.base_mat_w = 0
    self.hol_mat_h = 0
    self.hol_mat_w = 0 
    self.__time_analysis = 0
    
  def Dtype2Digit(self, dtype):
    if dtype=="FP8_E4M3":
      return np.uint8, self.DTYPE_FP8_E4M3
    if dtype=="FP8_E5M2":
      return np.uint8, self.DTYPE_FP8_E5M2
    if dtype=="FP32":
      return np.float32, self.DTYPE_FP32
    if dtype=="INT8":
      return np.int8, self.DTYPE_INT8     
              
  def __2Dto1D(self, mat, dtype):
    mat_h = mat.shape[0]
    mat_w = mat.shape[1]
    mat_in_row = np.reshape(mat, (mat_h*mat_w))
    mat_in_row = np.ascontiguousarray(mat_in_row, dtype=dtype)        
    return mat_in_row  

  # reshape the mat to shape mat_h, mat_w
  def __1Dto2D(self, mat, mat_h, mat_w):
    mat = np.reshape(mat, (mat_h, mat_w))
    return mat

  # this function transfer the matrix from "C order" to "F order" storage
  def __Transfer2ColumnMajot(self, mat):
    mat = np.transpose(mat)
    return mat
  
  def __Transfer2RowMajor(self, mat):  
    mat = np.transpose(mat)
    return mat
   
  def AllocateDB_GPU_Memory(self, mat_h=0, mat_w=0, db_frame_num = 0): 
    state = self.MatAcc_malloDataBaseMat(mat_h, mat_w, db_frame_num)
    if state==0:
      self.base_mat_h = mat_h
      self.base_mat_w = mat_w 
      print ("Allocate DB GPU memory for local feature success!")
  def SetDBHolMat(self, mat):
    mat_w = mat.shape[1]
    mat_h = mat.shape[0]
    mat = self.__2Dto1D(mat, dtype=np.float32)
    mat_in_byte = mat.tobytes()
    state = self.MatAcc_BatchAttachHolFeature(mat_h, mat_w, mat_in_byte)
    if state==0:
      print ("Allocate and assign holistic feature for DB success!")
      self.hol_mat_h = mat_h
      self.hol_mat_w = mat_w
    
    
  def SetBaseMat(self, mat):
    mat_h = mat.shape[1]
    mat_w = mat.shape[0]
    #mat_F = self.__Transfer2ColumnMajot(mat)
    mat = self.__2Dto1D(mat, dtype=np.float32)
    state = self.MatAcc_SetDataBaseMatFP32(mat_h, mat_w, mat)   
    if state==0:
      print ("Successfully set base Matrix to GPU memory with shape=(%d,%d)"%(mat_h, mat_w))   
      self.base_mat_h = mat_h
      self.base_mat_w = mat_w 
   
    else:
      print ("Failed to set base Matrix to GPU memory with shape=(%d,%d)"%(mat_h, mat_w))                      

  def SetDBMatInfo(self, db_frame_keypoint_num, db_frame_base_mat_start_col):
    db_frame_keypoint_num       = np.ascontiguousarray(db_frame_keypoint_num, dtype=np.int32) 
    db_frame_base_mat_start_col = np.ascontiguousarray(db_frame_base_mat_start_col, dtype=np.int32) 
    state=self.MatAcc_SetDBInfo(db_frame_keypoint_num, db_frame_base_mat_start_col)
    if state!=0:
      print ("Failed to set the DB info!!")
      
  def AttachGPU_DB_feature(self, mat, attached_frame_num, attached_frame_kp_num):
    mat_h = mat.shape[0]
    mat_w = mat.shape[1]
    mat = self.__2Dto1D(mat, dtype=mat.dtype)
    attached_frame_kp_num = np.ascontiguousarray(attached_frame_kp_num, dtype=np.int32)  
    data_in_byte = mat.tobytes()  
    state = self.MatAcc_BatchAttachDataBaseMatrix(mat_h, mat_w, data_in_byte, attached_frame_num, attached_frame_kp_num)
    if state==0:
      print ("Attach DB local feature success with feature shape = (%d, %d)!"%(mat_h, mat_w))

  def HolQuery(self, q_hol_mat):
    mat_w = q_hol_mat.shape[1]
    mat_h = q_hol_mat.shape[0]
    ####start_time = time.time()
    #mat_F = self.__Transfer2ColumnMajot(q_mat)
    q_hol_mat = self.__2Dto1D(q_hol_mat, dtype=np.float32)
    q_hol_mat_in_byte = q_hol_mat.tobytes()
    ####end_time = time.time()
    ###print (end_time-start_time)
    result = np.zeros((mat_h*self.hol_mat_h),dtype=np.float32) 
    #start_time = time.time()
    state = self.MatAcc_BatchQueryHolisticFeature(mat_h, mat_w, q_hol_mat_in_byte, result)
    #end_time = time.time()
    #self.__time_analysis += end_time-start_time
    #if state==0:
      #result = self.__1Dto2D(result_F, self.base_mat_w, mat_w )
      #result = self.__Transfer2RowMajor(result)
    return result

  def Query(self, q_mat):
    mat_h = q_mat.shape[0]
    mat_w = q_mat.shape[1]
    ####start_time = time.time()
    #mat_F = self.__Transfer2ColumnMajot(q_mat)
    q_mat = self.__2Dto1D(q_mat, dtype=q_mat.dtype)
    ####end_time = time.time()
    ###print (end_time-start_time)
    result = np.zeros((mat_h*self.base_mat_h),dtype=np.float32) 
    q_mat_in_byte = q_mat.tobytes()
    #start_time = time.time()
    state = self.MatAcc_BatchQueryDataBase(mat_h, mat_w, q_mat_in_byte, result)
    #end_time = time.time()
    #self.__time_analysis += end_time-start_time
    #if state==0:
      #result = self.__1Dto2D(result_F, self.base_mat_w, mat_w )
      #result = self.__Transfer2RowMajor(result)
    return result
  
  def QueryK(self, q_mat, topKIdx, total_topk_kp):
    mat_h = q_mat.shape[1]
    mat_w = q_mat.shape[0]
    q_mat = self.__2Dto1D(q_mat, dtype=np.float32)
    result = np.zeros((total_topk_kp*mat_w),dtype=np.float32) 
    result = np.ascontiguousarray(result, dtype=np.float32) 
    topKIdx = np.ascontiguousarray(topKIdx, dtype=np.int32) 
    state = self.MatAcc_QueryDataBaseN_FP32(mat_h, mat_w, q_mat, topKIdx.shape[0], topKIdx, total_topk_kp, result)
    return result   
  def NormalizeFeatures(self, feature_mat):
    mat_h = feature_mat.shape[1]
    mat_w = feature_mat.shape[0]
    feature_mat = self.__2Dto1D(feature_mat, dtype=np.float32)
    state = self.MatAcc_NormalizeBatchVectorFP32(mat_h, mat_w, feature_mat, feature_mat)  
    feature_mat = self.__1Dto2D(feature_mat, mat_w, mat_h)
    return feature_mat

  ###################### Image local feature datatype convertion ######################
  
  # This function convert the local feature of an image from np.float32/64 to FP8E4M3/E5M2 datatype. 
  # Each image feature is first normalized before converted to FP8 datatype.
  # img_local_feature: np array in shape [N, dim], where N indicates the Number of lcoal feature in the image, 
  #                    and the dim represents the local feature dimmension
  # datatype:     Indicates the conversion target datatype can be  ["FP8", "INT8", "INT16", "FP16"]
  # data_subtype: String indicates the fp8 type. It can be either "E4M3" or "E5M2"
  # return: The image local feature in fp8 datatype. The np.uint8 datatye are used to store the FP8 data.
  def CVT_ImageLocalFeatureDataType(self, img_local_feature, datatype="FP8", data_subtype="E4M3"):
    # normalize each local feature
    print(img_local_feature.shape)
    print(img_local_feature)
    feature = self.NormalizeFeatures(img_local_feature) 
    print(feature)
    # flatten the 2D image local feature array to 1D vector
    N   = feature.shape[0]
    dim = feature.shape[1] 
    feature_total_dim = N*dim
    feature = np.reshape(feature, (feature_total_dim)) 
    # continous storage mode
    feature = np.ascontiguousarray(feature, dtype=np.float32)
    
    if datatype == "FP8":
      # memory to store the converted fp8 data
      cvt_feature = np.zeros((feature_total_dim),dtype=np.uint8)
      # do the convertion
      if data_subtype=="E4M3":
        self.MatAcc_CVT_MatFP32ToFP8E4M3(feature, feature_total_dim, 1.0, cvt_feature)     
      elif data_subtype=="E5M2":
        self.MatAcc_CVT_MatFP32ToFP8E5M2(feature, feature_total_dim, 1.0, cvt_feature) 
      # reshape back to (N, dim) 2D dimension    
      cvt_feature = np.reshape(cvt_feature, (N,dim))
      return cvt_feature
    elif datatype == "INT8":
      return 0
    elif datatype == "INT16":
      return 0
    elif datatype == "FP16":
      return 0








