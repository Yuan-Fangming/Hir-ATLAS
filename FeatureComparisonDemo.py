import scipy.io
import h5py
import hdf5storage
import HirATLAS as ImgSim
import os, sys
import numpy as np
import time
import matplotlib.pyplot as plt

def AddPathFeatureFiles(dir, file_list):
  file_list.sort()
  for i in range(0,len(file_list)):
     file_list[i] = dir+'/'+ file_list[i]
  return file_list
def LoadFeatureFiles(dir, file_list):
  file_list.sort()
  content_mat = []
  for i in range(0,len(file_list)):
    content_mat.append(np.load(dir+'/'+file_list[i], allow_pickle=True, encoding="latin1"))
  content_mat = np.concatenate(content_mat, axis=0)
  return content_mat
  
  
  
  
  
# Root directory of the local feature files for query and database sequences
local_feature_file_dir = " "
# holistic feature files for query and database sequence
holistic_feature_DB_file = [" "]
holistic_feature_query_file = [" "]
gt_mat_file =  " "

kp_desc_DB_files = ["BenchMrkSys_GardensPointWalking_day_right_1.0_LocalSPED-SMP-Attention-1x1FMSplit-epoch120-POS_KP_feature_00000000.npy","BenchMrkSys_GardensPointWalking_day_right_1.0_LocalSPED-SMP-Attention-1x1FMSplit-epoch120-POS_KP_feature_00000001.npy"]
kp_desc_query_files = ["BenchMrkSys_GardensPointWalking_night_right_1.0_LocalSPED-SMP-Attention-1x1FMSplit-epoch120-POS_KP_feature_00000000.npy","BenchMrkSys_GardensPointWalking_night_right_1.0_LocalSPED-SMP-Attention-1x1FMSplit-epoch120-POS_KP_feature_00000001.npy"]


kp_loc_DB_files  = ["BenchMrkSys_GardensPointWalking_day_right_1.0_LocalSPED-SMP-Attention-1x1FMSplit-epoch120-POS_KP_norm_loc_00000000.npy","BenchMrkSys_GardensPointWalking_day_right_1.0_LocalSPED-SMP-Attention-1x1FMSplit-epoch120-POS_KP_norm_loc_00000001.npy"]
kp_loc_query_files  = ["BenchMrkSys_GardensPointWalking_night_right_1.0_LocalSPED-SMP-Attention-1x1FMSplit-epoch120-POS_KP_norm_loc_00000000.npy","BenchMrkSys_GardensPointWalking_night_right_1.0_LocalSPED-SMP-Attention-1x1FMSplit-epoch120-POS_KP_norm_loc_00000001.npy"]









kp_DB_neighbor_files = ["BenchMrkSys_GardensPointWalking_day_right_1.0_LocalSPED-SMP-Attention-1x1FMSplit-epoch120-POS_KP_norm_spatial_closest_neighbor_Xradius_Yradius_60_60_00000000.npy", "BenchMrkSys_GardensPointWalking_day_right_1.0_LocalSPED-SMP-Attention-1x1FMSplit-epoch120-POS_KP_norm_spatial_closest_neighbor_Xradius_Yradius_60_60_00000001.npy"]
kp_DB_neighbor_delimiter_files = ["BenchMrkSys_GardensPointWalking_day_right_1.0_LocalSPED-SMP-Attention-1x1FMSplit-epoch120-POS_KP_norm_spatial_closest_neighbor_delimiter_Xradius_Yradius_60_60_00000000.npy", "BenchMrkSys_GardensPointWalking_day_right_1.0_LocalSPED-SMP-Attention-1x1FMSplit-epoch120-POS_KP_norm_spatial_closest_neighbor_delimiter_Xradius_Yradius_60_60_00000001.npy"]




gt = scipy.io.loadmat(gt_mat_file)
gt_hard = gt["GT"]["GThard"][0,0]
gt_soft = gt["GT"]["GTsoft"][0,0]


kp_desc_DB_files = AddPathFeatureFiles(local_feature_file_dir, kp_desc_DB_files)
kp_loc_DB_files = AddPathFeatureFiles(local_feature_file_dir, kp_loc_DB_files)
kp_DB_neighbor_files = AddPathFeatureFiles(local_feature_file_dir, kp_DB_neighbor_files)
kp_DB_neighbor_delimiter_files = AddPathFeatureFiles(local_feature_file_dir, kp_DB_neighbor_delimiter_files) 


query_feature_array  = LoadFeatureFiles(local_feature_file_dir, kp_desc_query_files)
query_feature_loc_array = LoadFeatureFiles(local_feature_file_dir, kp_loc_query_files)
query_holistic_feature = scipy.io.loadmat(holistic_feature_query_file[0])['Y']





imgsim=ImgSim.HirATLAS(DB_LocFeature_files=kp_desc_DB_files, DB_LocFeature_loc_files=kp_loc_DB_files, DB_loc_H=100, DB_loc_W=100, \
                                  DB_HolFeature_files=holistic_feature_DB_file[0], \
                                  acc_file='AccLib/libvipracc', \
                                  cuda_acc_file = 'AccLib/libcuVIPRACCLib', \
                                  DB_kp_spatial_closest_neighbor_files=kp_DB_neighbor_files, DB_kp_spatial_closest_neighbor_delimiter_files=kp_DB_neighbor_delimiter_files, \
                                  gauss2d_y_size = 150, gauss2d_x_size = 150, \
                                  gauss_patch_x_size = 410, gauss_patch_y_size = 410, \
                                  gauss_sigma=1,\
                                  enable_topK=False,\
                                  DB_feature_gpu_preload=True,\
                                  Feature_dtype="FP32",\
                                  DB_feature_normalize=True) 




sim = np.zeros((200,200), dtype=np.float32)

query_holistic_feature = imgsim.NormalizeFeatures(query_holistic_feature)

query_num = query_holistic_feature.shape[0]
query_batch = 1
step     = int(query_num/query_batch)
reminder = int(query_num)%int(query_batch)
if reminder==0:
  end_iter_num = step
else:
  end_iter_num = step+1  

# normalize local feature
for ii in range(0, query_feature_array.shape[0]):
  query_feature_array[ii] = imgsim.NormalizeFeatures(query_feature_array[ii])



start_time = time.time()
for i in range(0, end_iter_num):
  frame_start = i*query_batch  
  if i!=step:
    frame_end = (i+1)*query_batch    
  else:
    frame_end = query_num   
  sim[:,frame_start:frame_end] = imgsim.cACC_MultiQueryRetrieveInDB_Versatile(query_feature_array[frame_start:frame_end], query_feature_loc_array[frame_start:frame_end], query_holistic_feature[frame_start:frame_end], \
                                                retrieve_top_N=-1, thread_num = 8,\
                                                #alg_type=imgsim.ALG_TYPE_PAIRWISE_RANSAC \
                                                alg_type=imgsim.ALG_TYPE_PAIRWISE_POSGRAPH \
                                                #alg_type=imgsim.ALG_TYPE_RSS\
                                                #alg_type=imgsim.ALG_TYPE_LPG_MATCH \
                                                #alg_type=imgsim.ALG_TYPE_PAIRWISE \
                                                #alg_type=imgsim.ALG_TYPE_HOLISTIC_FEATURE_MATCH\
                                                #alg_type=imgsim.PAIRWISE_STARGRAPH\
                                                )


  
end_time = time.time()


plt.imshow(sim)
plt.show(block=True)

avgP,A,B,D,E,F = CreatePR(sim=sim, GThard=gt_hard, GTsoft=gt_soft, cond_a = False, cond_b = False)


print ("avgP =%f"%(avgP)) 
print ("Processing 200 image in %s seconds"%(end_time-start_time));

a = 0
