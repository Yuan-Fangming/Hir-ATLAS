# This file is part of the Hir-ATLAS distribution (https://github.com/Yuan-Fangming/Hir-ATLAS).
# Copyright (c) 2024 Fangming Yuan.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from ATLAS import *
import os
import sys, getopt
import scipy.io
import h5py
import hdf5storage
import HirATLAS as ImgSim
from VIPRDLpy3 import *
import yaml
import fnmatch
from Acc_HirATLAS import Alg2Digit

def AddPathFeatureFiles(dir, file_list):
  file_list.sort()
  for i in range(0,len(file_list)):
     file_list[i] = dir+'/'+ file_list[i]
  return file_list

gt_mat_file =  "/home/fangming/Home_office/benchmark/dataset/ground_truth/GardensPointWalking/day_left--night_right/gt.mat"
gt = scipy.io.loadmat(gt_mat_file)
gt_hard = gt["GT"]["GThard"][0,0]
gt_soft = gt["GT"]["GTsoft"][0,0]


yaml_file =''
arg_tamplate = "y:"  # -y <yaml file path> 
if __name__ =="__main__":
  # parsing input argument
  opts,args = getopt.getopt(sys.argv[1:], arg_tamplate,["yaml="] )
  
  for opt,arg in opts:
    if opt in("-y","--yaml"):
      yaml_file = arg

  # open configuration file
  f = open(yaml_file)
  conf_dict = yaml.load(f,Loader=yaml.FullLoader) 
  f.close()
  # config parameters
  query_img_dir     = conf_dict['query_img_dir']
  database_feature_dir = conf_dict['database_feature_dir']

  vgg16_weights_file = conf_dict['vgg16_weights_file']
  ckpt_file          = conf_dict['ckpt_file']
  pca_file_prefix    = conf_dict['pca_file_prefix']
  HDC_file           = conf_dict['HDC_file']
  acc_file           = conf_dict['acc_file']
  cuda_acc_file       = conf_dict['cuda_acc_file']
  
  database_name = conf_dict['database_name']

  TopN = int(conf_dict['TopN'])

  kp_desc_dtype = conf_dict['kp_desc_dtype']
  
  rerank_method = conf_dict['rerank_method']
  lpg_window_x = int(conf_dict['lpg_window_x'])
  lpg_window_y = int(conf_dict['lpg_window_y'])

  is_enable_lpg = False
  if rerank_method == "LPG":
    is_enable_lpg = True
  elif rerank_method == "PW":
    dummy = dummy  
  elif rerank_method == "RANSAC":
    dummy = dummy      
  algorithm = Alg2Digit(rerank_method) 
  
  if TopN==-1:
    enable_topK_retrieve = False
  else:
    enable_topK_retrieve = True     
  # get query image files
  query_img_list = os.listdir(query_img_dir)
  query_img_list.sort()
  query_img_num = len(query_img_list)


  
  
  
  AtlasFeatureExtractor = AtlasFeatureExtraction(\
               vgg16_weights_file = vgg16_weights_file,\
               ckpt_file=ckpt_file,\
               pca_file=pca_file_prefix,\
               HDC_file=HDC_file,\
               cuda_acc_file = cuda_acc_file,\
               enable_atten =True)
  
  
  # Search the files for DB dataset 

  _, feature_extraction_method = AtlasFeatureExtractor.GenerateDBFileNames(database_name, kp_desc_dtype, is_enable_lpg, lpg_window_x, lpg_window_y)
  DB_kp_desc_files = fnmatch.filter(os.listdir(database_feature_dir), feature_extraction_method%('desc'))  
  DB_kp_loc_files  = fnmatch.filter(os.listdir(database_feature_dir), feature_extraction_method%('norm-loc'))  
  DB_kp_neighbor_files           = fnmatch.filter(os.listdir(database_feature_dir), feature_extraction_method%('lpg-neighbor'))  
  DB_kp_neighbor_delimiter_files = fnmatch.filter(os.listdir(database_feature_dir), feature_extraction_method%('delimiter-lpg-neighbor'))  
  DB_holistic_feature_file = [] #fnmatch.filter(os.listdir(database_feature_dir), feature_extraction_method%('desc'))    
  
  DB_kp_desc_files = AddPathFeatureFiles(database_feature_dir, DB_kp_desc_files) 
  DB_kp_loc_files = AddPathFeatureFiles(database_feature_dir, DB_kp_loc_files) 
  DB_kp_neighbor_files = AddPathFeatureFiles(database_feature_dir, DB_kp_neighbor_files) 
  DB_kp_neighbor_delimiter_files = AddPathFeatureFiles(database_feature_dir, DB_kp_neighbor_delimiter_files) 
  DB_holistic_feature_file =  "/media/fangming/yfmbuff/hdc-localSMPAtt/hdc-localSMPAtt/descriptors/GardensPointWalking/day_left/HDC-lspedSMPAtt_nx5_ny9.mat"  #AddPathFeatureFiles(database_feature_dir, DB_holistic_feature_file) 
  
  
  
  imgsim=ImgSim.HirATLAS(         DB_LocFeature_files=DB_kp_desc_files, \
                                  DB_LocFeature_loc_files=DB_kp_loc_files, \
                                  DB_loc_H=100, DB_loc_W=100, \
                                  DB_HolFeature_files=DB_holistic_feature_file, \
                                  acc_file=acc_file, \
                                  cuda_acc_file = cuda_acc_file, \
                                  DB_kp_spatial_closest_neighbor_files=DB_kp_neighbor_files, \
                                  DB_kp_spatial_closest_neighbor_delimiter_files=DB_kp_neighbor_delimiter_files, \
                                  gauss2d_y_size = 150, gauss2d_x_size = 150, \
                                  gauss_patch_x_size = 410, gauss_patch_y_size = 410, \
                                  gauss_sigma=1,\
                                  enable_topK=enable_topK_retrieve,\
                                  DB_feature_gpu_preload=True,\
                                  #Feature_dtype="FP32",\
                                  Feature_dtype=kp_desc_dtype,\
                                  DB_feature_normalize=False,\
                                  TopN_retrieve_thread_num=1) 
  
  sim = np.zeros((200,200), dtype=np.float32)
  
  
  query_hol_desc = scipy.io.loadmat("/media/fangming/yfmbuff/hdc-localSMPAtt/hdc-localSMPAtt/descriptors/GardensPointWalking/night_right/HDC-lspedSMPAtt_nx5_ny9.mat")['Y']
  query_hol_desc = imgsim.NormalizeFeatures(query_hol_desc)
  
  for i in range(0,query_img_num):
    
    query_loc, query_desc, query_loc_norm, a,b = AtlasFeatureExtractor.ExtractImgLocalFeature(query_img_dir, [query_img_list[i]], rescale=1.0, \
                                                                         local_feature_datatype= kp_desc_dtype,\
                                                                         nms_interp_enable=True, \
                                                                         normalize_loc=True, \
                                                                         find_lpg_closest_neighbor = is_enable_lpg,\
                                                                         closest_neighbor_win_width_x=60, closest_neighbor_win_width_y=60)
    
    query_loc_norm  = query_loc_norm[0,:,:]
    query_desc = query_desc[0,:,:]
    
    sim[:,i:i+1] = imgsim.cACC_MultiQueryRetrieveInDB_Versatile(\
                                                [query_desc], \
                                                [query_loc_norm], \
                                                query_hol_desc[i:i+1,:], \
                                                retrieve_top_N=TopN, thread_num = 8,\
                                                alg_type=algorithm \
                                                )
    
  plt.imshow(sim)
  plt.show(block=True)

  avgP,A,B,D,E,F = CreatePR(sim=sim, GThard=gt_hard, GTsoft=gt_soft, cond_a = False, cond_b = False)

  print ("avgP =%f"%(avgP)) 
