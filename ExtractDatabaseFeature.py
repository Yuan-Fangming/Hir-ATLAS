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
import sys, getopt
import os
import yaml
import fnmatch


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
  database_img_dir     = conf_dict['database_img_dir']
  database_feature_dir = conf_dict['database_feature_dir']

  vgg16_weights_file = conf_dict['vgg16_weights_file']
  ckpt_file          = conf_dict['ckpt_file']
  pca_file_prefix    = conf_dict['pca_file_prefix']
  HDC_file           = conf_dict['HDC_file']
  cuda_acc_file       = conf_dict['cuda_acc_file']
  
  database_name = conf_dict['database_name']
  batch_size    = int(conf_dict['batch_size'])

  kp_desc_dtype = conf_dict['kp_desc_dtype']
  
  enable_lpg_extraction = conf_dict['enable_lpg_extraction']
  lpg_window_x = int(conf_dict['lpg_window_x'])
  lpg_window_y = int(conf_dict['lpg_window_y'])





  AtlasFeatureExtractor = AtlasFeatureExtraction(\
               vgg16_weights_file = vgg16_weights_file,\
               ckpt_file = ckpt_file,\
               pca_file  = pca_file_prefix,\
               HDC_file  = HDC_file,\
               cuda_acc_file = cuda_acc_file,\
               enable_atten =True)


  db_img_list = os.listdir(database_img_dir)
  db_img_list.sort()
  db_img_num = len(db_img_list)
  db_batch_num = int(db_img_num/batch_size)
  last_batch_size = db_img_num%batch_size
  each_batch_size = [batch_size]*db_batch_num
  if last_batch_size==1:
    each_batch_size[-1]+=1
  elif last_batch_size!=0:
    each_batch_size.append(last_batch_size)
    db_batch_num += 1
  
  feature_extraction_method, _ = AtlasFeatureExtractor.GenerateDBFileNames(database_name, kp_desc_dtype, enable_lpg_extraction, lpg_window_x, lpg_window_y)
  
  DB_feature_file_template          = database_feature_dir+'/'+feature_extraction_method
  DB_feature_loc_file_template      = database_feature_dir+'/'+feature_extraction_method
  DB_feature_loc_norm_file_template = database_feature_dir+'/'+feature_extraction_method
  
  DB_feature_lpg_neighbor_file_template = database_feature_dir+'/'+feature_extraction_method
  DB_feature_lpg_neighbor_delimiter_file_template = database_feature_dir+'/'+feature_extraction_method
  
  DB_holistic_feature_file_template = database_feature_dir+'/'+feature_extraction_method
  
  start_img_idx = 0
  for i in range(0,db_batch_num):
    print ("######Start to extract database features for batch %d......  ######"%(i))
    img_list = db_img_list[start_img_idx:start_img_idx+each_batch_size[i]]
    start_img_idx += each_batch_size[i]
    loc, desc, norm_loc, lpg_neighbor, lpg_neighbor_delimiter = AtlasFeatureExtractor.ExtractImgLocalFeature(\
                                                                  database_img_dir,img_list, rescale=1.0, \
                                                                  local_feature_datatype = kp_desc_dtype,\
                                                                  nms_interp_enable=True,\
                                                                  normalize_loc=True, \
                                                                  find_lpg_closest_neighbor = enable_lpg_extraction,\
                                                                  closest_neighbor_win_width_x=lpg_window_x, closest_neighbor_win_width_y=lpg_window_y)
    np.save(DB_feature_file_template%('desc',i), desc)
    np.save(DB_feature_loc_file_template%('loc',i), loc)
    np.save(DB_feature_loc_norm_file_template%('norm-loc',i), norm_loc)
    np.save(DB_feature_lpg_neighbor_file_template%('lpg-neighbor',i), lpg_neighbor)
    np.save(DB_feature_lpg_neighbor_delimiter_file_template%('delimiter-lpg-neighbor',i), lpg_neighbor_delimiter)
  
  
  