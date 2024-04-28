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

import tensorflow as tf
import fnmatch
import cv2
import gc
import skimage.io
import skimage.transform
import numpy as np
from Acc_HirATLAS import * 
import os

#Ues only necessory GPU memory 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

################################################################################
#ckpt_file:     The check point file of the ATLAS feature extraction model
#pca_file:      The PCA matrix file used for the local feature PCA reduction
#HDC_file:      The HDC matric file used for the Holistic feature aggregation
#cuda_acc_file: The cuda acceleration file same as HirATLAS class. Here use only the 
#enable_atten:  Flag indicates if include the AttNet in the model (If not enabled the ATLAS pipeline can local the check point file of LocalSPED/LocalSPED-SoftMP)
#
#
################################################################################
class AtlasFeatureExtraction(CAccMatrixEx):
  def __init__(self, vgg16_weights_file =None,\
               ckpt_file=None,\
               pca_file=None,\
               HDC_file=None,\
               cuda_acc_file = '', \
               enable_atten = True,\
               ):
    # initialize the cuda acc lib
    CAccMatrixEx.__init__(self, acc_file=cuda_acc_file)    
    self.MatAcc_Init_cuVIPRMatAcc() 
    
    # Load VGG16 weights
    self.data_dict = np.load(vgg16_weights_file, encoding='latin1', allow_pickle=True).item()
    self.VGG_MEAN = [103.939, 116.779, 123.68]
    
    self.enable_atten = enable_atten
    
    self.vgg16_weights_list = []
    self.ProposalNet_weights_list = []
    self.AttNet_weights_list = []
    
    ####### create weights and bias for the model #######
    with tf.device('/gpu:0'):
      # vgg16 weights
      self.kernel_conv1_1 = self.get_conv_filter("conv1_1")
      self.bias_conv1_1 = self.get_bias("conv1_1")
      self.vgg16_weights_list.append(self.kernel_conv1_1)
      self.vgg16_weights_list.append(self.bias_conv1_1)
  
      self.kernel_conv1_2 = self.get_conv_filter("conv1_2")
      self.bias_conv1_2 = self.get_bias("conv1_2")
      self.vgg16_weights_list.append(self.kernel_conv1_2)
      self.vgg16_weights_list.append(self.bias_conv1_2)     
          # conv2
      self.kernel_conv2_1 = self.get_conv_filter("conv2_1")
      self.bias_conv2_1 = self.get_bias("conv2_1")
      self.vgg16_weights_list.append(self.kernel_conv2_1)
      self.vgg16_weights_list.append(self.bias_conv2_1)      

      self.kernel_conv2_2 = self.get_conv_filter("conv2_2")
      self.bias_conv2_2 = self.get_bias("conv2_2")
      self.vgg16_weights_list.append(self.kernel_conv2_2)
      self.vgg16_weights_list.append(self.bias_conv2_2)
          # conv3
      self.kernel_conv3_1 = self.get_conv_filter("conv3_1")
      self.bias_conv3_1 = self.get_bias("conv3_1")
      self.vgg16_weights_list.append(self.kernel_conv3_1)
      self.vgg16_weights_list.append(self.bias_conv3_1)
      
      self.kernel_conv3_2 = self.get_conv_filter("conv3_2")
      self.bias_conv3_2 = self.get_bias("conv3_2")
      self.vgg16_weights_list.append(self.kernel_conv3_2)
      self.vgg16_weights_list.append(self.bias_conv3_2)
      
      self.kernel_conv3_3 = self.get_conv_filter("conv3_3")
      self.bias_conv3_3 = self.get_bias("conv3_3")
      self.vgg16_weights_list.append(self.kernel_conv3_3)
      self.vgg16_weights_list.append(self.bias_conv3_3)
        # conv4
      self.kernel_conv4_1 = self.get_conv_filter("conv4_1")
      self.bias_conv4_1 = self.get_bias("conv4_1")
      self.vgg16_weights_list.append(self.kernel_conv4_1)
      self.vgg16_weights_list.append(self.bias_conv4_1)

      self.kernel_conv4_2 = self.get_conv_filter("conv4_2")
      self.bias_conv4_2 = self.get_bias("conv4_2")
      self.vgg16_weights_list.append(self.kernel_conv4_2)
      self.vgg16_weights_list.append(self.bias_conv4_2)

      self.kernel_conv4_3 = self.get_conv_filter("conv4_3")
      self.bias_conv4_3 = self.get_bias("conv4_3")    
      self.vgg16_weights_list.append(self.kernel_conv4_3)
      self.vgg16_weights_list.append(self.bias_conv4_3)
    
      # ProposalNet   
      self.conv0_w = self.weight_variable_n([3,3,512,1024], 0.1, name='conv0w')  #
      self.conv0_b = self.bias_variable_n  ([1024],name='conv0b')
      self.ProposalNet_weights_list.append(self.conv0_w)
      self.ProposalNet_weights_list.append(self.conv0_b)
      
      self.conv1_w = self.weight_variable_n([3,3,1024,1024], 0.1, name='conv1w')  #
      self.conv1_b = self.bias_variable_n  ([1024],name='conv1b')
      self.ProposalNet_weights_list.append(self.conv1_w)
      self.ProposalNet_weights_list.append(self.conv1_b)
      
      self.conv2_w = self.weight_variable_n([3,3,1024,1024], 0.1, name='conv2w')  #
      self.conv2_b = self.bias_variable_n  ([1024],name='conv2b')
      self.ProposalNet_weights_list.append(self.conv2_w)
      self.ProposalNet_weights_list.append(self.conv2_b)
      
      self.conv3_w = self.weight_variable_n([1,1,1024,512], 0.1, name='conv3w')  #
      self.conv3_b = self.bias_variable_n  ([512],name='conv3b')
      self.ProposalNet_weights_list.append(self.conv3_w)
      self.ProposalNet_weights_list.append(self.conv3_b)
      
      if self.enable_atten == True:
        # AttentionBlock
        self.atten0_w = self.weight_variable_n([1,1,512,1024], 0.1, name='atten0w')  #
        self.atten0_b = self.bias_variable_n  ([1024],name='atten0b')
        self.AttNet_weights_list.append(self.atten0_w)  
        self.AttNet_weights_list.append(self.atten0_b) 
      
        self.atten1_w = self.weight_variable_n([1,1,1024,1], 0.1, name='atten1w')  #
        self.atten1_b = self.bias_variable_n  ([1],name='atten1b')
        self.AttNet_weights_list.append(self.atten1_w)  
        self.AttNet_weights_list.append(self.atten1_b)   
    
    # load weights for ProposalNet and AttNet    
    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.Saver(self.ProposalNet_weights_list+self.AttNet_weights_list) 
    saver.restore(sess, ckpt_file)
    sess.close()
    
    # load PCA matrix
    self.pca_mean      = np.load(pca_file+"-mean.npy")
    self.pca_component = np.load(pca_file+"-component.npy")
    self.pca_component = self.pca_component.T
    
              
  #### Forward pass of the ATLAS pipeline ####     
  # This function takes the raw image batch array as input, and output the dense feature map from ProposalNet and indication map from the AttNet.
  # img: In shape [ImgNum, ImgH, ImgW, 3]
  #
  # return: fm:    in shape [ImgNum, H,W,C]. The dense feature maps from the ProposalNet
  #         atten: in shape [ImgNum, H,W,1]. The indication map from the AttNet  
  def model(self, img):
    bgr = self.rgb2bgr(img)
    #print ("bgr value:", bgr[0,0:5,0:5,0])
    with tf.device('/gpu:0'):
      ##### VGG16 #####
      bgr = tf.constant(bgr)
      conv1_1 = tf.nn.conv2d(bgr, self.kernel_conv1_1, [1, 1, 1, 1], padding='SAME')
      #print ("pool1 value:", conv1_1[0,0:10,0:10,0])
      #print ("conv1_1 weights value:", self.kernel_conv1_1[0,:,:,0])
      conv1_1 = tf.nn.relu(tf.nn.bias_add(conv1_1, self.bias_conv1_1))
      conv1_2 = tf.nn.conv2d(conv1_1, self.kernel_conv1_2, [1, 1, 1, 1], padding='SAME')
      conv1_2 = tf.nn.relu(tf.nn.bias_add(conv1_2, self.bias_conv1_2))
      pool1 = self.max_pool(conv1_2, 'pool1')

      conv2_1 = tf.nn.conv2d(pool1,   self.kernel_conv2_1, [1, 1, 1, 1], padding='SAME')
      conv2_1 = tf.nn.relu(tf.nn.bias_add(conv2_1, self.bias_conv2_1))
      conv2_2 = tf.nn.conv2d(conv2_1, self.kernel_conv2_2, [1, 1, 1, 1], padding='SAME')
      conv2_2 = tf.nn.relu(tf.nn.bias_add(conv2_2, self.bias_conv2_2))
      pool2 = self.max_pool(conv2_2, 'pool2')

      conv3_1 = tf.nn.conv2d(pool2,   self.kernel_conv3_1, [1, 1, 1, 1], padding='SAME')
      conv3_1 = tf.nn.relu(tf.nn.bias_add(conv3_1, self.bias_conv3_1))
      conv3_2 = tf.nn.conv2d(conv3_1, self.kernel_conv3_2, [1, 1, 1, 1], padding='SAME')
      conv3_2 = tf.nn.relu(tf.nn.bias_add(conv3_2, self.bias_conv3_2))
      conv3_3 = tf.nn.conv2d(conv3_2, self.kernel_conv3_3, [1, 1, 1, 1], padding='SAME')
      conv3_3 = tf.nn.relu(tf.nn.bias_add(conv3_3, self.bias_conv3_3))
      pool3 = self.max_pool(conv3_3, 'pool3')

      conv4_1 = tf.nn.conv2d(pool3,   self.kernel_conv4_1, [1, 1, 1, 1], padding='SAME')
      conv4_1 = tf.nn.relu(tf.nn.bias_add(conv4_1, self.bias_conv4_1))
      conv4_2 = tf.nn.conv2d(conv4_1, self.kernel_conv4_2, [1, 1, 1, 1], padding='SAME')
      conv4_2 = tf.nn.relu(tf.nn.bias_add(conv4_2, self.bias_conv4_2))
      conv4_3 = tf.nn.conv2d(conv4_2, self.kernel_conv4_3, [1, 1, 1, 1], padding='SAME')
      conv4_3 = tf.nn.relu(tf.nn.bias_add(conv4_3, self.bias_conv4_3))
      pool4 = self.max_pool(conv4_3, 'pool4')

      #print ("pool4 value:", pool4[0,0:10,0:10,0])
      ##### ProposalNet #####
      mean, variant = tf.nn.moments(pool4, [0,1,2])
      fm_in = tf.nn.batch_normalization(pool4, mean, variant, 0,  1., variance_epsilon=0.001)    

      fm0 = tf.nn.conv2d(fm_in, self.conv0_w, strides=[1,1,1,1], padding = 'SAME') +self.conv0_b
      fm0 = self.lrelu(fm0)
    
      fm1 = tf.nn.conv2d(fm0, self.conv1_w, strides=[1,1,1,1], padding = 'SAME') +self.conv1_b
      fm1 = self.lrelu(fm1)
    
      fm2 = tf.nn.conv2d(fm1, self.conv2_w, strides=[1,1,1,1], padding = 'SAME') +self.conv2_b
      fm2 = self.lrelu(fm2)
      mean, variant = tf.nn.moments(fm2, [0,1,2])
      fm2 = tf.nn.batch_normalization(fm2, mean, variant, 0,  1., variance_epsilon=0.001)    
    
      fm3 = tf.nn.conv2d(fm2, self.conv3_w, strides=[1,1,1,1], padding = 'SAME') +self.conv3_b
      fm3 = self.lrelu(fm3)
    
      # normalize the output feature map
      mean, variant = tf.nn.moments(fm3, [0,1,2])
      fm3 = tf.nn.batch_normalization(fm3, mean, variant, 0,  1., variance_epsilon=0.001)  
      
      #print ("ProposalNet output value:", fm3[0,0:5,0:5,0])
      # AttNet
      if self.enable_atten == True:
        # attention network  
        atten_fm = tf.nn.conv2d(fm3, self.atten0_w, strides=[1,1,1,1], padding = 'SAME') +self.atten0_b
        atten_fm = self.lrelu(atten_fm)

        atten_fm = tf.nn.conv2d(atten_fm, self.atten1_w, strides=[1,1,1,1], padding = 'SAME') +self.atten1_b
        atten_mask = self.lrelu(atten_fm)
        
        #print ("attenmap value:", atten_mask[0,0:5,0:5,0])
      else:
        atten_mask = 0        
       
    return  fm3, atten_mask

  # This function takes the image file list as input and output the dense feature map from ProposalNet and indication map from the AttNet.
  # img_dir="":        The root directory of the image files.
  # img_file_list=[]:  String list holds the image file names.
  # rescale=1.0:       Image rescale factor.
  # local_feature_datatype = 'FP32':  Datatype of the local feature. Can be one of ['FP32', 'FP8E4M3', 'INT8']
  # inference_batch=4: The inference will be split into batches to fit the GPU memory.
  # normalize_loc=True: Indicate if normalize the local feature coordinate to range [0,100].
  # find_lpg_closest_neighbor = False:   Inidcate if find the local feature closest neighbor for the LPG algorithm.
  # closest_neighbor_win_width_x=60:  Window size for x of Closest neighbor search for LPG algorithm. This param should be in range [0,100]
  # closest_neighbor_win_width_y=60:  Window size for y of Closest neighbor search for LPG algorithm. This param should be in range [0,100]
  # return: fm:    in shape [ImgNum, H,W,C]. The dense feature maps from the ProposalNet
  #         atten: in shape [ImgNum, H,W,1]. The indication map from the AttNet  
  def ExtractImgLocalFeature(self, img_dir="", img_file_list=[], rescale=1.0, \
                             local_feature_datatype='FP32',\
                             inference_batch=1, \
                             nms_interp_enable=True, \
                             normalize_loc=True, \
                             find_lpg_closest_neighbor = False,\
                             closest_neighbor_win_width_x=60, closest_neighbor_win_width_y=60):
    imgs = self.ReadBatchImagesFromFile(img_dir=img_dir, img_file_list=img_file_list, rescale=rescale)
    
    img_num = imgs.shape[0]
    fm = []
    atten = []
    iteration = int(img_num/inference_batch)
    if img_num%inference_batch !=0:
      iteration += 1
    for i in range(0, iteration):
      print ("Extract dense feature for frame %d to %d"%(i*inference_batch, (i+1)*inference_batch))
      if i== iteration-1:
        batch_imgs = imgs[i*inference_batch:, :,:,:]
      else:
        batch_imgs = imgs[i*inference_batch:(i+1)*inference_batch, :,:,:]
      tf_fm, tf_atten = self.model(batch_imgs)  
      fm.append(tf_fm.numpy())
      atten.append(tf_atten.numpy())
    fm = np.concatenate(fm, axis = 0)
    atten = np.concatenate(atten, axis = 0)
    

    batch_img_kp_desc = []
    batch_img_kp_coor = []
    batch_img_kp_norm_coor = []
    batch_img_neighbor_list = []  
    batch_img_neighbor_delimiter_list = []
    img_num = fm.shape[0]
    for i in range(0, img_num): # extract local feature for each image
      img_fm    = fm[i:i+1,:,:,:] # get the feature map of an image
      img_atten = atten[i,:,:,0]
      
      img_fm_h = img_fm.shape[1]
      img_fm_w = img_fm.shape[2]
      
      # extract keypoint coordinate
      img_kp_coor, kp_score = self.NMS(img_atten+100.0, 1, interp_enable=nms_interp_enable)
      img_kp_coor = img_kp_coor.astype(np.float16)
      batch_img_kp_coor.append(img_kp_coor)
      loc = np.zeros(img_kp_coor.shape)
      if normalize_loc==True:
        # normalize local feature coordinate to range [0,100]
        scale_factor_y = 100.0/img_fm_h
        scale_factor_x = 100.0/img_fm_w
        loc[:,0] = img_kp_coor[:,0]*scale_factor_y
        loc[:,1] = img_kp_coor[:,1]*scale_factor_x
        loc = loc+0.5
        loc = loc.astype(np.int32)
        batch_img_kp_norm_coor.append(loc)
      # find the closest neighbor
      if find_lpg_closest_neighbor==True:  
        frame_neighbor_mat, kp_neighbor_num = self.FindEveryKPClosetNeighborInMat(closest_neighbor_win_width_x, closest_neighbor_win_width_y, loc, [])
        frame_neighbor_mat = frame_neighbor_mat.astype(np.int32)
        kp_neighbor_num    = kp_neighbor_num.astype(np.int32)
        batch_img_neighbor_list.append(frame_neighbor_mat) 
        batch_img_neighbor_delimiter_list.append(kp_neighbor_num)
      # extract local descriptor
      img_desc = self.GetKeyPointDescriptor(img_fm, img_kp_coor,3)
      img_desc= img_desc-np.mean(img_desc)
      img_desc = img_desc/np.std(img_desc)
      img_desc = np.dot(img_desc-self.pca_mean, self.pca_component)
      img_desc = img_desc.astype(np.float32)
      if local_feature_datatype == 'FP32': 
        img_desc = self.NormalizeFeatures(img_desc)
      elif local_feature_datatype == 'FP8_E4M3':
        img_desc = self.CVT_ImageLocalFeatureDataType(img_desc, 'FP8', 'E4M3') 
      elif local_feature_datatype == 'INT8': # support in the future. default to FP32
        img_desc = self.NormalizeFeatures(img_desc)
        
        img_desc = self.CVT_ImageLocalFeatureDataType(img_desc, 'INT8', 'SYMMETRY', img_desc.max(), img_desc.min()) 
      else:
        print("############Unsupported datatypes for Database local feature extraction!############")
      batch_img_kp_desc.append(img_desc)
    return np.array(batch_img_kp_coor, dtype=object), \
           np.array(batch_img_kp_desc, dtype=object), \
           np.array(batch_img_kp_norm_coor, dtype=object), \
           np.array(batch_img_neighbor_list, dtype=object), \
           np.array(batch_img_neighbor_delimiter_list, dtype=object)
           





  # return bounding box center coordinate in fm.
  # indi_map:           in shape [H, W]. Indication map.
  # det_win_radious:    The NMS operation window radious. When det_win_radious=1,  NMS window width equals 3.
  # interp_enable=True: Indicate if interpolation is enabled for sub-pixel maxima detection. 
  
  # np array bbx_center: bounding box center in float point. In shape [KpNum,2] with format [[y0,x0], [y1,x1]...]
  # np_array center_value: the center bin value in the sumed feature map for each bbx  
  def NMS(self, indi_map, det_win_radious, interp_enable=True):
    detect_window_radious = det_win_radious  #3x3 detection window
    dt_window_size = 1+detect_window_radious*2
    h = indi_map.shape[0]
    w = indi_map.shape[1]
    
    #normalize
    max_value = np.amax(indi_map)
    fm2 = indi_map[:]
    fm2 = indi_map/max_value
    
    bbx_center = []
    center_value = []
    for y in range(detect_window_radious, h-detect_window_radious):
        for x in range(detect_window_radious, w-detect_window_radious):
            patch = fm2[y-detect_window_radious:y+detect_window_radious+1,  x-detect_window_radious:x+detect_window_radious+1]
            is_local_max, max_x, max_y = self.__IsCentreMax(patch, detect_window_radious, 0.0, interp_enable=interp_enable)
            
            if is_local_max==True:
                bbx_center.append([y-det_win_radious+max_y,x-det_win_radious+max_x])
                center_value.append(indi_map[y,x])
    return np.array(bbx_center), np.array(center_value)


# detect bounding box based on the non-maximum suppression
# fm:           feature map np array [H,W,1]
# det_win_radious:  detection bounding box radious 

  def __loc_interpolation(self, patch, x,y):
    hori_line_v = patch[y, x-1:x+2]
    vert_line_v = patch[y-1:y+2, x]
    
    hori_d0 = hori_line_v[1]-hori_line_v[0]
    hori_d1 = hori_line_v[2]-hori_line_v[1]
    
    vert_d0 = vert_line_v[1] - vert_line_v[0]
    vert_d1 = vert_line_v[2] - vert_line_v[1]
    
    xf = x-0.5 + hori_d0/(hori_d0 - hori_d1)
    yf = y-0.5 + vert_d0/(vert_d0 - vert_d1)
    
    return xf, yf

  def __IsCentreMax(self, patch, detect_window_radious, threshold, interp_enable=True):
    window_size = int(detect_window_radious*2+1)
    max_idx = np.argmax(patch)
    max_value = np.amax(patch)
    x = max_idx%window_size
    y = int(max_idx/window_size)
    
    if x!=0 and x!=window_size-1 and y!=0 and y!=window_size-1 and max_value>threshold:
        if interp_enable==True:
          xf,yf = self.__loc_interpolation(patch, x,y)
        else:
          xf = x
          yf = y
        return True, xf,yf
    return False, 0,0




  # return a dict. with given X coordinate as key. Return the location index in loc_mat_a
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


  #
  def GetKeyPointDescriptor(self, fm, keypoint, window_radious):
    desc = []
    if keypoint.shape[0] ==0:
        print ("Function GetKeyPointDescriptor: input keypoint is empty")
    
    for i in range(0,keypoint.shape[0]):
        y = int(keypoint[i][0]+0.5)
        x = int(keypoint[i][1]+0.5)
        ##v = fm[0,y,x,:]
        v = []
        for y_p in range(y-window_radious, y+window_radious+1):
            for x_p in range(x-window_radious, x+window_radious+1):

                if y_p <0 or y_p>fm.shape[1]-1 or x_p<0 or x_p>fm.shape[2]-1:
                   v+= [0]*fm.shape[3]
                else:
                   v=v+fm[0,y_p,x_p,:].tolist()

        #v = np.concatenate((fm[0,y-1,x-1,:],fm[0,y-1,x,:],fm[0,y-1,x+1,:],fm[0,y,x-1,:],fm[0,y-1,x,:],fm[0,y-1,x+1,:],fm[0,y+1,x-1,:],fm[0,y+1,x,:],fm[0,y+1,x+1,:]),axis=0)
        
        desc.append(v)
    #print keypoint.shape, len(desc)
    return np.float32(np.array(desc))

  def ReadBatchImagesFromFile(self, img_dir="", img_file_list=[], rescale=1.0):  
    img_num = len(img_file_list)
    img_array = []
    for i in range(0,img_num):
      img = skimage.io.imread(img_dir+'/'+img_file_list[i])
      if rescale >=1.0005 or rescale <=0.9995:
        img = skimage.transform.rescale(img, rescale, anti_aliasing=True)  
        img = img*255.
      if i==0:
        img_array =np.zeros((img_num,img.shape[0],img.shape[1],img.shape[2]),dtype='float16')
      img_array[i,:,:,:] = img/255.0
      
    img_array = img_array.astype(np.float32)  
    return img_array
       
  def weight_variable_n(self,shape, std_value, name):
      with tf.device('/gpu:0'):
        initial = tf.random.normal(shape, dtype=tf.float32, stddev=std_value)
        return tf.Variable(initial, name= name)

  def bias_variable_n(self,shape, name):
      with tf.device('/gpu:0'):
        initial = tf.constant(0., shape=shape)
        return tf.Variable(initial, name= name)

  def lrelu(self,x, leak=0.2, lash_positive=0.0,lash_negative=0.0, name="lrelu"):
      f1 = 0.5 * (1 + leak)
      f2 = 0.5 * (1 - leak)
      a = lash_positive
      b = lash_negative    
      leak = leak/2.0
      return f1 * x + f2 * abs(x) +0.5*abs(x-a) - (0.5*abs(x)+a/2.0) +leak*abs(x)+leak*b -leak*abs(x+b) #+ 0.001*x       
        
  def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

  def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

  def get_conv_filter(self, name):
        with tf.device('/gpu:0'):
            return tf.Variable(self.data_dict[name][0], name=name+'w')

  def get_bias(self, name):
        with tf.device('/gpu:0'):
            return tf.Variable(self.data_dict[name][1], name=name+'b')
                    
  def rgb2bgr(self,rgb):
        rgb_scaled = rgb*255.0
        red, green, blue = np.split(rgb_scaled,3,axis=3)
        bgr = np.concatenate( [
            blue - self.VGG_MEAN[0],
            green - self.VGG_MEAN[1],
            red - self.VGG_MEAN[2],
        ],axis=3)
        return bgr       
        
  def GenerateDBFileNames(self, database_name, kp_desc_dtype, enable_lpg_extraction, lpg_window_x, lpg_window_y):
      DB_feature_file_template = database_name+'-'+kp_desc_dtype
      if enable_lpg_extraction==True:
        DB_feature_file_template += '-LPG-'+'X'+str(lpg_window_x)+'-'+'Y'+str(lpg_window_y)
       
      return DB_feature_file_template +'-%s-%08d.npy', DB_feature_file_template +'-%s-*.npy'
            
