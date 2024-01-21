from Acc_HirATLAS import *  
import numpy as np
import scipy.io
import time
import gc
use_DELF = False  # if load DELF local feature file set this to True

class HirATLAS(CAccFeatureMatch, CAccMatrixEx):
  # DB_xxxx_files are list with string of the absolute file path
  def __init__(self, DB_LocFeature_files=[], DB_LocFeature_loc_files=[], DB_loc_H=100, DB_loc_W=100, \
               DB_HolFeature_files=[], \
               acc_file='/home/fangming/svn_test/SVN_repository/PHD/RTX3090/lib/FeatureMatchAcc.so', \
               cuda_acc_file = '/home/fangming/svn_test/SVN_repository/PHD/RTX3090/lib/cudaPrj/cuVIPRACCLib/Debug/libcuVIPRACCLib', \
               DB_kp_spatial_closest_neighbor_files=[], DB_kp_spatial_closest_neighbor_delimiter_files=[], \
               gauss2d_y_size = 150, gauss2d_x_size = 150,\
               gauss_patch_x_size = 410, gauss_patch_y_size = 410,\
               gauss_sigma=1.0,\
               enable_topK=True,\
               DB_feature_gpu_preload = False,\
               Feature_dtype="FP32",\
               DB_feature_normalize = True):#indicate if normalize the DB feature during preload to GPU memory 
    #super(ImgSimilarityRANSAC, self).__init__(acc_file) # initialize CAccFeatureMatch class
    #super(ImgSimilarityRANSAC, self).__init__() # initialize CAccMatrix class
    CAccFeatureMatch.__init__(self, acc_file)
    CAccMatrixEx.__init__(self, acc_file=cuda_acc_file)
    
    self.feature_storage_type, self.feature_dtype_digit = self.Dtype2Digit(Feature_dtype)
    self.MatAcc_InitMatAccExEnv(self.feature_dtype_digit)
    self.MatAcc_Init_cuVIPRMatAcc()     # Initialize the legacy VPRMatAcc because need to use the NormalizeBatchVectorFP32 function 
    #super().__init__(acc_file)
    self.DB_feature_gpu_preload = DB_feature_gpu_preload
    self.enable_topK = enable_topK
    
    #if DB_kp_spatial_closest_neighbor_files==[]:
    #  DB_kp_spatial_closest_neighbor_files=None
    #if DB_kp_spatial_closest_neighbor_delimiter_files==[]:
    #  DB_kp_spatial_closest_neighbor_delimiter_files = None
    #if DB_HolFeature_files==[]:
    #  DB_HolFeature_files = None
      
    # db feature caches
    self.db_LocFeature_cache = []            # in shape [place_N] type Object. each object is np array has shape [LocFeature_N, 1024] 
    self.db_LocFeature_loc_cache = []        # in shape [place_N] type Object. each object is np array has shape [LocFeature_N, 2] 
    self.db_frame_loc_feature_delimiter= []  # in shape [place_N] type int. each holds the number of local features in each db frame
    self.db_frame_loc_feature_start_idx = [] # in shape [place_N] type int. each holds the local features start idx in each db frame
    self.db_HolFeature_cache = []            # in shape [place_N, 4096]
    
    self.db_kp_spatial_closest_neighbor_cache = []            # in shape [place_N] type Object. each object is np array has shape [LocFeature_N, 3] 
    self.db_kp_spatial_closest_neighbor_delimiter_cache = []  # in shape [place_N] type Object. each object is np array has shape [LocFeature_N]
    self.db_kp_spatial_closest_neighbor_frame_start_idx = []  # in shape [place_N] type int. each is the index of db_kp_spatial_closest_neighbor_cache of closest neighbor for each database frame
    
    self.db_HolFeature_mean = []
    
    self.db_enabled_place = []    # in shape [enabled_place_N] type int. each indicate the place id in the database.
    

    
    self.gaussian_patch = self.Generate2DGaussian(gauss2d_y_size,gauss2d_x_size,gauss_patch_y_size, gauss_patch_x_size, gauss_sigma)
    self.gaussian_patch, self.gaussian_h, self.gaussian_w = self.Array2Dto1D(self.gaussian_patch, dtype=np.float32)
   
    self.db_place_number = 0
    
    self.DB_feature_normalize = DB_feature_normalize
   
    # load db features
    if len(DB_LocFeature_files)!=0 and len(DB_LocFeature_loc_files)!=0:
      # Probe the database feature file to get the local feature dimension, total local feature in database, frame local feature delimiter 
      db_feature_dim, db_totally_feature_num, self.db_frame_loc_feature_delimiter = self.__ProbeDBFeatureScale( DB_LocFeature_files)
      self.db_place_number = self.db_frame_loc_feature_delimiter.shape[0]
     
      self.db_frame_loc_feature_start_idx = np.zeros((self.db_place_number), dtype=np.int32)
      for i in range(0,self.db_place_number):
        self.db_frame_loc_feature_start_idx[i] = np.sum(self.db_frame_loc_feature_delimiter[0:i])
      
      self.db_frame_loc_feature_delimiter = np.ascontiguousarray(self.db_frame_loc_feature_delimiter, dtype=np.int32)  
      self.db_frame_loc_feature_start_idx = np.ascontiguousarray(self.db_frame_loc_feature_start_idx, dtype=np.int32)  
      self.db_LocFeature_loc_cache = self.__LoadDB_loc_cache(DB_LocFeature_loc_files)
      self.AllocateDB_GPU_Memory(db_totally_feature_num, db_feature_dim, self.db_place_number)
      self.SetDBMatInfo(self.db_frame_loc_feature_delimiter , self.db_frame_loc_feature_start_idx)
      self.__PreLoadDBFeature2GPU(DB_LocFeature_files)
      
      self.db_LocFeature_loc_cache, a,b = self.Array2Dto1D(np.concatenate(self.db_LocFeature_loc_cache), dtype=np.int32)
      self.db_enabled_place = np.array(range(0,self.db_place_number),dtype=np.int32)  
      # db parameter
     
      if len(DB_kp_spatial_closest_neighbor_files)!=0 and len(DB_kp_spatial_closest_neighbor_delimiter_files)!=0:
        print("Now load spatial closest neighbor")
        self.db_kp_spatial_closest_neighbor_cache, self.db_kp_spatial_closest_neighbor_delimiter_cache = self.__LoadDB_cache(DB_kp_spatial_closest_neighbor_files, DB_kp_spatial_closest_neighbor_delimiter_files, True)
        _, _, self.db_kp_spatial_closest_neighbor_frame_start_idx = self.__ExtractCopyTopNDB_Kp_ClosestNeighbor_Port(self.db_enabled_place)
        
        self.db_kp_spatial_closest_neighbor_cache            = np.concatenate(self.db_kp_spatial_closest_neighbor_cache)
        self.db_kp_spatial_closest_neighbor_cache,a,b        = self.Array2Dto1D(self.db_kp_spatial_closest_neighbor_cache, dtype=np.int32)
        
        self.db_kp_spatial_closest_neighbor_delimiter_cache  = np.concatenate(self.db_kp_spatial_closest_neighbor_delimiter_cache)
        self.db_kp_spatial_closest_neighbor_delimiter_cache  = np.ascontiguousarray(self.db_kp_spatial_closest_neighbor_delimiter_cache, dtype=np.int32)
        #if self.enable_topK==False: # if the topK retrieval is disabled
          #self.AllDB_KP_neighbor, self.AllDB_KP_in_frame_neighbor_delimiter, self.AllDB_KP_neighbor_start_idx = self.__ExtractCopyTopNDB_Kp_ClosestNeighbor_Port(top_N_idx_vect)
        
        self.db_kp_spatial_closest_neighbor_frame_start_idx = np.ascontiguousarray(self.db_kp_spatial_closest_neighbor_frame_start_idx, dtype=np.int32)   
    

          
    if len(DB_HolFeature_files)!=0:
      self.__LoadDB_cache_HolFeature(DB_HolFeature_files)
      if self.db_place_number != self.db_HolFeature_cache.shape[0]: 
        self.db_place_number = self.db_HolFeature_cache.shape[0]  # in this case the no local feature are available, so use holistic feature to determine the db place number
        #print ("Holistic feature number is not correspond to the number of place in the database!!")
        #print ("Should=%d, but=%d"%(self.db_place_number, self.db_HolFeature_cache.shape[0]))
        
      # normalize database holistic feature
      self.db_HolFeature_cache = self.NormalizeFeatures(self.db_HolFeature_cache)  # use the Legacy VPRMatAcc library for this operation
      # set the database holistic features in the GPU
      self.SetDBHolMat(self.db_HolFeature_cache) # set the normalized hol matrix to the VPRMatAccExEnv
      
      
    self.DB_loc_H = DB_loc_H
    self.DB_loc_W = DB_loc_W
    

 

  def __NormalizeHolFeature(self):
    self.db_HolFeature_mean = np.mean(self.db_HolFeature_cache, axis=0)
    #self.db_HolFeature_cache = self.db_HolFeature_cache-self.db_HolFeature_mean[None,:]
    norm_vect = np.linalg.norm(self.db_HolFeature_cache, axis=1)  
    self.db_HolFeature_cache = (self.db_HolFeature_cache.T/norm_vect).T
    
  def __NormalizeQueryHolFeature(self, hol_feature):
    if self.db_HolFeature_cache == []:
      return 0
    #normalized_hol_feature = hol_feature-self.db_HolFeature_mean
    normalized_hol_feature = hol_feature
    normalized_hol_feature = normalized_hol_feature.T/np.linalg.norm(normalized_hol_feature)
    return normalized_hol_feature.T
   
  def __LoadDB_cache2(self, DB_LocFeature_files, DB_LocFeature_loc_files):
    # load all the feature files
    print("aaaaaa")
    feature_array_list = []
    feature_loc_array_list = []
    for i in range(0,len(DB_LocFeature_files)):
      print (DB_LocFeature_files[i])
      feature = np.load(DB_LocFeature_files[i], allow_pickle=True, encoding="latin1")
      feature_loc = np.load(DB_LocFeature_loc_files[i], allow_pickle=True, encoding="latin1")
      ## in case feature.shape == [1,X,X], make it [X,X]
      if feature.shape[0]==1:
         feature = feature[0]
         feature_loc = feature_loc[0]#np.array([feature_loc], dtype=np.object)
         feature_array_list.append(feature)
         feature_loc_array_list.append(feature_loc)
      else:
         feature_array_list += feature.tolist()
         feature_loc_array_list += feature_loc.tolist()
      print (feature.shape, feature.dtype) 
    return np.array(feature_array_list, dtype=object), np.array(feature_loc_array_list, dtype=object)
    
  def __LoadDB_cache_Kp_Spatial_Closest_Neighbor_Port(self, DB_kp_spatial_closest_neighbor_files, DB_kp_spatial_closest_neighbor_delimiter_files):
    # load all the feature files
    feature_array_list = []
    feature_loc_array_list = []
    for i in range(0,len(DB_kp_spatial_closest_neighbor_files)):
      #print (DB_kp_spatial_closest_neighbor_files[i])
      # feature in shape [PlaceNum] object.Each in shape[frame_KpNum, 3]
      feature = np.load(DB_kp_spatial_closest_neighbor_files[i], allow_pickle=True, encoding="latin1")
      # feature_loc in shape [PlaceNum] object.Each in shape[frame_KpNum, 2]
      feature_loc = np.load(DB_kp_spatial_closest_neighbor_delimiter_files[i], allow_pickle=True, encoding="latin1")
      ## in case feature.shape == [1,X,X], make it [X,X]
      if feature.shape[0]==1:
         feature = feature[0]
         feature_loc = feature_loc[0]#np.array([feature_loc], dtype=np.object)
         feature_array_list.append(feature)
         feature_loc_array_list.append(feature_loc)         
      feature_array_list.append(feature)
      feature_loc_array_list.append(feature_loc)
      #print (feature.shape, feature.dtype) 
    self.db_kp_spatial_closest_neighbor_cache           = np.concatenate(feature_array_list, axis = 0)
    self.db_kp_spatial_closest_neighbor_delimiter_cache = np.concatenate(feature_loc_array_list, axis = 0)     
   
#  def __LoadDB_cache_TopK_disabled(self, DB_LocFeature_files, DB_LocFeature_loc_files):
    
  # probe the totall number of local features in db and the dimension of local feature  
  # probe number of local feature in each frame in db and output as db_frame_loc_feature_delimiter
  def __ProbeDBFeatureScale(self, DB_LocFeature_files):
    if use_DELF == False:
      db_feature_dim = 0   # feature dimension
      db_totally_frame_feature_num   = 0   # total number of features
      db_frame_loc_feature_delimiter = []
      for i in range(0,len(DB_LocFeature_files)):
        # feature in shape [PlaceNum] object.Each in shape[frame_KpNum, Feature_dim]
        feature = np.load(DB_LocFeature_files[i], allow_pickle=True, encoding="latin1")
        print(feature.shape)
        for j in range(0,feature.shape[0]):
          db_feature_dim                   = feature[j].shape[1]
          frame_feature_num                = feature[j].shape[0]
          db_totally_frame_feature_num    += frame_feature_num
       
          db_frame_loc_feature_delimiter.append(frame_feature_num)
    else:
      db_feature_dim = 0   # feature dimension
      db_totally_frame_feature_num   = 0   # total number of features
      db_frame_loc_feature_delimiter = []
      for i in range(0,len(DB_LocFeature_files)):
        # feature in shape [PlaceNum] object.Each in shape[frame_KpNum, Feature_dim]
        feature = np.load(DB_LocFeature_files[i], allow_pickle=True, encoding="latin1")
        for j in range(0,feature.shape[0]):
          db_feature_dim                   = feature.shape[2]
          frame_feature_num                = feature.shape[1]
          db_totally_frame_feature_num    += frame_feature_num
          db_frame_loc_feature_delimiter.append(frame_feature_num)    
    return db_feature_dim, db_totally_frame_feature_num, np.array(db_frame_loc_feature_delimiter)
  
  def __PreLoadDBFeature2GPU(self, DB_LocFeature_files):
    if use_DELF == False:
      for i in range(0,len(DB_LocFeature_files)):
        # feature in shape [PlaceNum] object.Each in shape[frame_KpNum, Feature_dim]
        feature = np.load(DB_LocFeature_files[i], allow_pickle=True, encoding="latin1")
        batch_db_frame_num = feature.shape[0]
        batch_db_frame_kp_num_array = []
        for j in range(0, batch_db_frame_num):
          batch_db_frame_kp_num_array.append(feature[j].shape[0])
        batch_db_frame_kp_num_array = np.array(batch_db_frame_kp_num_array)
        feature = np.concatenate(feature)
        if self.DB_feature_normalize==True:
          feature = self.NormalizeFeatures(feature)
        print (feature.shape)
        self.AttachGPU_DB_feature(feature, batch_db_frame_num, batch_db_frame_kp_num_array)
    else:
      for i in range(0,len(DB_LocFeature_files)):
        # feature in shape [PlaceNum] object.Each in shape[frame_KpNum, Feature_dim]
        feature = np.load(DB_LocFeature_files[i], allow_pickle=True, encoding="latin1")
        batch_db_frame_num = feature.shape[0]
        batch_db_frame_kp_num_array = []
        for j in range(0, batch_db_frame_num):
          batch_db_frame_kp_num_array.append(feature.shape[1])
        batch_db_frame_kp_num_array = np.array(batch_db_frame_kp_num_array)
        feature = np.concatenate(feature)
        if self.DB_feature_normalize==True:
          feature = self.NormalizeFeatures(feature)
        self.AttachGPU_DB_feature(feature, batch_db_frame_num, batch_db_frame_kp_num_array)       
  
  # is_load_closest_neighbor: this is to indicate if the keypoint closest neighbor file is loaded (this is to let the code to compatiable to load DELF feature)
  def __LoadDB_cache(self, DB_LocFeature_files, DB_LocFeature_loc_files, is_load_closest_neighbor=False):
   if use_DELF == True and is_load_closest_neighbor==False:
    # load all the feature files
    feature_array_list = []
    feature_loc_array_list = []
    for i in range(0,len(DB_LocFeature_files)):
      #print (DB_LocFeature_files[i])
      feature = np.load(DB_LocFeature_files[i], allow_pickle=True, encoding="latin1")
      feature_loc = np.load(DB_LocFeature_loc_files[i], allow_pickle=True, encoding="latin1")
      ## in case feature.shape == [1,X,X], make it [X,X]
      feature_array_list.append(feature)
      feature_loc_array_list.append(feature_loc)
      #print (feature.shape, feature.dtype) 
    return np.concatenate(feature_array_list, axis = 0), np.concatenate(feature_loc_array_list, axis = 0)   
   elif use_DELF == True and is_load_closest_neighbor==True:
   # load all the feature files
    feature_array_list = []
    feature_loc_array_list = []
    for i in range(0,len(DB_LocFeature_files)):
      #print (DB_LocFeature_files[i])
      feature = np.load(DB_LocFeature_files[i], allow_pickle=True, encoding="latin1")
      feature_loc = np.load(DB_LocFeature_loc_files[i], allow_pickle=True, encoding="latin1")
      ## in case feature.shape == [1,X,X], make it [X,X]
      if feature.shape[0]==1:
         feature = feature[0]
         feature_loc = feature_loc[0]#np.array([feature_loc], dtype=np.object)
         feature_array_list.append(feature)
         feature_loc_array_list.append(feature_loc)
      else:
         feature_array_list += feature.tolist()
         feature_loc_array_list += feature_loc.tolist()
      #print (feature.shape, feature.dtype) 
    return np.array(feature_array_list, dtype=object), np.array(feature_loc_array_list, dtype=object)
   elif use_DELF == False:
   # load all the feature files
    feature_array_list = []
    feature_loc_array_list = []
    for i in range(0,len(DB_LocFeature_files)):
      #print (DB_LocFeature_files[i])
      feature = np.load(DB_LocFeature_files[i], allow_pickle=True, encoding="latin1")
      feature_loc = np.load(DB_LocFeature_loc_files[i], allow_pickle=True, encoding="latin1")
      ## in case feature.shape == [1,X,X], make it [X,X]
      if feature.shape[0]==1:
         feature = feature[0]
         feature_loc = feature_loc[0]#np.array([feature_loc], dtype=np.object)
         feature_array_list.append(feature)
         feature_loc_array_list.append(feature_loc)
      else:
         feature_array_list += feature.tolist()
         feature_loc_array_list += feature_loc.tolist()
      #print (feature.shape, feature.dtype) 
    return np.array(feature_array_list, dtype=object), np.array(feature_loc_array_list, dtype=object)
   else:
    return [],[]
  def __LoadDB_loc_cache(self, DB_LocFeature_loc_files):
   if use_DELF == True:
    # load all the feature files
 
    feature_loc_array_list = []
    for i in range(0,len(DB_LocFeature_loc_files)):
      #print (DB_LocFeature_files[i])
      feature_loc = np.load(DB_LocFeature_loc_files[i], allow_pickle=True, encoding="latin1")
      ## in case feature.shape == [1,X,X], make it [X,X]
      feature_loc_array_list.append(feature_loc)
      #print (feature.shape, feature.dtype) 
    return np.concatenate(feature_loc_array_list, axis = 0)   
   elif use_DELF == True:
   # load all the feature files
    feature_loc_array_list = []
    for i in range(0,len(DB_LocFeature_loc_files)):
      #print (DB_LocFeature_files[i])
      feature_loc = np.load(DB_LocFeature_loc_files[i], allow_pickle=True, encoding="latin1")
      ## in case feature.shape == [1,X,X], make it [X,X]
      if feature_loc.shape[0]==1:
         feature_loc = feature_loc[0]#np.array([feature_loc], dtype=np.object)
         feature_loc_array_list.append(feature_loc)
      else:
         feature_loc_array_list += feature_loc.tolist()
      #print (feature.shape, feature.dtype) 
    return np.array(feature_loc_array_list, dtype=object)
   elif use_DELF == False:
   # load all the feature files
    feature_loc_array_list = []
    for i in range(0,len(DB_LocFeature_loc_files)):
      #print (DB_LocFeature_files[i])
      feature_loc = np.load(DB_LocFeature_loc_files[i], allow_pickle=True, encoding="latin1")
      ## in case feature.shape == [1,X,X], make it [X,X]
      if feature_loc.shape[0]==1:
         feature_loc = feature_loc[0]#np.array([feature_loc], dtype=np.object)
         feature_loc_array_list.append(feature_loc)
      else:
         feature_loc_array_list += feature_loc.tolist()
      #print (feature.shape, feature.dtype) 
    return np.array(feature_loc_array_list, dtype=object)
   else:
    return [],[]
  
  #def __LoadDB_cache_Kp_Spatial_Closest_Neighbor_Port(self, DB_kp_spatial_closest_neighbor_files, DB_kp_spatial_closest_neighbor_delimiter_files):
  #  # load all the feature files
  #  feature_array_list = []
  #  feature_loc_array_list = []
  #  for i in range(0,len(DB_kp_spatial_closest_neighbor_files)):
  #    print (DB_kp_spatial_closest_neighbor_files[i])
  #    # feature in shape [N] object.Each in shape[frame_KpNum, 3]
  #    feature = np.load(DB_kp_spatial_closest_neighbor_files[i], allow_pickle=True, encoding="latin1")
  #    # feature_loc in shape [N] object.Each in shape[frame_KpNum]
  #    feature_loc = np.load(DB_kp_spatial_closest_neighbor_delimiter_files[i], allow_pickle=True, encoding="latin1")
  #    ## in case feature.shape == [1,X,X], make it [X,X]
  #    feature_array_list.append(feature)
  #    feature_loc_array_list.append(feature_loc)
  #    print (feature.shape, feature.dtype) 
  #  self.db_kp_spatial_closest_neighbor_cache           = np.concatenate(feature_array_list, axis = 0)
  #  self.db_kp_spatial_closest_neighbor_delimiter_cache = np.concatenate(feature_loc_array_list, axis = 0)     
    
  def __LoadDB_cache_HolFeature(self, DB_HolFeature_files): 
    feature = self.Load_holistic_feature_file_Port(DB_HolFeature_files)
    self.db_HolFeature_cache = feature
    self.__NormalizeHolFeature()

  
  def Load_holistic_feature_file_Port(self, file_name):
    if  file_name[-4:]==".mat":
      hol_feature = scipy.io.loadmat(file_name)
      feature = hol_feature['Y']  
    elif file_name[-4:]==".npy":
      feature = np.load(file_name) 
    return feature
  
  # retrieve the top N most similarity entries in database
  # holistic_feature_query_img: the query image holistic feature. in shape in shape [K, hol_dim]
  # db_batch_size:  because the GPU memory size is limited so the database holistic feature is load in batches
  # retrieve_top_N: retrive the top N most similar entries. When ==-1 means retrieve all the candidates in database
  # return: if retrieve_top_N==-1    ---> Return the similarity matrix [K, retrieve_top_N] holds similarity of each query holistic feature to all the database holistic feature
  # return: if retrieve_top_N>=0     ---> Index array in shape [K, retrieve_top_N] that holds indecs of the top-(retrieve_top_N) candidate in the database
  def __HolFeatureRetrieve_Port(self, holistic_feature_query, db_batch_size = 512, retrieve_top_N=100, metric="cosine"):
#    if holistic_feature_query.ndim==1:
#      hol_feature = holistic_feature_query.reshape(1,holistic_feature_query.shape[0])
#      similarity = np.zeros((self.db_place_number), dtype=np.float)
#    else:
#      hol_feature = holistic_feature_query 
    
    similarity = self.HolQuery(holistic_feature_query)
    similarity =np.reshape(similarity, (holistic_feature_query.shape[0], self.db_place_number))
    if retrieve_top_N ==-1:
      return similarity
    else:
      return similarity[:].argsort()[:,-retrieve_top_N:] 



    steps = int(self.db_place_number / db_batch_size)
    batch_remine = int(self.db_place_number % db_batch_size)
    similarity = np.zeros((holistic_feature_query.shape[0], self.db_place_number), dtype=np.float32)
       
    for i in range(0, steps):
      db_batch = self.db_HolFeature_cache[i*db_batch_size: (i+1)*db_batch_size, :]
      if metric=="cosine":
        ## GPU cosine similarity calculation
        similarity[:, i*db_batch_size: (i+1)*db_batch_size] = self.cross_cosine_distance_ArrayVsArray_cpu(holistic_feature_query, db_batch)
      elif metric == "euclidean":
        similarity[:, i*db_batch_size: (i+1)*db_batch_size] = self.cross_cosine_distance_ArrayVsArray_cpu(holistic_feature_query, db_batch)
 
    db_batch = self.db_HolFeature_cache[steps*db_batch_size:, :]
    if metric=="cosine":
      similarity[:, steps*db_batch_size:]=self.cross_cosine_distance_ArrayVsArray_cpu(holistic_feature_query, db_batch)
    elif metric == "euclidean":
      similarity[:, steps*db_batch_size:]=self.cross_cosine_distance_ArrayVsArray_cpu(holistic_feature_query, db_batch)   
    if retrieve_top_N ==-1:
      return similarity
    else:
      return similarity[:].argsort()[:,-retrieve_top_N:] 
  # Calculate the similarity between query holistic feature to all the database holistic feature
  # holistic_feature_query_img: the query image holistic feature. in shape [4096]
  # db_batch_size:  number of holistic to calculate with the query feature in database. because the GPU memory size is limited so the similarity is calculated in batches
  # similarity: in shape [self.db_place_number] holds similarity of query to all the database image holistic features
  def __HolFeatureRetrieveAll_Port(self, holistic_feature_query_img, db_batch_size = 512):
    steps = int(self.db_place_number / db_batch_size)
    batch_remine = int(self.db_place_number % db_batch_size)
    similarity = np.zeros((self.db_place_number), dtype=np.float)
    for i in range(0, steps):
      db_batch = self.db_HolFeature_cache[i*db_batch_size: (i+1)*db_batch_size, :]
      ## GPU cosine similarity calculation
      similarity[i*db_batch_size: (i+1)*db_batch_size] = self.cross_cosine_distance_VectVsArray_cpu(holistic_feature_query_img, db_batch)
    db_batch = self.db_HolFeature_cache[steps*db_batch_size:, :]
    similarity[steps*db_batch_size:]=self.cross_cosine_distance_VectVsArray_cpu(holistic_feature_query_img, db_batch)
    return similarity

  def __NormalizePoint(self, p):
    p_num = p.shape[0]
    y_vect = p[:,0]
    x_vect = p[:,1]
    mean_y = np.mean(y_vect)
    mean_x = np.mean(x_vect)
    s = np.sum(np.square(y_vect-mean_y)+np.square(x_vect-mean_x))/(2*p_num)
    s = 1.0/np.sqrt(s)

    H = np.zeros((3,3), dtype=np.float)
    H[2,2] = 1.0
    H[0,0] = s
    H[0,2] = -mean_y*s
    H[1,1] = s
    H[1,2] = -mean_x*s
    
    p_norm = np.transpose(np.matmul(H, np.transpose(p)))
    return p_norm, H

  def __GetPointNormMat(self, p):
    p_num = p.shape[0]
    y_vect = p[:,0]
    x_vect = p[:,1]
    mean_y = np.mean(y_vect)
    mean_x = np.mean(x_vect)
    s = np.sum(np.square(y_vect-mean_y)+np.square(x_vect-mean_x))/(2*p_num)
    s = 1.0/np.sqrt(s)

    H = np.zeros((3,3), dtype=np.float)
    H[2,2] = 1.0
    H[0,0] = s
    H[0,2] = -mean_y*s
    H[1,1] = s
    H[1,2] = -mean_x*s
    
    return H
  
  def __ChoicePoint(self, point_a_homo, point_b_homo, PointNum=12, sample_times=32):

    match_count = point_a_homo.shape[0]  # get number of matches
    # check if matched point smaller than 8
    if match_count<8:
      return False, None, None, None, None
    sample_point_a_homo = np.ones((sample_times, PointNum, 3), dtype=np.float)
    sample_point_b_homo = np.ones((sample_times, PointNum, 3), dtype=np.float)
    sample_point_Ha_matrix = np.zeros((sample_times,3,3), dtype=np.float)
    sample_point_Hb_matrix = np.zeros((sample_times,3,3), dtype=np.float)
    choice_idx = np.arange(0,match_count)
    for i in range(0, sample_times):
      random_idx = np.random.choice(choice_idx, (PointNum), replace=False)
      
      sample_point_a_homo[i,:,:] = point_a_homo[random_idx,:]
      sample_point_b_homo[i,:,:] = point_b_homo[random_idx,:]
      # normalize the sampled point
      sample_point_Ha_matrix[i,:,:] = self.__GetPointNormMat(sample_point_a_homo[i,:,:])
      sample_point_Hb_matrix[i,:,:] = self.__GetPointNormMat(sample_point_b_homo[i,:,:])
      
    return  True, sample_point_a_homo, sample_point_b_homo, sample_point_Ha_matrix, sample_point_Ha_matrix


  def FindFundamentalMatrix_RANSAC_8point_gpu(self, kp_loc_a, kp_loc_b, match_a2b, similarity_a2b, ransac_try_count=1, PointNum = 12):
    matched_kp_a_idx = np.where(match_a2b>=0)[0]
    matched_kp_b_idx = match_a2b[matched_kp_a_idx]
    match_count = matched_kp_a_idx.shape[0]  # get number of matches
    match_similarity = match_a2b[matched_kp_a_idx]
    # matched point
    A_homo = np.ones((match_count, 3), dtype=np.float)
    B_homo = np.ones((match_count, 3), dtype=np.float)
    A_homo[:,0:2] = kp_loc_a[matched_kp_a_idx,:]
    B_homo[:,0:2] = kp_loc_b[matched_kp_b_idx,:]    
    cond, sample_A_homo, sample_B_homo, Ha, Hb = self.__ChoicePoint(A_homo, B_homo, PointNum=PointNum, sample_times=ransac_try_count)
    with tf.device('/gpu:0'):
      tf_A_homo = tf.Variable(A_homo, dtype=tf.float32)
      tf_B_homo = tf.Variable(B_homo, dtype=tf.float32)  
      tf_sample_A_homo = tf.Variable(sample_A_homo, dtype=tf.float32)
      tf_sample_B_homo = tf.Variable(sample_B_homo, dtype=tf.float32) 
      tf_Ha = tf.Variable(Ha, dtype=tf.float32)
      tf_Hb = tf.Variable(Hb, dtype=tf.float32)
      # normalize the sampled point
      tf_sample_A_homo_norm = tf.transpose(tf.matmul(tf_Ha, tf.transpose(tf_sample_A_homo, perm=[0,2,1])), perm=[0,2,1])
      tf_sample_B_homo_norm = tf.transpose(tf.matmul(tf_Hb, tf.transpose(tf_sample_B_homo, perm=[0,2,1])), perm=[0,2,1])
      sample_A_homo_norm = tf_sample_A_homo_norm.numpy()
      sample_B_homo_norm = tf_sample_B_homo_norm.numpy()
      # construct coordinate matrix
      coor_mat = np.ones((ransac_try_count, PointNum,8), dtype=np.float)
      
      coor_mat[:,:,0] = sample_A_homo_norm[:,:,1]*sample_B_homo_norm[:,:,1]
      coor_mat[:,:,1] = sample_A_homo_norm[:,:,1]*sample_B_homo_norm[:,:,0]
      coor_mat[:,:,2] = sample_A_homo_norm[:,:,1]
      coor_mat[:,:,3] = sample_A_homo_norm[:,:,0]*sample_B_homo_norm[:,:,1]
      coor_mat[:,:,4] = sample_A_homo_norm[:,:,0]*sample_B_homo_norm[:,:,0]
      coor_mat[:,:,5] = sample_A_homo_norm[:,:,0]
      coor_mat[:,:,6] =                           sample_B_homo_norm[:,:,1]
      coor_mat[:,:,7] =                           sample_B_homo_norm[:,:,0]
      
      tf_coor_mat = tf.Variable(coor_mat, dtype=tf.float32)
      
      # solve the fundamental matrix
      equation_result_vect = np.full((ransac_try_count, PointNum, 1), -1.0, dtype=np.float)
      tf_equation_result_vect = tf.Variable(equation_result_vect, dtype=tf.float32)
      tf_fundamental_mat_norm = tf.linalg.lstsq(tf_coor_mat, tf_equation_result_vect)
      tf_fundamental_mat_norm = tf.reshape(tf_fundamental_mat_norm, [ransac_try_count,8])
      fundamental_mat_norm = np.ones((ransac_try_count,9),dtype=np.float)
      fundamental_mat_norm[:,0:8] = tf_fundamental_mat_norm.numpy()
      tf_fundamental_mat_norm = tf.Variable(fundamental_mat_norm,dtype=tf.float32)
      tf_fundamental_mat_norm = tf.reshape(tf_fundamental_mat_norm, [ransac_try_count,3,3])
      # denormalize the fundamental matrix
      tf_fundamental_mat = tf.matmul(tf.transpose(tf_Ha, perm=[0,2,1]), tf_fundamental_mat_norm)
      tf_fundamental_mat = tf.matmul(tf_fundamental_mat, tf_Hb)
      # calclate the epipolar line
      tf_epline_B = tf.matmul(tf_A_homo, tf_fundamental_mat)
      tf_epline2B_distance = tf.math.multiply(tf_epline_B, tf_B_homo)
      tf_epline2B_distance = tf.math.reduce_sum(tf_epline2B_distance, axis=2)
      tf_epline2B_distance = tf.math.abs(tf_epline2B_distance)
      tf_epline2B_distance_norm_factor = tf.math.sqrt(tf.math.square(tf_epline_B[:,:,0])+tf.math.square(tf_epline_B[:,:,1]))
      tf_epline2B_distance_norm = tf.math.divide(tf_epline2B_distance, tf_epline2B_distance_norm_factor)
    epline2B_distance_norm = tf_epline2B_distance_norm.numpy()  
    threshold = (self.DB_loc_H + self.DB_loc_W)/20
    
    one_cond = np.full((ransac_try_count, match_count),True)
    zero_cond = np.full((ransac_try_count, match_count),False)
    inliner_map = np.where(epline2B_distance_norm<threshold, one_cond, zero_cond)
    max_ransac_sample_idx = np.argmax(np.sum(inliner_map, axis=1))
    inliner_in_match_mask = inliner_map[max_ransac_sample_idx,:]
    a = matched_kp_a_idx[inliner_in_match_mask]
    return True, tf_fundamental_mat[max_ransac_sample_idx, : :].numpy(), None, matched_kp_a_idx[inliner_in_match_mask]
      
  #calculate the similarity of the query image feature inside the database     
  # kp_feature_query_img: in shape [KpN, 1024]
  # kp_loc_query_img:     in shape [KpN, 2]
  # holistic_feature_query_img: in shape [4096]
  def FindFundamentalMatrix_RANSAC_8point(self, kp_loc_a, kp_loc_b, match_a2b, similarity_a2b, ransac_try_count=1, PointNum = 12):
    ###### find all matched kp
    
    matched_kp_a_idx = np.where(match_a2b>=0)[0]
    matched_kp_b_idx = match_a2b[matched_kp_a_idx]
    match_count = matched_kp_a_idx.shape[0]  # get number of matches
    # check if matched point smaller than 8
    if match_count<PointNum:
      return False, None, None, None
    ###### Original and Homogeneous koordinate of matched keypoint A  #####
    matched_A_kp_homo = np.ones((match_count, 3), dtype=np.float) 
    matched_B_kp_homo = np.ones((match_count, 3), dtype=np.float)
    # copy coordinate
    matched_A_kp_homo[:,0] = kp_loc_a[matched_kp_a_idx,0]
    matched_B_kp_homo[:,0] = kp_loc_b[matched_kp_b_idx,0] 
    matched_A_kp_homo[:,1] = kp_loc_a[matched_kp_a_idx,1]
    matched_B_kp_homo[:,1] = kp_loc_b[matched_kp_b_idx,1] 
    
    match_similarity = similarity_a2b[matched_kp_a_idx]
    match_similarity_norm = match_similarity/np.sum(match_similarity)
    ##### random select 8 matched keypoints and calculate the fundamental matrix####
    fundamental_mat = np.ones((ransac_try_count, 9), dtype=np.float)
    choice_idx = np.arange(0,match_count)
    equation_result_vect = np.full((PointNum),-1.0, dtype=np.float)
    for i in range(0,ransac_try_count):
      # sample 8 points
      random_idx = np.random.choice(choice_idx, (PointNum), replace=False, p=match_similarity_norm)
      sample_8point_A_homo = matched_A_kp_homo[random_idx, :]
      sample_8point_B_homo = matched_B_kp_homo[random_idx, :]   
      # normalize
      sample_8point_A_homo_norm, Ha = self.__NormalizePoint(sample_8point_A_homo)
      sample_8point_B_homo_norm, Hb = self.__NormalizePoint(sample_8point_B_homo)
      # construct coordinate matrix
      coor_mat = np.ones((PointNum,8), dtype=np.float)
      coor_mat[:,0] = sample_8point_A_homo_norm[:,1]*sample_8point_B_homo_norm[:,1]
      coor_mat[:,1] = sample_8point_A_homo_norm[:,1]*sample_8point_B_homo_norm[:,0]
      coor_mat[:,2] = sample_8point_A_homo_norm[:,1]
      coor_mat[:,3] = sample_8point_A_homo_norm[:,0]*sample_8point_B_homo_norm[:,1]
      coor_mat[:,4] = sample_8point_A_homo_norm[:,0]*sample_8point_B_homo_norm[:,0]
      coor_mat[:,5] = sample_8point_A_homo_norm[:,0]
      coor_mat[:,6] =                                sample_8point_B_homo_norm[:,1]
      coor_mat[:,7] =                                sample_8point_B_homo_norm[:,0]
      
      #coor_mat_transpose = np.transpose(coor_mat)
      #pseudo_coor_mat = np.matmul(coor_mat_transpose, coor_mat)
      #inv_pseudo_coor_mat = np.linalg.inv(pseudo_coor_mat)
      #least_sqare_coor_mat = np.matmul(inv_pseudo_coor_mat, coor_mat_transpose)
      #rank = np.linalg.matrix_rank(coor_mat)
      #if rank <8:
      #  c=0
      fundamental_mat[i,0:8] = np.linalg.lstsq(coor_mat, equation_result_vect)[0] #np.linalg.solve(coor_mat, equation_result_vect)

    fundamental_mat = np.reshape(fundamental_mat, (ransac_try_count, 3,3))

    #### check inliner ####
    threshold = (self.DB_loc_H + self.DB_loc_W)/(2.0*5.0)
    inliner_idx_list = []
    inliner_count = np.zeros((ransac_try_count))
    #transpos_matched_B_kp_homo = np.transpose(matched_B_kp_homo)
    for i in range(0,ransac_try_count):
      funda_mat = fundamental_mat[i]
      
      # fundamental matrix denormalize
      funda_mat = np.matmul(np.transpose(Ha),funda_mat)
      funda_mat = np.matmul(funda_mat, Hb)  
      # enforce rank-2 of funda mat    
      u,s,v = np.linalg.svd(funda_mat)
      s[-1] = 0.0
      s = np.diag(s)
      funda_mat = np.matmul(u,s)
      funda_mat = np.matmul(funda_mat, v)
      #### fiter inliners ######
      # epipolar line
      epilines = np.matmul(matched_A_kp_homo,funda_mat) # [b,a,c]
      # distance of each point in B to epipolar line  
      epline_distance = np.multiply(epilines,matched_B_kp_homo)
      epline_distance = np.abs(np.sum(epline_distance, axis=1))
      epline_distance_norm = epline_distance/np.sqrt(epilines[:,0]*epilines[:,0] + epilines[:,1]*epilines[:,1])
      # thresh the distance to epline
      inliner_idx = np.where(epline_distance_norm<threshold)[0]
      inliner_count[i] = inliner_idx.shape[0]
      inliner_idx_list.append(inliner_idx)
    
    ransac_iter_idx = np.argmax(inliner_count) 
    inliner_idx = inliner_idx_list[ransac_iter_idx]
    inliner_in_distance_idx = matched_kp_a_idx[inliner_idx]
    #print ("Found %3d inliners from %3d matches"%(len(inliner_idx), match_count))
    mask = np.zeros(kp_loc_a.shape[0],dtype=np.int)
    mask[inliner_in_distance_idx] = 1
    funda_mat = fundamental_mat[ransac_iter_idx]
    #frame_sim = np.sum(match_similarity[inliner_idx])  
    #frame_sim /= np.sqrt(kp_a.shape[0]*kp_b.shape[0])
    return True, funda_mat, mask, inliner_in_distance_idx#np.amax(inliner_count), np.amax(inliner_count)/match_count, frame_sim
  
  def RetrieveInDB_RANSAC(self, kp_feature_query,   kp_loc_query,   holistic_feature_query, \
                                 retrieve_top_N=100                       
                           ):
    # normalize the query image holistic feature
    normalized_holistic_feature_query = holistic_feature_query/np.linalg.norm(holistic_feature_query) # normalze the holistic vector                     
    # retrieve the top N candidate by holsitic feature
    top_N_idx_vect = self.__HolFeatureRetrieve_Port(normalized_holistic_feature_query, retrieve_top_N=retrieve_top_N)
    # extract top N frame Kp feature and its coordinate into variable with continous memory footprint
    topN_DBKpFeature, topN_DBKpFeature_loc, topN_Kpdelimiter = self.__ExtractCopyTopNDBLocalFeature_Port(top_N_idx_vect)
    # calculate similarity of query image kp feature and db top N candidate kp features
    KpSimMat = self.cross_cosine_distance_ArrayVsArray(kp_feature_query, topN_DBKpFeature)
    # perform pairwise match of keypoint
    a_kp_num = kp_feature_query.shape[0]
    query2topN_similarity, query2topN_match, query2topN_distance = self.CAcc_PairwiseSimilarityRowPosGraph__MultiThread_WarpperEx(KpSimMat, a_kp_num, a_pos_patch=None, b_pos_mat=None, b_delimiter=topN_Kpdelimiter, dim_y=0, dim_x=0, a_neighbor_patch=None, gaussian_patch=None, a_pos_in_row=None, a_neighbor_patch_mat_contignous=None, a_kp_neighbor_num_mat_contignous=None, para_thread_num=8, alg_type=self.ALG_TYPE_ONLY_PAIRWISE_MATCH)
    
    #perform RANSAC match for each candidate and calculate the similarity between query and topN frame in DB
    query2db_sim = np.zeros(self.db_place_number, dtype=np.float)
    b_kp_idx_start = 0
    for i in range(0, top_N_idx_vect.shape[0]):
      b_kp_i = topN_DBKpFeature_loc[b_kp_idx_start:b_kp_idx_start+topN_Kpdelimiter[i]]
      match_query2b_i = query2topN_match[a_kp_num*i:a_kp_num*(i+1)]
      distance_query2b_i = query2topN_distance[a_kp_num*i:a_kp_num*(i+1)]
      ransac_cond, funda_mat, inliner_mask, inliner_in_distance_idx = self.FindFundamentalMatrix_RANSAC_8point(kp_loc_query, b_kp_i, match_query2b_i, distance_query2b_i, ransac_try_count=30)
      if ransac_cond == False:
        sim = 0
      else:
        sim =  np.sum(distance_query2b_i[inliner_in_distance_idx])
        sim /= np.sqrt(a_kp_num*b_kp_i.shape[0])
      
      query2db_sim[top_N_idx_vect[i]]=sim
      b_kp_idx_start = b_kp_idx_start+topN_Kpdelimiter[i]
    return query2db_sim

  def cACC_RetrieveInDB_RANSAC(self, kp_feature_query,   kp_loc_query,   holistic_feature_query, \
                                 retrieve_top_N=100                       
                           ):
    # normalize the query image holistic feature
    normalized_holistic_feature_query = holistic_feature_query/np.linalg.norm(holistic_feature_query) # normalze the holistic vector                     
    # retrieve the top N candidate by holsitic feature
    top_N_idx_vect = self.__HolFeatureRetrieve_Port(normalized_holistic_feature_query, retrieve_top_N=retrieve_top_N)
    # extract top N frame Kp feature and its coordinate into variable with continous memory footprint
    topN_DBKpFeature, topN_DBKpFeature_loc, topN_Kpdelimiter = self.__ExtractCopyTopNDBLocalFeature_Port(top_N_idx_vect)
    # calculate similarity of query image kp feature and db top N candidate kp features
    KpSimMat = self.cross_cosine_distance_ArrayVsArray(kp_feature_query, topN_DBKpFeature)
    # perform pairwise match of keypoint
    a_kp_num = kp_feature_query.shape[0]
    query2topN_similarity, query2topN_match, query2topN_distance = self.CAcc_PairwiseSimilarityRowPosGraph__MultiThread_WarpperEx(KpSimMat, a_kp_num, a_pos_patch=kp_loc_query, b_pos_mat=topN_DBKpFeature_loc, b_delimiter=topN_Kpdelimiter, dim_y=0, dim_x=0, a_neighbor_patch=None, gaussian_patch=None, a_pos_in_row=None, a_neighbor_patch_mat_contignous=None, a_kp_neighbor_num_mat_contignous=None, para_thread_num=8, alg_type=self.ALG_TYPE_PAIRWISE_RANSAC) #ALG_TYPE_PAIRWISE_RANSAC
    
    #perform RANSAC match for each candidate and calculate the similarity between query and topN frame in DB
    query2db_sim = np.zeros(self.db_place_number, dtype=np.float)
    query2db_sim[top_N_idx_vect] = query2topN_similarity
    return query2db_sim
    
    
  def cACC_RetrieveInDB_Versatile(self, kp_feature_query,   kp_loc_query,   holistic_feature_query, \
                                 retrieve_top_N=100, thread_num=1, alg_type=0, hol_feature_metric="cosine"                      
                           ):
    holistic_feature_query = np.expand_dims(holistic_feature_query, axis=0)  
    # normalize the query image holistic feature
    #normalized_holistic_feature_query = holistic_feature_query/np.linalg.norm(holistic_feature_query) # normalze the holistic vector  
    normalized_holistic_feature_query = self.__NormalizeQueryHolFeature(holistic_feature_query)
    # if only use holistic feature for retrieval
    if alg_type == self.ALG_TYPE_HOLISTIC_FEATURE_MATCH:
      hol_sim = self.__HolFeatureRetrieve_Port(normalized_holistic_feature_query, db_batch_size=1000, retrieve_top_N=-1, metric=hol_feature_metric)
      return np.squeeze(hol_sim)
         
    # prepare local feature dummy positional information                   
    DBtopN_KP_neighbor                    = np.zeros((2,2),dtype=np.int32)
    DBtopN_KP_in_frame_neighbor_delimiter = np.zeros((2,2),dtype=np.int32)
    DBtopN_KP_neighbor_start_idx          = np.zeros((2,2),dtype=np.int32)    
                   
    top_N_idx_vect = []                   
    # retrieve all the places in db
    if  self.enable_topK == False:
      top_N_idx_vect = np.array(range(0,self.db_place_number),dtype=np.int)
      topN_DBKpFeature     = self.db_LocFeature_cache
      topN_DBKpFeature_loc = self.db_LocFeature_loc_cache
      topN_Kpdelimiter     = self.All_Kpdelimiter
      if alg_type == self.ALG_TYPE_PAIRWISE_POSGRAPH or alg_type == self.ALG_TYPE_LPG_MATCH: # prepare local feature positional information
        DBtopN_KP_neighbor                    = self.AllDB_KP_neighbor 
        DBtopN_KP_in_frame_neighbor_delimiter = self.AllDB_KP_in_frame_neighbor_delimiter
        DBtopN_KP_neighbor_start_idx          = self.AllDB_KP_neighbor_start_idx

    else: #retrieve the top N candidate by holsitic feature
      top_N_idx_vect = self.__HolFeatureRetrieve_Port(normalized_holistic_feature_query, retrieve_top_N=retrieve_top_N, metric=hol_feature_metric)
      top_N_idx_vect = np.squeeze(top_N_idx_vect)
      # extract top N frame Kp feature and its coordinate into variable with continous memory footprint
      topN_DBKpFeature, topN_DBKpFeature_loc, topN_Kpdelimiter = self.__ExtractCopyTopNDBLocalFeature_Port(top_N_idx_vect)
      #if alg_type == self.ALG_TYPE_PAIRWISE_POSGRAPH:
      if alg_type == self.ALG_TYPE_PAIRWISE_POSGRAPH or alg_type == self.ALG_TYPE_LPG_MATCH: # prepare local feature positional information
        DBtopN_KP_neighbor, DBtopN_KP_in_frame_neighbor_delimiter, DBtopN_KP_neighbor_start_idx = self.__ExtractCopyTopNDB_Kp_ClosestNeighbor_Port(top_N_idx_vect)
    # calculate similarity of query image kp feature and db top N candidate kp features
    KpSimMat = self.cross_cosine_distance_ArrayVsArray_batch(kp_feature_query, topN_DBKpFeature)   
    #KpSimMat = self.cross_cosine_distance_ArrayVsArray(kp_feature_query, topN_DBKpFeature)
    # perform pairwise match of keypoint
    query_kp_num = kp_feature_query.shape[0]
    
    query2topN_similarity, query2topN_match, query2topN_distance, query2topN_distance_aux = self.CAcc_PairwiseSimilarityVersatile__MultiThread_Warpper(\
                    KpSimMat, \
                    query_kp_num, query_pos_patch=kp_loc_query, \
                    DBtopN_pos_mat=topN_DBKpFeature_loc, DBtopN_delimiter=topN_Kpdelimiter, \
                    dim_y=60, dim_x=60, \
                    DBtopN_KP_neighbor=DBtopN_KP_neighbor, DBtopN_neighbor_in_frame_delimiter=DBtopN_KP_in_frame_neighbor_delimiter,  DBtopN_frame_neighbor_start_idx=DBtopN_KP_neighbor_start_idx,
                    gaussian_patch=self.gaussian_patch,\
                    para_thread_num=thread_num, alg_type=alg_type) #ALG_TYPE_PAIRWISE_RANSAC
  
      
    #perform RANSAC match for each candidate and calculate the similarity between query and topN frame in DB
    query2db_sim = np.zeros(self.db_place_number, dtype=np.float)
    query2db_sim[top_N_idx_vect] = query2topN_similarity
    return query2db_sim  

#######Retrieve K queries from the database for VPR################
# kp_feature_query_list: list for K query local features   
# kp_loc_query_list: list for Kquery local feature loc  
# holistic_feature_query_array: in shape[K,hol_dim] 
# retrieve_top_N: The top N candidates retrieved by the holistic feature
# thread_num: The number of thread to work for the refinement stage 
# alg_type: The type of algorithm for the refinement stage 
# hol_feature_metric: Holistic feature metric 
  def cACC_MultiQueryRetrieveInDB_Versatile(self, kp_feature_query_list,   kp_loc_query_list,   holistic_feature_query_array, \
                                 retrieve_top_N=100, thread_num=1, alg_type=0, hol_feature_metric="cosine"                      
                           ): 
    query_K = len(kp_feature_query_list)  # get the number of query frame

    # normalize the query image holistic feature
    #normalized_holistic_feature_query = holistic_feature_query/np.linalg.norm(holistic_feature_query) # normalze the holistic vector  
    #normalized_holistic_feature_query_array = holistic_feature_query_array #self.__NormalizeQueryHolFeature(holistic_feature_query_array)
    
    normalized_holistic_feature_query_array = self.__NormalizeQueryHolFeature(holistic_feature_query_array)

    ######### Check if only do holistic feature retrieval
    if alg_type == self.ALG_TYPE_HOLISTIC_FEATURE_MATCH:
      hol_sim = self.__HolFeatureRetrieve_Port(normalized_holistic_feature_query_array, db_batch_size=1000, retrieve_top_N=-1, metric=hol_feature_metric)
      return hol_sim.T
         
    # prepare local feature dummy positional information                   
    #DBtopN_KP_neighbor                    = np.zeros((2,2),dtype=np.int32)
    #DBtopN_KP_in_frame_neighbor_delimiter = np.zeros((2,2),dtype=np.int32)
    #DBtopN_KP_neighbor_start_idx          = np.zeros((2,2),dtype=np.int32)    
    # concatenate the kp feature
    kp_feature_query = np.concatenate(kp_feature_query_list, axis=0)
    #print ("query feature num %d"%(kp_feature_query.shape[0]))  
    query2db_sim = np.zeros((self.db_place_number,query_K), dtype=np.float32)
    top_N_idx_vect = []   # in shape [K,retrieve_top_N]. The top N candidates positions in the database for each query (totally K queries). The N defined by retrieve_top_N         
    top_N_frame_kp_neigbor_idx = []

    ######### Retrieve all in the database
    # (STEP. 1)  Calculate the cross similarity by GPU of local features of each query to all in the database 
    if  self.enable_topK == False: # retrieve all the places in db
      #top_N_idx_vect = np.array ([1,4,5,7,9, 10,11,14,15,20,27, 100,101,102,103], dtype=np.int32) #self.db_enabled_place
      top_N_idx_vect = self.db_enabled_place
      # calculate cross similarity of the batch query frame kp with all the kp in the database
      #start_time = time.time()
      # (STEP. 2)  calculate final similarity of each query to all in the database using CPUs
      KpSimMat = self.Query(kp_feature_query) # KpSimMat in column-major storage format
      KpSimMat_w = self.base_mat_h
      #end_time = time.time()
      #self.time_analysis += end_time-start_time
      
      #top_N_frame_kp_neigbor_idx = self.db_kp_spatial_closest_neighbor_frame_start_idx 
      #top_N_frame_kp_start_idx         = self.db_frame_loc_feature_start_idx
    else: # refine the top K candidates (currently does not support batch query top K )
      
      top_N_idx_vect = self.__HolFeatureRetrieve_Port(normalized_holistic_feature_query_array, retrieve_top_N=retrieve_top_N, metric=hol_feature_metric)
      
      top_N_idx_vect = top_N_idx_vect[0,:]
      #top_N_idx_vect = np.sort(top_N_idx_vect)
      # get all the frame kp neighbor start idx
      #top_N_frame_kp_neigbor_idx = self.db_kp_spatial_closest_neighbor_frame_start_idx[top_N_idx_vect]  
      #top_N_frame_kp_start_idx   = self.db_frame_loc_feature_start_idx[top_N_idx_vect]
      total_topk_kp              = np.sum(self.db_frame_loc_feature_delimiter[top_N_idx_vect])
      # (STEP. 2) calculate cross similarity of the query frame kp with kp of the top K candidate frames in the database
    
      KpSimMat = self.QueryK(kp_feature_query, top_N_idx_vect, total_topk_kp) # KpSimMat in column-major storage format

      KpSimMat_w = total_topk_kp
    query_kp_idx_start = 0 # the dim-0 idx for KpSimMat 
      
    for k in range(0, query_K): 
      kp_loc_query = kp_loc_query_list[k]
      query_kp_num = kp_loc_query.shape[0]
      
      query2topN_similarity, query2topN_match, query2topN_distance, query2topN_distance_aux = self.CAcc_PairwiseSimilarityVersatile__MultiThread_Warpper(\
                    #KpSimMat[query_kp_idx_start:query_kp_idx_start+query_kp_num,:], \
                    KpSimMat[query_kp_idx_start*KpSimMat_w:(query_kp_idx_start+query_kp_num)*KpSimMat_w], query_kp_num, KpSimMat_w,\
                    query_kp_num, query_pos_patch=kp_loc_query, \
                    DBtopN_pos_mat=self.db_LocFeature_loc_cache,\
                    DBtopN_delimiter=self.db_frame_loc_feature_delimiter, \
                    DB_frame_kp_start_idx = self.db_frame_loc_feature_start_idx, \
                    dim_y=60, dim_x=60, \
                    DBtopN_KP_neighbor=self.db_kp_spatial_closest_neighbor_cache, DBtopN_neighbor_in_frame_delimiter=self.db_kp_spatial_closest_neighbor_delimiter_cache,  DBtopN_frame_neighbor_start_idx=self.db_kp_spatial_closest_neighbor_frame_start_idx ,
                    valid_db_frame_idx = top_N_idx_vect, \
                    gaussian_patch=self.gaussian_patch, gauss_h = self.gaussian_h, gauss_w = self.gaussian_w,\
                    para_thread_num=thread_num, alg_type=alg_type) #ALG_TYPE_PAIRWISE_RANSAC 
      query_kp_idx_start += query_kp_num  
      query2db_sim[top_N_idx_vect, k] = query2topN_similarity 
      
    # return the result similarity 

    #print (end_time-start_time)
    return query2db_sim
        



  def cACC_RetrieveInDB_Versatile_with_TopN_indices(self, kp_feature_query,   kp_loc_query, top2down_sort_indices,\
                                 retrieve_top_N=100, thread_num=1, alg_type=0                       
                           ):                   
    # retrieve the top N candidate by holsitic feature
    top_N_idx_vect = top2down_sort_indices[0:retrieve_top_N]
    # extract top N frame Kp feature and its coordinate into variable with continous memory footprint
    topN_DBKpFeature, topN_DBKpFeature_loc, topN_Kpdelimiter = self.__ExtractCopyTopNDBLocalFeature_Port(top_N_idx_vect)
    # calculate similarity of query image kp feature and db top N candidate kp features
    KpSimMat = self.cross_cosine_distance_ArrayVsArray(kp_feature_query, topN_DBKpFeature)
    # perform pairwise match of keypoint
    query_kp_num = kp_feature_query.shape[0]
    #if alg_type == self.ALG_TYPE_PAIRWISE_POSGRAPH:
    if alg_type == self.ALG_TYPE_PAIRWISE_POSGRAPH:
      DBtopN_KP_neighbor, DBtopN_KP_in_frame_neighbor_delimiter, DBtopN_KP_neighbor_start_idx = self.__ExtractCopyTopNDB_Kp_ClosestNeighbor_Port(top_N_idx_vect)
    else:
      DBtopN_KP_neighbor                    = np.zeros((2,2),dtype=np.int32)
      DBtopN_KP_in_frame_neighbor_delimiter = np.zeros((2,2),dtype=np.int32)
      DBtopN_KP_neighbor_start_idx          = np.zeros((2,2),dtype=np.int32)
    query2topN_similarity, query2topN_match, query2topN_distance = self.CAcc_PairwiseSimilarityVersatile__MultiThread_Warpper(\
                    KpSimMat, \
                    query_kp_num, query_pos_patch=kp_loc_query, \
                    DBtopN_pos_mat=topN_DBKpFeature_loc, DBtopN_delimiter=topN_Kpdelimiter, \
                    dim_y=0, dim_x=0, \
                    DBtopN_KP_neighbor=DBtopN_KP_neighbor, DBtopN_neighbor_in_frame_delimiter=DBtopN_KP_in_frame_neighbor_delimiter,  DBtopN_frame_neighbor_start_idx=DBtopN_KP_neighbor_start_idx,
                    gaussian_patch=self.gaussian_patch,\
                    para_thread_num=thread_num, alg_type=alg_type) #ALG_TYPE_PAIRWISE_RANSAC
  
    
    #perform RANSAC match for each candidate and calculate the similarity between query and topN frame in DB
    query2db_sim = np.zeros(self.db_place_number, dtype=np.float)
    query2db_sim[top_N_idx_vect] = query2topN_similarity
    return query2db_sim      

  # extract the top N local feature entries in db and copy them to a new variable with continous memory footprint
  # To accelerate this function you can use scatter-gether DMA to copy the data to memory location of destionation variables
  # @top_N_idx_vect: 1D vector that contains indices of top N entry in db LocFeature and LocFeature
  def __ExtractCopyTopNDBLocalFeature_Port(self, top_N_idx_vect, without_feature=False):
    feature_list = []
    feature_loc_list = []
    delimiter = []

    for i in range(0,top_N_idx_vect.shape[0]): # extract and copy each frame in db according to the top N idx
      idx = top_N_idx_vect[i]
      feature_list.append(np.copy(self.db_LocFeature_cache[idx], order='C'))
      feature_loc_list.append(np.copy(self.db_LocFeature_loc_cache[idx], order='C'))
      delimiter.append(self.db_LocFeature_cache[idx].shape[0])
    return np.concatenate(feature_list, axis = 0),  np.concatenate(feature_loc_list, axis = 0), np.array(delimiter,dtype=np.int)

  def GetDBTopNDBFrameDelimiter(self, top_N_idx_vect):
    delimiter = []

    for i in range(0,top_N_idx_vect.shape[0]): # extract and copy each frame in db according to the top N idx
      idx = top_N_idx_vect[i]
      delimiter.append(self.db_LocFeature_cache[idx].shape[0])
    return np.array(delimiter,dtype=np.int)
    
  # return 1. closest neighbor in shape [J,3] 
  # return 2. in frame closest neighbor delimiter in shame [Total keypoint Num]. The delimiter of this return is keypoint number of each frame
  # return 3. frame direct index in shape[TopN]. Each holds the index of start of closest neighbor of each frame in 1
  def __ExtractCopyTopNDB_Kp_ClosestNeighbor_Port(self, top_N_idx_vect):
    start_idx_list = []
    neighbor_list = []
    delimiter_list = []
    accumlate_idx = 0
    for i in range(0,top_N_idx_vect.shape[0]): # extract and copy each frame in db according to the top N idx
      frame_idx = top_N_idx_vect[i]
      start_idx_list.append(accumlate_idx)
      neighbor_list.append(np.copy(self.db_kp_spatial_closest_neighbor_cache[frame_idx], order='C'))
      delimiter_list.append(np.copy(self.db_kp_spatial_closest_neighbor_delimiter_cache[frame_idx], order='C'))
      accumlate_idx += self.db_kp_spatial_closest_neighbor_cache[frame_idx].shape[0]
      
    return np.concatenate(neighbor_list, axis = 0),  np.concatenate(delimiter_list, axis = 0), np.array(start_idx_list,dtype=np.int32)      

#def CalculateRANSAC(db_f_files_list, db_f_loc_files_list, db_hol_files_list\
#                    query_f_files_list, query_f_loc_files_list, query_hol_file, fm_h, fm_w):
#  imgsim=ImgSim.ImgSimilarityRANSAC(db_f_files_list, db_f_loc_files_list, fm_h, fm_w, db_hol_files_list, \
#                                    acc_file='/home/fangming/svn_test/SVN_repository/PHD/RTX3090/lib/FeatureMatchAcc.so') 
#  query_hol_feature = scipy.io.loadmat(query_hol_file)
#  for i in range(0,len(query_files_list)):
#    kp_feature = np.load(query_f_files_list[i], allow_pickle=True, encoding="latin1")
#    kp_loc     = np.load(query_f_loc_files_list[i], allow_pickle=True, encoding="latin1")
#    batch_frame_num = kp_feature.shape[0]
#    for i in range(0, query_hol_feature.shape[0]):
#      imgsim.RetrieveInDB_RANSAC()
    
    
      
