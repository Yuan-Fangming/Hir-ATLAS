import os
import random
import numpy as np
####################Benchmark Code####################################
def clearGTposBoundry(sim, GTsoft_hard, minValue):
  for i in range(0, GTsoft_hard.shape[0]):
    for j in range(0, GTsoft_hard.shape[1]):
       if GTsoft_hard[i,j]==True:
        sim[i,j]=minValue
  return sim
def CreatePR(sim, GThard, GTsoft=None, cond_a = False, cond_b = False):
    
  print (GThard[0:10,0:10])  
    
  logic_gt_hard = GThard > 0.5  #np.greater(GThard, 0.5).all()
  logic_gt_soft = GTsoft > 0.5  #np.greater(GTsoft, 0.5).all()
  
  
  gt_size = GThard.shape[0]*GThard.shape[1]

  
  
  GTsoft_hard = np.logical_and(GTsoft, np.logical_not(GThard))
  sim = clearGTposBoundry(sim, GTsoft_hard, np.amin(sim) ) 
   
   
  R=[0]
  P=[1]

  startV = np.amax(sim)
  endV   = np.amin(sim)   
   
  bestF = 0
  bestT = startV
  bestP = 0
  bestR = 0  
  
  inc_value = (startV-endV)/99.0
  thrshold = startV
  for i in range(0,100):
    B = np.greater(sim, thrshold-inc_value/2.0)
    not_B = np.logical_not(B)
    TP = np.count_nonzero(np.logical_and(logic_gt_hard, B))
    if TP ==0:
      TP = 0.00000000001
    FN = np.count_nonzero(np.logical_and(logic_gt_hard, not_B))
    FP = gt_size - np.count_nonzero(np.logical_or(not_B, logic_gt_hard))
    
    P.append(TP/(TP+FP))
    R.append(TP/(TP+FN))
    F = 2.0*P[i]*R[i]/(P[i]+R[i])
    if F>bestF:
      bestF = F
      bestT = thrshold
      bestP = P[i]
      bestR = R[i]
      
    thrshold-=inc_value
  avgP = np.trapz(P,x=R)
  return avgP, P, R, bestP, bestR, bestF



def RecallK(sim, logic_gt_hard, K_list):
  recall_at_K = []
  for j in range(0,len(K_list)):
    K = K_list[j]
    predict_positive_mat = np.zeros(sim.shape) 
    for i in range(0, sim.shape[1]):
      predict_positive_mat[sim[:,i].argsort()[-K:],i]=1.0
    predict_positive_mat = np.logical_and(predict_positive_mat>0.5, logic_gt_hard)
    recall_at_K.append(float(np.sum(predict_positive_mat))/float(np.sum(logic_gt_hard)) )
  return recall_at_K
 
def CreatePR_Recall_K(sim, GThard, GTsoft=None, recall_K_list=[], cond_a = False, cond_b = False):
    
  logic_gt_hard = GThard > 0.5  #np.greater(GThard, 0.5).all()
  logic_gt_soft = GTsoft > 0.5  #np.greater(GTsoft, 0.5).all()
  
  
  gt_size = GThard.shape[0]*GThard.shape[1]

  
  
  GTsoft_hard = np.logical_and(GTsoft, np.logical_not(GThard))
  sim = clearGTposBoundry(sim, GTsoft_hard, np.amin(sim) ) 
  
  
  
  ######## calculate recall@K ########
  recall_at_K = RecallK(sim, logic_gt_hard, recall_K_list)

    
   
  R=[0]
  P=[1]

  startV = np.amax(sim)
  endV   = np.amin(sim)   
   
  bestF = 0
  bestT = startV
  bestP = 0
  bestR = 0  
  
  inc_value = (startV-endV)/99.0
  thrshold = startV
  for i in range(0,100):
    B = np.greater(sim, thrshold-inc_value/2.0)
    not_B = np.logical_not(B)
    TP = np.count_nonzero(np.logical_and(logic_gt_hard, B))
    if TP ==0:
      TP = 0.00000000001
    FN = np.count_nonzero(np.logical_and(logic_gt_hard, not_B))
    FP = gt_size - np.count_nonzero(np.logical_or(not_B, logic_gt_hard))
    
    P.append(TP/(TP+FP))
    R.append(TP/(TP+FN))
    F = 2.0*P[i]*R[i]/(P[i]+R[i])
    if F>bestF:
      bestF = F
      bestT = thrshold
      bestP = P[i]
      bestR = R[i]
      
    thrshold-=inc_value
  avgP = np.trapz(P,x=R)
  return avgP, P, R, bestP, bestR, bestF, recall_at_K
  
