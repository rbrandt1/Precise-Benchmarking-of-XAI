
# ---------------------- Import Libs ---------------------- #
from dataset_synth import dataset_synth
from groundtruth_model import groundtruth_model

from xai_metrics.Our_CompactnessMetric import Our_CompactnessMetric
from xai_metrics.Our_CompletenessMetric import Our_CompletenessMetric
from xai_metrics.Our_CorrectnessMetric import Our_CorrectnessMetric

import cv2

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *

import pandas as pd

import numpy as np

import time

from xplique.attributions import DeconvNet,GradCAM,GradCAMPP,GradientInput,GuidedBackprop,IntegratedGradients,Saliency,SmoothGrad,SquareGrad,VarGrad,KernelShap,Lime,Occlusion,Rise
from xplique.metrics import MuFidelity,Deletion,Insertion

from skimage.metrics import structural_similarity

import sklearn.metrics

import os

# ---------------------- Global Settings ---------------------- #

settings = {}

settings['num_examples_per_class'] = 16
settings['normalize_explanations'] = True 

settings['saveSelectedImages'] = True
settings['saveAllImages'] = False
settings['resize_Images'] = False

all_xai_methods = [GradCAM,GradCAMPP,Saliency,DeconvNet, GradientInput, GuidedBackprop, IntegratedGradients, SmoothGrad, SquareGrad,VarGrad,Occlusion,Rise,KernelShap,Lime] 
all_xai_methods_names = ["GradCAM","GradCAMPP","Saliency","DeconvNet", "GradientInput", "GuidedBackprop", "IntegratedGradients", "SmoothGrad", "SquareGrad","VarGrad","Occlusion","Rise","KernelShap","Lime"]
      
all_xai_metrics = [Deletion,Insertion, MuFidelity] 
all_xai_metrics_names = ["Deletion","Insertion", "MuFidelity"]

    
# ---------------------- Functions ---------------------- #

def checkNum(num):
    if (num == None) or (not np.isreal(num)) or (np.isnan(num)):
        print("ERROR checkNum: ",num)
        exit()

def checkList(list_var):

    for num in list_var:
        checkNum(num)

def saveImages(path,images,conceptid, applyColormap,data,postfix=""):

    for img,id in zip(images,range(len(images))):
    
        if settings['resize_Images']:
            img2 = cv2.resize(img,(72,72), interpolation = cv2.INTER_NEAREST)
        else:
            img2 = img
        
        if applyColormap:
            saveImageColomap(path+str(conceptid)+"_"+str(np.argmax(data['test']['y'][id]))+"_"+str(id)+postfix+".png", img2)
        else:
            saveImage(path+str(conceptid)+"_"+str(np.argmax(data['test']['y'][id]))+"_"+str(id)+postfix+".png", img2)

def saveImage(path, image):
    cv2.imwrite(path, image*255) 


def saveImageColomap(path, image):
    if image.shape[-1] == 3:
        image2 = dataset_synth.convert_to_2D(image)
    else:
        image2 = image
    
    scaled = (((image2 + 1)/2.0)*255).astype(np.uint8)
    im_color = cv2.applyColorMap(scaled, cv2.COLORMAP_CIVIDIS)
    cv2.imwrite(path, im_color) 
    

def saveColormap():

    img = np.zeros((256,10))
    
    for i in range(0,256):
        img[i,:] = (i/255.0)*2-1
    
    saveImageColomap("./paper_images/selected/colormap.png", img)
    
def startTimer():       
    return time.time()
    
def endTimer(start_time, num_items):
    return ((time.time() - start_time)*1000.0) / num_items


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict
   
   
def flatten_and_turn_binary(explanations_copy,gt_explanation_copy):
    explanations_copy_flat = np.zeros((explanations_copy.shape[0], np.prod(explanations_copy.shape[1:])), dtype='bool')
    gt_explanation_copy_flat = np.zeros((gt_explanation_copy.shape[0], np.prod(gt_explanation_copy.shape[1:])), dtype='bool')
    
    for idx_iter in range(gt_explanation_copy.shape[0]):
         
        gt_binary = np.zeros(gt_explanation_copy[idx_iter].shape, dtype='bool')
        gt_binary[gt_explanation_copy[idx_iter] != 0] = True
        
        gt_explanation_copy_flat[idx_iter] = gt_binary.flatten()
        
        
        explanation_binary = np.zeros(explanations_copy[idx_iter].shape, dtype='bool')
        explanation_binary[explanations_copy[idx_iter] != 0] = True
        
        explanations_copy_flat[idx_iter] = explanation_binary.flatten()
        
    return explanations_copy_flat, gt_explanation_copy_flat
   
def turn_binary(explanations_copy, invert):

    if invert:
        explanations_copy_binary = np.ones(explanations_copy.shape, dtype='bool')
        
        explanations_copy_binary[explanations_copy != 0] = False
    else:
        explanations_copy_binary = np.zeros(explanations_copy.shape, dtype='bool')

        explanations_copy_binary[explanations_copy != 0] = True
    
    return explanations_copy_binary
   
   
def flatten_only(explanations_copy,gt_explanation_copy):
    explanations_copy_flat = np.zeros((explanations_copy.shape[0], np.prod(explanations_copy.shape[1:])))
    gt_explanation_copy_flat = np.zeros((gt_explanation_copy.shape[0], np.prod(gt_explanation_copy.shape[1:])))
    
    for idx_iter in range(gt_explanation_copy.shape[0]):
        gt_explanation_copy_flat[idx_iter] = gt_explanation_copy[idx_iter].flatten()
        explanations_copy_flat[idx_iter] = explanations_copy[idx_iter].flatten()
   
    return explanations_copy_flat, gt_explanation_copy_flat
   
   
def calc_metrics_for_XAI_method(Xai_method, xai_name, model, data, conceptid):
        print("*** ",xai_name," ***")

        result_data_xaimethod = {}
        result_data_xaimethod_time = {}


        # calc num_items
        if data['test']['x'].shape[0] != data['test']['gt_explanation'].shape[0]:
            print("ERROR! data['test']['x'].shape[0] != data['test']['gt_explanation'].shape[0]")
            exit()
        num_items = data['test']['x'].shape[0]
        
        # Run xai method and time
                
        start_time = startTimer()
        explainer = Xai_method(model)
        explanations = explainer(data['test']['x'], data['test']['y'])
        explanations = explanations.numpy()
        xai_method_time = endTimer(start_time, num_items)
        
        # normalize explanations
        if settings['normalize_explanations']:
            for i in range(len(explanations)):
                explanations[i] = dataset_synth.normalize_explanation(explanations[i]) 
        else:
            print("WARNING! NOT NORMALIZING EXPLANATIONS! ")
   


        # Determine GT: 2D or 3D? 
        
        if explanations.shape[-1] == 3:# 3D GT
            gt_explanation = data['test']['gt_explanation3D']
            
            print("3D explanations")

            # mufidelity fix
            for i in range(len(explanations)):
                if abs(explanations[i]).sum() == 0:
                    explanations[i][0][0][:] = 0.000000001
            
        else: #2D GT
            gt_explanation = data['test']['gt_explanation']
            
            print("2D explanations")
            
            # mufidelity fix 
            for i in range(len(explanations)):
                if abs(explanations[i]).sum() == 0:
                    explanations[i][0][0] = 0.000000001

        
        # Calc prior metrics
                
        for Metric, metric_name in zip(all_xai_metrics,all_xai_metrics_names):
            print(metric_name)
            
            data_x_copy = data['test']['x'].astype('float32')
            data_y_copy = data['test']['y'].astype('float32')
            explanations_copy = explanations.astype('float32')
            
            start_time = startTimer()
            explainer = Metric(model, data_x_copy, data_y_copy)
            result_data_xaimethod[metric_name] =  explainer(explanations_copy)
            result_data_xaimethod_time[metric_name] = endTimer(start_time, num_items)
        
            checkNum(result_data_xaimethod[metric_name])
    
    
        
        # Cosine Similarity (GUIDOTTI2021103428)
        print("CosSim")

        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')
        
        
        explanations_copy_flat, gt_explanation_copy_flat = flatten_only(explanations_copy, gt_explanation_copy)
        
        start_time = startTimer()
        scores = []
        for gt, explanation in zip(gt_explanation_copy_flat, explanations_copy_flat):
            
            score = np.linalg.norm(np.dot(explanation,gt))/(np.linalg.norm(explanation)*np.linalg.norm(gt))
            
            scores.append(score)

        result_data_xaimethod_time['CosSim'] = endTimer(start_time, num_items)
        result_data_xaimethod['CosSim'] = sum(scores) / len(scores)
        
        checkList(scores)
      
      
      
        # SSIM
        print("SSIM")

        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')
        
        start_time = startTimer()
        scores = []
        for gt, explanation in zip(gt_explanation_copy, explanations_copy):
            
            if explanation.shape[-1] == 3:
                score = structural_similarity(gt,explanation, channel_axis=-1)
            else:
                score = structural_similarity(gt, explanation)
            
            scores.append(score)

        result_data_xaimethod_time['SSIM'] = endTimer(start_time, num_items)
        result_data_xaimethod['SSIM'] = sum(scores) / len(scores)
        
        checkList(scores)
        
        
        
        
        # CONCISSENESS (Amparore_2021)
        print("CONCISSENESS")
        
        explanations_copy = explanations.astype('bool')
              
        start_time = startTimer()
        scores = []
        for explanation in explanations_copy:
            total = np.prod(explanation.shape)
            
            iszero = (total - np.count_nonzero(explanation))+0.0
    
            score = iszero / total
            
            scores.append(score)
            
        
        result_data_xaimethod_time['Conciseness'] = endTimer(start_time,num_items )
        result_data_xaimethod['Conciseness'] = sum(scores) / len(scores)
        
        checkList(scores)
        
        
        
        
        # F1 
        print("F1")
        
        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')
        
        explanations_copy_flat, gt_explanation_copy_flat = flatten_and_turn_binary(explanations_copy,gt_explanation_copy)
            
        start_time = startTimer()
        scores = []
        for gt, explanation in zip(gt_explanation_copy_flat, explanations_copy_flat):
            
            score = sklearn.metrics.f1_score(gt, explanation, average='binary', pos_label=1)
            
            scores.append(score)

        
        result_data_xaimethod_time['F1'] = endTimer(start_time, num_items)
        result_data_xaimethod['F1'] = sum(scores) / len(scores)
        
        checkList(scores)
        
        
        
        
        
        # MAE
        print("MAE")
        
        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')
        
        explanations_copy_flat, gt_explanation_copy_flat = flatten_only(explanations_copy, gt_explanation_copy)
            
        start_time = startTimer()
        scores = []
        for gt, explanation in zip(gt_explanation_copy_flat, explanations_copy_flat):
 
            score = sklearn.metrics.mean_absolute_error(gt, explanation)
            
            scores.append(score)

        
        result_data_xaimethod_time['MAE'] = endTimer(start_time, num_items)
        result_data_xaimethod['MAE'] = sum(scores) / len(scores)
        
        checkList(scores)
   
   
   
   
        # Intersection over union IoU
        print("IoU")
        
        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')
        
        explanations_copy_flat, gt_explanation_copy_flat = flatten_and_turn_binary(explanations_copy,gt_explanation_copy)
        
        start_time = startTimer()
        scores = []
        for gt, explanation in zip(gt_explanation_copy_flat, explanations_copy_flat):
            
            score = sklearn.metrics.jaccard_score(gt, explanation, average='binary', pos_label=1)
            
            scores.append(score)

        result_data_xaimethod_time['IoU'] = endTimer(start_time, num_items)
        result_data_xaimethod['IoU'] = sum(scores) / len(scores)
        
        checkList(scores)
   
   
   
   
        # Precision
        print("Precision")
        
        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')
        
        explanations_copy_flat, gt_explanation_copy_flat = flatten_and_turn_binary(explanations_copy,gt_explanation_copy)
        
        start_time = startTimer()
        scores = []
        for gt, explanation in zip(gt_explanation_copy_flat, explanations_copy_flat):
                        
            score = sklearn.metrics.precision_score(gt, explanation, average='binary', pos_label=1)
            
            scores.append(score)

        
        result_data_xaimethod_time['PR'] = endTimer(start_time, num_items)
        result_data_xaimethod['PR'] = sum(scores) / len(scores)
        
        checkList(scores)
        
        
        
        
        
        
        # Recall
        print("Recall")
        
        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')
        
        explanations_copy_flat, gt_explanation_copy_flat = flatten_and_turn_binary(explanations_copy,gt_explanation_copy)
        
        start_time = startTimer()
        scores = []
        for gt, explanation in zip(gt_explanation_copy_flat, explanations_copy_flat):
            
            score = sklearn.metrics.recall_score(gt, explanation, average='binary', pos_label=1)
            
            scores.append(score)

        
        result_data_xaimethod_time['RE'] = endTimer(start_time, num_items)
        result_data_xaimethod['RE'] = sum(scores) / len(scores)
        
        checkList(scores)
        
        
        
        
        
        
        # Energy-Based Pointing Game (DBLP:journals/corr/abs-1910-01279)
        print("Energy-Based Pointing Game")
        
        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')
        
        gt_explanation_copy_binary = turn_binary(gt_explanation_copy,True)
        
        start_time = startTimer()
        scores = []
        for gt, explanation in zip(gt_explanation_copy_binary, explanations_copy):
        
            top = np.ma.array(abs(explanation), mask = gt).sum()
            if not top:
                top = 0.0
                
            bottom = abs(explanation).sum()
            
            if top == 0 and bottom == 0:
                score = 1.0
            elif top == 0:
                score = 0.0
            else:
                score = top / bottom 
            
            scores.append(score)
            
            
        
        result_data_xaimethod_time['EBPG'] = endTimer(start_time, num_items)
        result_data_xaimethod['EBPG'] = sum(scores) / len(scores)
        
        checkList(scores)
        
        
        
        
        # Relevance Rank Accuracy (arras2021ground)
        print("Relevance Rank Accuracy")
        
        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')
        
        gt_explanation_copy_binary = turn_binary(gt_explanation_copy,False)
        
        start_time = startTimer()
        scores = []
        for gt, explanation in zip(gt_explanation_copy_binary, explanations_copy):
                      
            num_K = np.count_nonzero(gt)
            
            kth_highest_value = np.partition(abs(explanation).flatten(), -(num_K))[-(num_K)]
            
            exp_greater_k = np.zeros(explanation.shape, dtype='bool')
            exp_greater_k[abs(explanation) >= kth_highest_value] = True
            exp_greater_k[gt == 0] = False
            
            score =  exp_greater_k.sum() / num_K
            
            scores.append(score)
            
        result_data_xaimethod_time['RRA'] = endTimer(start_time, num_items)
        result_data_xaimethod['RRA'] = sum(scores) / len(scores)
        
        checkList(scores)
        
        
        

        # Calc our metrics

        print("our metrics")
        
        for operator in ["!=",">","<"]:
            
            explanations_copy = explanations.astype('float32')
            gt_explanation_copy = gt_explanation.astype('float32')
            
            start_time = startTimer()
            compactnessMetric = Our_CompactnessMetric(gt_explanation_copy, explanations_copy,operator)
            result_data_xaimethod['cpa'+operator] = compactnessMetric.get_score()
            result_data_xaimethod_time['cpa'+operator] = endTimer(start_time, num_items)
                   
            checkList(compactnessMetric._scores)
            
            
            explanations_copy = explanations.astype('float32')
            gt_explanation_copy = gt_explanation.astype('float32')

            start_time = startTimer()
            completenessMetric = Our_CompletenessMetric( gt_explanation_copy,  explanations_copy,operator)
            result_data_xaimethod['cpl'+operator] = completenessMetric.get_score()
            result_data_xaimethod_time['cpl'+operator] = endTimer(start_time, num_items)
                     
            checkList(completenessMetric._scores)
                     
                     
            explanations_copy = explanations.astype('float32')
            gt_explanation_copy = gt_explanation.astype('float32')
            
            start_time = startTimer()
            compactnessMetric = Our_CompactnessMetric(gt_explanation_copy, explanations_copy,operator)
            completenessMetric = Our_CompletenessMetric( gt_explanation_copy,  explanations_copy,operator)
            correctnessMetric = Our_CorrectnessMetric(completenessMetric,compactnessMetric)
            result_data_xaimethod['cor'+operator] = correctnessMetric.get_score()
            result_data_xaimethod_time['cor'+operator] = endTimer(start_time, num_items)
        
            checkList(compactnessMetric._scores)
            checkList(completenessMetric._scores)
            
        
        #cor_nosign
        
        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')
       
        start_time = startTimer()
        compactnessMetric = Our_CompactnessMetric( gt_explanation_copy,  explanations_copy,"!=")
        completenessMetric = Our_CompletenessMetric( gt_explanation_copy,  explanations_copy,"!=")
        correctnessMetric = Our_CorrectnessMetric(completenessMetric,compactnessMetric)
        result_data_xaimethod['cor_nosign'] = correctnessMetric.get_score()
        result_data_xaimethod_time['cor_nosign'] = endTimer(start_time, num_items)
        
        checkList(compactnessMetric._scores)
        checkList(completenessMetric._scores)
        
        
        #cor_sign
        
        explanations_copy = explanations.astype('float32')
        gt_explanation_copy = gt_explanation.astype('float32')
        
        start_time = startTimer()
        compactnessMetric = Our_CompactnessMetric(gt_explanation_copy,  explanations_copy,">")
        completenessMetric = Our_CompletenessMetric(gt_explanation_copy,  explanations_copy,">")
        correctnessMetric = Our_CorrectnessMetric(completenessMetric,compactnessMetric)
        completenessscore_greater = correctnessMetric.get_score()
        
        compactnessMetric2 = Our_CompactnessMetric(gt_explanation_copy,  explanations_copy,"<")
        completenessMetric2 = Our_CompletenessMetric(gt_explanation_copy,  explanations_copy,"<")
        correctnessMetric2 = Our_CorrectnessMetric(completenessMetric2,compactnessMetric2)
        completenessscore_less = correctnessMetric2.get_score()
        
        result_data_xaimethod['cor_sign'] = (completenessscore_greater+completenessscore_less)/2.0
        result_data_xaimethod_time['cor_sign'] = endTimer(start_time, num_items)
      
        checkList(compactnessMetric._scores)
        checkList(completenessMetric._scores)
        
        checkList(compactnessMetric2._scores)
        checkList(completenessMetric2._scores)
  
        # Time metric
        
        result_data_xaimethod['Time'] = xai_method_time



        # Save images
        if settings['saveSelectedImages']:
            print("saving images...")
            selected_idx = ([0,3,1,4,2][conceptid])*settings['num_examples_per_class']
            class_idx = [0,3,1,4,2][conceptid]
            
            saveImage("./paper_images/selected/nocolormap_"+xai_name+"_"+str(class_idx)+".png", explanations[selected_idx])
            saveImageColomap("./paper_images/selected/"+xai_name+"_"+str(class_idx)+".png", explanations[selected_idx])
            saveImageColomap("./paper_images/selected/gt2d_"+str(class_idx)+".png", gt_explanation[selected_idx])
            saveImage("./paper_images/selected/input_"+str(class_idx)+".png", data['test']['x'][selected_idx])

            if settings['saveAllImages']:
                
                saveImages("./paper_images/explanations/",explanations,conceptid, True,data,"_"+xai_name) # (path,images,conceptid, applyColormap,data,postfix="")
                saveImages("./paper_images/input/",data['test']['x'],conceptid, False,data)
                saveImages("./paper_images/input/",gt_explanation,conceptid, True,data,"_gt")
                
            else:
                print("!!! NOT saving all images... !!!")
        else:
            print("!!! NOT saving images... !!!")
       
        # Store results
        
        if not xai_name in result_data:
            result_data[xai_name] = [result_data_xaimethod]
            result_data_time[xai_name] = [result_data_xaimethod_time]
                        
        else:
            result_data[xai_name].append(result_data_xaimethod)
            result_data_time[xai_name].append(result_data_xaimethod_time)
            
            
        # ---------------------- test ---------------------- #
        
        if settings['normalize_explanations']:
            for exp in range(len(data['test']['y'])):

                if np.amax(explanations[exp]) > 1 or np.amin(explanations[exp]) < -1:
                    print("ERROR!")
                    exit()
       
                if np.amax(data['test']['x'][exp]) != 1 or np.amin(data['test']['x'][exp]) != 0:
                    print("ERROR!")
                    exit()
                if np.amax(data['test']['gt_explanation'][exp]) > 1 or np.amin(data['test']['gt_explanation'][exp]) < -1:
                    print("ERROR!")
                    exit()
                if np.amax(data['test']['gt_explanation3D'][exp]) > 1 or np.amin(data['test']['gt_explanation3D'][exp]) < -1:
                    print("ERROR!")
                    exit()
                if np.amax(data['test']['gt_explanation3D'][exp]) != np.amax(data['test']['gt_explanation'][exp]) or np.amin(data['test']['gt_explanation3D'][exp]) !=  np.amin(data['test']['gt_explanation'][exp]):
                    print("ERROR!")
                    exit()


result_data = {}
result_data_time = {}

def main():
    
    saveColormap()
    
    
    for conceptid in range(5): 
               
    
        # ---------------------- Load Data ---------------------- #
        print("*** conceptid ",conceptid," ***")
        data = dataset_synth.generate(settings['num_examples_per_class'], conceptid)
        model_classifier = groundtruth_model.genModel(conceptid)


        # ---------------------- Evaluate model ---------------------- #
        loss = model_classifier.evaluate(data['test']['x'], data['test']['y'])
        if loss[0] != 0.0:
            print("ERROR! Loss != 0")
            exit()
        
        # ---------------------- calc_metrics_for_XAI_method ---------------------- #
        
        for xai_method, xai_name in zip(all_xai_methods, all_xai_methods_names):
            calc_metrics_for_XAI_method(xai_method, xai_name, model_classifier, data, conceptid)
        
        
        # ---------------------- test ---------------------- #
        if settings['normalize_explanations']:
            for exp in range(len(data['test']['y'])):
                if np.amax(data['test']['x'][exp]) != 1 or np.amin(data['test']['x'][exp]) != 0:
                    print("ERROR!")
                    exit()
                if np.amax(data['test']['gt_explanation'][exp]) > 1 or np.amin(data['test']['gt_explanation'][exp]) < -1:
                    print("ERROR!")
                    exit()
                if np.amax(data['test']['gt_explanation3D'][exp]) > 1 or np.amin(data['test']['gt_explanation3D'][exp]) < -1:
                    print("ERROR!")
                    exit()
                if np.amax(data['test']['gt_explanation3D'][exp]) != np.amax(data['test']['gt_explanation'][exp]) or np.amin(data['test']['gt_explanation3D'][exp]) !=  np.amin(data['test']['gt_explanation'][exp]):
                    print("ERROR!")
                    exit()
        

    # ---------------------- Print and save results ---------------------- #
        
    for xai_name in all_xai_methods_names:   
        result_data[xai_name] = dict_mean(result_data[xai_name])
        result_data_time[xai_name] = dict_mean(result_data_time[xai_name])
        
    df = pd.DataFrame(data=result_data)
    df.to_csv(r'table_results_.csv', index = True)
    print(df.to_latex(index=True))  
    
    print("")
    print("")
    print("")

    df_time = pd.DataFrame(data=result_data_time)
    #df_time = df_time.mean(axis=1)
    df_time.to_csv(r'table_times_.csv', index = True)
    print(df_time.to_latex(index=True))  
    
if __name__ == "__main__":
    main()
