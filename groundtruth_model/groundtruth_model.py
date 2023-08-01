
# ---------------------- Import Libs ---------------------- #

from dataset_synth import dataset_synth

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *

import numpy as np


def genModel(conceptid):
    print("# ---------------------- Define Model ---------------------- #")

    #### Concept classification

    x = Input(shape = (2*3*3*2, 2*3*3*2, 3), name='input0')
    y = MaxPooling2D(pool_size=(2,2),strides=(2,2), padding='valid')(x)
    y = Conv2D(12*3, (3,3), strides=(3,3), padding='valid', name='conv0', use_bias=True, dtype='float32')(y)
    y = ReLU(max_value=1)(y)
    y = Conv2D(8, (2,2), strides=(2,2), padding='valid',  name='conv1', use_bias=True)(y)
    y = ReLU(max_value=1)(y)
    y = Conv2D(8*2, (1,1), strides=(1,1), padding='valid',name='conv2', use_bias=True)(y)
    y = ReLU(max_value=1)(y)
    y = Conv2D(5, (1,1), strides=(1,1), padding='valid', name='conv3', use_bias=True)(y)
    y = ReLU(max_value=1)(y)

    y = Flatten()(y)
    y = Dense(30, name='dense0', use_bias=True)(y)
    y = ReLU(max_value=1)(y)
    y = Dense(60, name='dense1', use_bias=True)(y)
    y = ReLU(max_value=1)(y)
    y = Dense(5, name='dense2', use_bias=True)(y)
    y = ReLU(max_value=1)(y)

    model_classifier = Model(x,y)
    model_classifier.compile(loss='binary_crossentropy', optimizer="adam", metrics=['acc'])
    model_classifier.summary()


    # # ---------------------- Set weights of model ---------------------- #
    print("# ---------------------- Set weights of model ---------------------- #")
    
    

    # ****              CONV0: pc+channel*12 - PC detector at channel                   ****

    weigths = model_classifier.get_layer('conv0').get_weights()
    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0

    for channel in range(3):
        for pc in range(12):
            weigths[0][:,:,channel,pc+channel*12] = dataset_synth.gen_concept_part_weights(pc)
            weigths[1][pc+channel*12] = -1.0
         
    model_classifier.get_layer('conv0').set_weights(weigths)


    # ****              CONV1: a&b                   ****

    candidates = [["and",7,0,0,   10,0,1],
                  ["and",4,1,0,   1,1,1],
                  
                  ["and",6,0,0,   9,0,1],
                  ["and",3,1,0,   0,1,1],
                  
                  ["or",7+12*1,0,0,   10+12*1,0,1],
                  
                  ["and",8+12*1,0,0,   11+12*1,0,1],
                  
                  ["or",8+12*1,0,0,   11+12*1,0,1],
                  
                  ["and",4+12*2,1,0,   1+12*2,1,1]]

    weigths = model_classifier.get_layer('conv1').get_weights()
    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0

    for operator in ["and","or"]:
        for left in range(12*3):
            for right in range(12*3):
                for y_left in range(2):
                    for x_left in range(2):
                        for y_right in range(2):
                            for x_right in range(2):
                            
                                if [operator,  left, y_left, x_left,      right, y_right, x_right ] in candidates:
                                    
                                    index = [str(x) for x in candidates].index(str([operator,  left, y_left, x_left,      right, y_right, x_right ]))
                                    
                                    if operator == "and":
                                        weigths[0][y_left, x_left, left, index] = 1.0
                                        weigths[0][y_right, x_right, right, index] = 1.0
                                        
                                        weigths[1][index] = -1.0

                                    elif operator == "or":
                                        weigths[0][y_left, x_left, left, index] = 1.0
                                        weigths[0][y_right, x_right, right, index] = 1.0
                                        
                                        weigths[1][index] = 0.0
                                 
    model_classifier.get_layer('conv1').set_weights(weigths)

                 
    # ****              CONV2: c&d                  ****

    weigths = model_classifier.get_layer('conv2').get_weights()
    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0
    
    for id in range(8):            
        weigths[0][0, 0, id, id] = 1.0
        weigths[1][id] = 0
        
        weigths[0][0, 0, id, id+8] = -1.0
        weigths[1][id+8] = 1.0
    
    model_classifier.get_layer('conv2').set_weights(weigths)


    # ****              CONV3: e                  ****


    weigths = model_classifier.get_layer('conv3').get_weights()
    weigths[0][:,:,:,:] = 0 # (Y kernel, X kernel, in channel, out channel)
    weigths[1][:] = 0   
    
          
    weigths[0][0, 0, 0, 0] = 1
    weigths[0][0, 0, 1, 0] = 1
    weigths[1][0] = 0
    
    weigths[0][0, 0, 2, 1] = 1
    weigths[0][0, 0, 3, 1] = 1
    weigths[1][1] = -1

    weigths[0][0, 0, 4, 2] = 1
    weigths[0][0, 0, 7, 2] = 1
    weigths[1][2] = -1
    
    weigths[0][0, 0, 5, 3] = 1
   
    weigths[0][0, 0, 6, 4] = 1
    weigths[0][0, 0, 5+8, 4] = 1
    weigths[1][4] = -1
    
            
    model_classifier.get_layer('conv3').set_weights(weigths)



    #  Dense0:  (None, 45) (concept vector top left, '' top middle, '' top right, etc) -> (45,30)



    weigths = model_classifier.get_layer('dense0').get_weights()
    weigths[0][:,:] = 0 # (in node, out node)
    weigths[1][:] = 0
    
    idx = 0
    
    for concept_id in range(5):

        weigths[0][concept_id+0*5, idx] = 1 # pos 0
        weigths[1][idx] = 0
        idx += 1
        
        
        weigths[0][concept_id+3*5, idx] = 1 # pos 3
        idx += 1        
        
        weigths[0][concept_id+1*5, idx] = 1 # pos 1 AND pos 2
        weigths[0][concept_id+2*5, idx] = 1
        weigths[1][idx] = -1
        idx += 1
        
        
        weigths[0][concept_id+1*5, idx] = 1 # pos 1 OR pos 2
        weigths[0][concept_id+2*5, idx] = 1 
        weigths[1][idx] = 0
        idx += 1
        
        
        weigths[0][concept_id+0*5, idx] = 1 # pos 0 OR pos 1
        weigths[0][concept_id+1*5, idx] = 1
        weigths[1][idx] = 0
        idx += 1

        
        weigths[0][concept_id+2*5, idx] = 1# pos 2 OR pos 3
        weigths[0][concept_id+3*5, idx] = 1
        weigths[1][idx] = 0
        idx += 1

            
    model_classifier.get_layer('dense0').set_weights(weigths)


    # ****              Dense1: c&d                  ****


    weigths = model_classifier.get_layer('dense1').get_weights()
    weigths[0][:,:] = 0 # (in node, out node)
    weigths[1][:] = 0
    

    for id in range(30):            
        weigths[0][id, id] = 1.0
        weigths[1][id] = 0.0
        
        weigths[0][id, id+30] = -1.0
        weigths[1][id+30] = 1.0
    
    model_classifier.get_layer('dense1').set_weights(weigths)


    # ****              Dense2: e                  ****

    weigths = model_classifier.get_layer('dense2').get_weights()
    weigths[0][:,:] = 0 # (in node, out node)
    weigths[1][:] = 0
    
    
    weigths[0][0+conceptid*6, 0] = 1
    weigths[1][0] = 0
    
    weigths[0][1+conceptid*6, 1] = 1
    weigths[1][1] = 0
        
    weigths[0][2+conceptid*6, 2] = 1
    weigths[1][2] = 0
    
    weigths[0][3+conceptid*6, 3] = 1
    weigths[0][2+30+conceptid*6, 3] = 1
    weigths[1][3] = -1
    
    weigths[0][4+30+conceptid*6, 4] = 1
    weigths[0][5+30+conceptid*6, 4] = 1
    weigths[1][4] = -1

    model_classifier.get_layer('dense2').set_weights(weigths)
    
    
    return model_classifier
