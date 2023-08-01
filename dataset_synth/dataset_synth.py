
import numpy as np
import random


def normalize_explanation(exp):
   
    max = np.amax(exp)
    min = np.amin(exp)
    
    if max != 0 or min != 0:
        if abs(min) > abs(max):
            exp = exp * 1.0/abs(min)
        else:
            exp = exp * 1.0/abs(max)
    
    if (np.amin(exp) < -1 or np.amax(exp) > 1) or ((np.amin(exp) != -1 and np.amax(exp) != 1) and not (np.amin(exp) == 0 and np.amax(exp) == 0)):
        print("ERROR: np.amin(exp) < -1 or np.amax(exp) > 1 ")
        print(np.amin(exp),np.amax(exp))
        exit()

    return exp
    
def gen_concept_part_weights(id):

    if id == 0:
        return [[2,-.5,1],[-.5,0,-1],[1,0,-1]] # row 2 is residual,     0,2       1,2
    if id == 1:
        return [[0,1,1],[1,-1,-1],[1,0,-1]] # row 2 is residual,     0,2       1,2
    if id == 2:
        return [[0,0,1],[0,2,-1],[1,0,-1]] # row 2 is residual,     0,2       1,2
    if id == 3:
        return [[1,-.5,2],[-1,0,-.5],[1,0,-1]] # row 2 is residual,     0,0       1,0
    if id == 4:
        return [[1,1,0],[-1,-1,1],[1,0,-1]] # row 2 is residual,     0,0       1,0
    if id == 5:
        return [[1,0,0],[-1,2,0],[1,0,-1]] # row 2 is residual,     0,0       1,0
    if id == 6:
        return [[1,0,-1],[1,0,-.5],[-1,-.5,2]] # row 0 is residual,     1,0       2,0
    if id == 7:
        return [[1,0,-1],[1,-1,1],[-1,1,0]]  # row 0 is residual,     1,0       2,0
    if id == 8:
        return [[1,0,-1],[1,2,0],[-1,0,0]]  # row 0 is residual,     1,0       2,0
    if id == 9: 
        return [[1,0,-1],[-.5,0,1],[2,-.5,-1]]  # row 0 is residual,     1,2       2,2
    if id == 10:
        return [[1,0,-1],[1,-1,1],[0,1,-1]]  # row 0 is residual,     1,2       2,2
    if id == 11:
        return [[1,0,-1],[0,2,1],[0,0,-1]] # row 0 is residual,     1,2       2,2
        
def gen_concept_part_example(id):
    if id == 0:
        a =  [[1,0,0],[0,0,0],[.5,1,.5]]
    if id == 1:
        a =  [[1,1,0],[1,0,0],[.5,1,.5]]
    if id == 2:
        a =  [[1,1,0],[1,1,0],[.5,1,.5]]
    if id == 3:
        a =  [[0,0,1],[0,0,0],[.5,1,.5]]
    if id == 4:
        a =  [[0,1,1],[0,0,1],[.5,1,.5]]
    if id == 5:
        a =  [[0,1,1],[0,1,1],[.5,1,.5]]
    if id == 6:
        a =  [[.5,1,.5],[0,0,0],[0,0,1]]
    if id == 7:
        a =  [[.5,1,.5],[0,0,1],[0,1,1]]
    if id == 8:
        a =  [[.5,1,.5],[0,1,1],[0,1,1]]
    if id == 9:
        a =  [[.5,1,.5],[0,0,0],[1,0,0]]
    if id == 10:
        a =  [[.5,1,.5],[1,0,0],[1,1,0]]
    if id == 11:
        a =  [[.5,1,.5],[1,1,0],[1,1,0]]

    
    b = gen_concept_part_weights(id)

    return a, b
        
    
    
     
def calc_attr_neg_concept(img_0,img_1,exp_1):
    img_0 = np.array(img_0)
    img_1 = np.array(img_1)
    exp_1 = np.array(exp_1)
    
    tmp = np.zeros(img_0.shape)
    for y in range(img_0.shape[0]):
        for x in range(img_0.shape[1]):
            if len(img_0.shape) == 3:
                for c in range(img_0.shape[2]):
                    tmp[y,x,c] =  (1 - img_0[y,x,c]) * exp_1[y,x,c]
            else:
                tmp[y,x] =  (1 - img_0[y,x]) * exp_1[y,x]
    return tmp
    
def calc_attr_pos_concept(array):
    img,exp,concept = array
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for chan in range(img.shape[2]):
                exp[y,x,chan] =  img[y,x,chan] * exp[y,x,chan]
    return img,exp,concept
    
    
def gen_concept_examples(id, numexamples):
    
    output = []
    
    for i in range(numexamples):
    
        tmp = np.zeros((6,6,3))
        tmp_gt_exp = np.zeros((6,6,3))
        tmp_gt_concepts = np.zeros((5,6,6,3)).astype(bool) 
    
        if id == 0:
        
        
            items0 = [random.choice([0,1])]
            items1 = [random.choice([0,1])]
            items = list(set(items0+items1))
            
         

            if 0 in items:
                tmp[0:3,0:3,0], tmp_gt_exp[0:3,0:3,0] = gen_concept_part_example(7)
                tmp[0:3,3:,0],tmp_gt_exp[0:3,3:,0] = gen_concept_part_example(10)

            if 1 in items:
                tmp[3:,0:3,0],tmp_gt_exp[3:,0:3,0] = gen_concept_part_example(4)
                tmp[3:,3:,0],tmp_gt_exp[3:,3:,0] = gen_concept_part_example(1)
                
            tmp_gt_exp = tmp_gt_exp/len(items)
            tmp_gt_exp = tmp_gt_exp/len(items)
              

        if id == 1:
            tmp[0:3,0:3,0],tmp_gt_exp[0:3,0:3,0] = gen_concept_part_example(6)
            tmp[0:3,3:,0],tmp_gt_exp[0:3,3:,0] = gen_concept_part_example(9)
            tmp[3:,0:3,0],tmp_gt_exp[3:,0:3,0] = gen_concept_part_example(3)
            tmp[3:,3:,0],tmp_gt_exp[3:,3:,0] = gen_concept_part_example(0)
      
        if id == 2:
            
            items0 = [random.choice([0,1])]
            items1 = [random.choice([0,1])]
            items = list(set(items0+items1))
            
            tmp[3:,0:3,2],tmp_gt_exp[3:,0:3,2] = gen_concept_part_example(4)
            tmp[3:,3:,2],tmp_gt_exp[3:,3:,2] = gen_concept_part_example(1)
            
            if 0 in items:
                tmp[0:3,0:3,1],tmp_gt_exp[0:3,0:3,1] = gen_concept_part_example(7)

            if 1 in items:
                tmp[0:3,3:,1],tmp_gt_exp[0:3,3:,1] = gen_concept_part_example(10)
                
            tmp_gt_exp[0:3,0:3,1] = tmp_gt_exp[0:3,0:3,1]/len(items)
            tmp_gt_exp[0:3,3:,1] = tmp_gt_exp[0:3,3:,1]/len(items)
              
                
        if id == 3: 
            tmp[0:3,0:3,1], tmp_gt_exp[0:3,0:3,1] = gen_concept_part_example(8)
            tmp[0:3,3:,1], tmp_gt_exp[0:3,3:,1] = gen_concept_part_example(11)
            
        if id == 4:
            if random.uniform(0, 1) > 0.5:
                img_0, exp_0 = gen_concept_part_example(8) 
                img_1, exp_1 = gen_concept_part_example(11) 
                tmp[0:3,0:3,1] = img_0
                tmp_gt_exp[0:3,0:3,1] = exp_0
                tmp_gt_exp[0:3,3:,1] = calc_attr_neg_concept(img_0,img_1,exp_1)
            else:
                img_0, exp_0  = gen_concept_part_example(11) 
                img_1, exp_1 = gen_concept_part_example(8) 
                tmp[0:3,3:,1] = img_0
                tmp_gt_exp[0:3,3:,1] = exp_0
                tmp_gt_exp[0:3,0:3,1] = calc_attr_neg_concept(img_0,img_1,exp_1)

        tmp_gt_concepts[id,:,:,:] = tmp_gt_exp.astype(bool)

        output.append([tmp, tmp_gt_exp, tmp_gt_concepts])
        
    return output

def cncpt(arr):
   
    return gen_concept_examples(random.choice(arr), 1)[0]



def gen_class_examples(conceptid, classid, numexamples):

    output = []
    gt_exp = []
    gt_concepts = []
    
    c = [0,1,2,3,4] # concepten
    c_id_only = [conceptid]
    c_expluding = [0,1,2,3,4]
    c_expluding.pop(conceptid)
    

        
    for i in range(numexamples):
    
        tmp = np.zeros((6*3,6*3,3))
        tmp_gt_exp = np.zeros((6*3,6*3,3))
        tmp_gt_concepts = np.zeros((5,6*3,6*3,3))
    
        if classid % 5 == 0:
        
            tmp[0:6,0:6,:], tmp_gt_exp[0:6,0:6,:], tmp_gt_concepts[:,0:6,0:6,:] = calc_attr_pos_concept(cncpt(c_id_only))
            tmp[6:12,0:6,:],_,_ = cncpt(c_expluding)
            tmp[12:,0:6,:],_,_ = cncpt(c)

            tmp[0:6,6:12,:],_,_ = cncpt(c_expluding)
            tmp[6:12,6:12,:],_,_ = cncpt(c)
            tmp[12:,6:12,:],_,_ = cncpt(c)

            tmp[0:6,12:,:],_,_ = cncpt(c_expluding)
            tmp[6:12,12:,:],_,_ = cncpt(c)
            tmp[12:,12:,:],_,_ = cncpt(c)

        if classid % 5 == 1:
     
            tmp[0:6,0:6,:],_ ,_ = cncpt(c_expluding)
            tmp[6:12,0:6,:],tmp_gt_exp[6:12,0:6,:],tmp_gt_concepts[:,6:12,0:6,:] = calc_attr_pos_concept(cncpt(c_id_only))
            tmp[12:,0:6,:],_,_= cncpt(c)

            tmp[0:6,6:12,:],_,_ = cncpt(c_expluding)
            tmp[6:12,6:12,:],_,_ = cncpt(c)
            tmp[12:,6:12,:],_,_ = cncpt(c)

            tmp[0:6,12:,:],_,_ = cncpt(c_expluding)
            tmp[6:12,12:,:],_,_ = cncpt(c)
            tmp[12:,12:,:],_,_ = cncpt(c)
            
                
        if classid % 5 == 2:

            tmp[0:6,0:6,:],_,_ = cncpt(c_expluding)
            tmp[6:12,0:6,:],_,_ = cncpt(c_expluding)
            tmp[12:,0:6,:],_,_ = cncpt(c)

            tmp[0:6,6:12,:],tmp_gt_exp[0:6,6:12,:],tmp_gt_concepts[:,0:6,6:12,:] = calc_attr_pos_concept(cncpt(c_id_only))
            tmp[6:12,6:12,:],_,_ = cncpt(c)
            tmp[12:,6:12,:],_ ,_ = cncpt(c)

            tmp[0:6,12:,:],tmp_gt_exp[0:6,12:,:],tmp_gt_concepts[:,0:6,12:,:] = calc_attr_pos_concept(cncpt(c_id_only))
            tmp[6:12,12:,:],_,_ = cncpt(c)
            tmp[12:,12:,:],_,_= cncpt(c)
       
        if classid % 5 == 3:
            if random.uniform(0, 1) > 0.5:
                   
                tmp[0:6,0:6,:],_,_ = cncpt(c_expluding)
                tmp[6:12,0:6,:],_,_ = cncpt(c_expluding)
                tmp[12:,0:6,:],_,_ = cncpt(c)

                img_0, exp_0, _ =  cncpt(c_expluding)
                img_1, exp_1, tmp_gt_concepts[:,0:6,6:12,:] = cncpt(c_id_only) 
                tmp[0:6,6:12,:] = img_0
                tmp_gt_exp[0:6,6:12,:] = calc_attr_neg_concept(img_0,img_1,exp_1)
                
                
                tmp[6:12,6:12,:],_ ,_ = cncpt(c)
                tmp[12:,6:12,:],_,_ = cncpt(c)

                tmp[0:6,12:,:],tmp_gt_exp[0:6,12:,:],tmp_gt_concepts[:,0:6,12:,:] = calc_attr_pos_concept(cncpt(c_id_only))
                tmp[6:12,12:,:],_,_ = cncpt(c)
                tmp[12:,12:,:],_ ,_ =cncpt(c)

            else:

                tmp[0:6,0:6,:],_,_ = cncpt(c_expluding)
                tmp[6:12,0:6,:],_,_ = cncpt(c_expluding)
                tmp[12:,0:6,:],_ ,_ = cncpt(c)

                tmp[0:6,6:12,:],tmp_gt_exp[0:6,6:12,:],tmp_gt_concepts[:,0:6,6:12,:] = calc_attr_pos_concept(cncpt(c_id_only))
                tmp[6:12,6:12,:],_,_ = cncpt(c)
                tmp[12:,6:12,:],_,_ = cncpt(c)

                img_0, exp_0, _ = cncpt(c_expluding)
                img_1, exp_1, tmp_gt_concepts[:,0:6,12:,:] = cncpt(c_id_only)        
                tmp[0:6,12:,:] = img_0
                tmp_gt_exp[0:6,12:,:] = calc_attr_neg_concept(img_0,img_1,exp_1)
                 
                tmp[6:12,12:,:],_,_= cncpt(c)
                tmp[12:,12:,:],_ ,_ = cncpt(c)

        if classid % 5 == 4:
   
            img_0,exp_0, _ = cncpt(c_expluding)
            img_1, exp_1, tmp_gt_concepts[:,0:6,0:6,:] = cncpt(c_id_only)    
            tmp[0:6,0:6,:] = img_0
            tmp_gt_exp[0:6,0:6,:] = calc_attr_neg_concept(img_0,img_1,exp_1)
            
            
            img_0, exp_0, _ = cncpt(c_expluding)
            img_1, exp_1, tmp_gt_concepts[:,6:12,0:6,:] = cncpt(c_id_only)     
            tmp[6:12,0:6,:] = img_0
            tmp_gt_exp[6:12,0:6,:] = calc_attr_neg_concept(img_0,img_1,exp_1)
            
            
            tmp[12:,0:6,:],_ ,_ = cncpt(c)
            
            
            img_0, exp_0, _ = cncpt(c_expluding)
            img_1, exp_1, tmp_gt_concepts[:,0:6,6:12,:] = cncpt(c_id_only)    
            tmp[0:6,6:12,:] = img_0
            tmp_gt_exp[0:6,6:12,:] = calc_attr_neg_concept(img_0,img_1,exp_1)
            
            
            tmp[6:12,6:12,:],_,_ = cncpt(c)
            tmp[12:,6:12,:],_,_ = cncpt(c)

            img_0, exp_0, _ = cncpt(c_expluding)
            img_1, exp_1, tmp_gt_concepts[:,0:6,12:,:] = cncpt(c_id_only)     
            tmp[0:6,12:,:] = img_0
            tmp_gt_exp[0:6,12:,:] = calc_attr_neg_concept(img_0,img_1,exp_1)
            
            
            tmp[6:12,12:,:],_,_ = cncpt(c)
            tmp[12:,12:,:],_,_ = cncpt(c)

        output.append(tmp)
        
        gt_exp.append(tmp_gt_exp)

        gt_concepts.append(tmp_gt_concepts.astype(bool))
            
    return np.array(output), np.array(gt_exp), np.array(gt_concepts)










def convert_to_2D(explanation): 
    output = np.zeros((explanation.shape[0], explanation.shape[1]))
    for y in range(explanation.shape[0]):
        for x in range(explanation.shape[1]):
            mostExtremeValue = 0.0

            for c in range(3):
                if abs(explanation[y,x,c]) > abs(mostExtremeValue):
                    if mostExtremeValue != 0.0:
                        print("ERROR!!!!! mostExtremeValue != 0")
                        exit()
                    mostExtremeValue = explanation[y,x,c]
                    
            output[y,x] = mostExtremeValue
            
    return output


def special_upscale(img,gt):
    new_img = np.zeros((36,36,3))
    new_gt = np.zeros((36,36,3))
    for y in range(18):
        for x in range(18):
            
            r_0 = random.randint(0, 1)
            r_1 = random.randint(0, 1)
                        
            new_img[y*2+r_0,x*2+r_1,:] = img[y,x,:]
            new_gt[y*2+r_0,x*2+r_1,:] = gt[y,x,:]

    return new_img, new_gt

def generate(num_examples_per_class, conceptid):
    

    data_x_test, data_y_test, data_gt_explanation_test3D, data_gt_explanation_concepts_test3D, data_gt_residual_test3D , data_gt_explanation_test, data_gt_explanation_concepts_test, data_gt_residual_test = [],[],[],[],[],[],[],[]
    
    for classid in range(5):
    
        # Filter out ambiguous GTs
        if conceptid == 0 or conceptid == 2 or conceptid == 4:
            if classid == 3 or classid == 4:
                continue
              
        examples, data_gt_explanation, data_gt_concepts = gen_class_examples(conceptid, classid, num_examples_per_class)
        
        vec = np.zeros(5)
        vec[classid] = 1
        
        
        # normalize explanations
        for i in range(len(data_gt_explanation)):
            data_gt_explanation[i] = normalize_explanation(data_gt_explanation[i]) 
        
        
        for example, explanation, concepts in zip(examples, data_gt_explanation, data_gt_concepts):
        
            example, explanation = special_upscale(example, explanation)
        
            data_x_test.append(example)
            
            data_y_test.append(vec)
            
            
            data_gt_explanation_test3D.append(explanation)
            data_gt_explanation_test.append(convert_to_2D(explanation))
            
            
            tmp = np.zeros((36,36,3)).astype(bool)
            for x in range(36):
                for y in range(36):
                    for c in range(3):
                        if explanation[y,x,c] == 0:
                            tmp[y,x,c] = True
            data_gt_residual_test3D.append(tmp)
            data_gt_residual_test.append(np.amin(tmp,axis=-1)) # boolean, else convert_to_2D(explanation) would have to be used


            data_gt_explanation_concepts_test3D.append(concepts)
            data_gt_explanation_concepts_test.append(np.amax(concepts,axis=-1)) # boolean, else convert_to_2D(explanation) would have to be used
           
            

        
    data_x_test = np.array(data_x_test)
    data_y_test = np.array(data_y_test)
    data_gt_explanation_test = np.array(data_gt_explanation_test)
    data_gt_explanation_concepts_test = np.array(data_gt_explanation_concepts_test)
    data_gt_residual_test = np.array(data_gt_residual_test)
    
    data_gt_residual_test3D = np.array(data_gt_residual_test3D)
    data_gt_explanation_concepts_test3D = np.array(data_gt_explanation_concepts_test3D)
    data_gt_explanation_test3D = np.array(data_gt_explanation_test3D)
    
    
    return {'test': {'x':data_x_test, 'y':data_y_test, 'gt_explanation':data_gt_explanation_test, 'gt_explanation_concepts':data_gt_explanation_concepts_test, 'gt_residual': data_gt_residual_test,'gt_residual3D':data_gt_residual_test3D,'gt_explanation3D':data_gt_explanation_test3D,'gt_explanation_concepts3D':data_gt_explanation_concepts_test3D}}
   
