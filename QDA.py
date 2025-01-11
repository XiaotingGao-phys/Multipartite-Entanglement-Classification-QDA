import numpy as np

def data_aug_perm_3mode(inputs, outputs, j):
    if inputs.shape[0] == outputs.shape[0]:
        inputs_new = np.zeros(inputs.shape)
        outputs_new = np.copy(outputs)
        idx = [[0,1],[1,2],[0,2]]
        for i in range(outputs.shape[0]):
            subarrays = np.array([inputs[i, 0:4, :, :], inputs[i, 4:8, :, :], inputs[i, 8:12, :, :]])
            subarrays_copy = np.copy(subarrays)
            subarrays[idx[j][0],:,:,:], subarrays[idx[j][1],:,:,:] = subarrays_copy[idx[j][1],:,:,:], subarrays_copy[idx[j][0],:,:,:]
            if j == 1:
                inputs_new[i] = np.concatenate(subarrays, axis=0)
            else:
                subarrays_copy = np.copy(subarrays)
                subarrays[idx[j][0],:,:,:] = np.stack([np.flipud(np.flipud(subarrays[idx[j][0],k,:,:]).T) for k in range(4)], axis=0)
                subarrays[idx[j][1],:,:,:] = np.stack([np.flipud(np.flipud(subarrays[idx[j][1],k,:,:]).T) for k in range(4)], axis=0)
                subarrays_copy[idx[j][0],:,:,:] = np.stack([np.flipud(np.flipud(subarrays_copy[idx[j][0],k,:,:]).T) for k in range(4)], axis=0)
                subarrays_copy[idx[j][1],:,:,:] = np.stack([np.flipud(np.flipud(subarrays_copy[idx[j][1],k,:,:]).T) for k in range(4)], axis=0)
                subarrays[idx[j][0],1,:,:], subarrays[idx[j][0],2,:,:] = subarrays_copy[idx[j][0],2,:,:], subarrays_copy[idx[j][0],1,:,:]
                subarrays[idx[j][1],1,:,:], subarrays[idx[j][1],2,:,:] = subarrays_copy[idx[j][1],2,:,:], subarrays_copy[idx[j][1],1,:,:]
                inputs_new[i] = np.concatenate(subarrays, axis=0)
            
            outputs_new = outputs
        return inputs_new, outputs_new
    else:
        print("Length of inputs and outputs doesn't match.")
        
def data_aug_perm_4mode(inputs, outputs, j):
    if inputs.shape[0] == outputs.shape[0]:
        inputs_new = np.zeros(inputs.shape)
        outputs_new = np.copy(outputs)
        idx = [[0,1],[1,2],[0,2],[0,3],[1,3],[2,3]]
        for i in range(outputs.shape[0]):
            subarrays = np.array([inputs[i, 0:4, :, :], inputs[i, 4:8, :, :], inputs[i, 8:12, :, :], inputs[i, 12:16, :, :]])
            subarrays_copy = np.copy(subarrays)
            subarrays[idx[j][0],:,:,:], subarrays[idx[j][1],:,:,:] = subarrays_copy[idx[j][1],:,:,:], subarrays_copy[idx[j][0],:,:,:]
            if j == 1 or j == 4 or j == 5:
                inputs_new[i] = np.concatenate(subarrays, axis=0)
            else:
                subarrays_copy = np.copy(subarrays)
                subarrays[idx[j][0],:,:,:] = np.stack([np.flipud(np.flipud(subarrays[idx[j][0],k,:,:]).T) for k in range(4)], axis=0)
                subarrays[idx[j][1],:,:,:] = np.stack([np.flipud(np.flipud(subarrays[idx[j][1],k,:,:]).T) for k in range(4)], axis=0)
                subarrays_copy[idx[j][0],:,:,:] = np.stack([np.flipud(np.flipud(subarrays_copy[idx[j][0],k,:,:]).T) for k in range(4)], axis=0)
                subarrays_copy[idx[j][1],:,:,:] = np.stack([np.flipud(np.flipud(subarrays_copy[idx[j][1],k,:,:]).T) for k in range(4)], axis=0)
                subarrays[idx[j][0],1,:,:], subarrays[idx[j][0],2,:,:] = subarrays_copy[idx[j][0],2,:,:], subarrays_copy[idx[j][0],1,:,:]
                subarrays[idx[j][1],1,:,:], subarrays[idx[j][1],2,:,:] = subarrays_copy[idx[j][1],2,:,:], subarrays_copy[idx[j][1],1,:,:]
                inputs_new[i] = np.concatenate(subarrays, axis=0)
            
            outputs_new = outputs
        return inputs_new, outputs_new
    else:
        print("Length of inputs and outputs doesn't match.")
    
def data_aug_add(inputs, outputs, idx, num_aug):
    if inputs.shape[0] == outputs.shape[0]:
        idx_aug = np.where(np.all(outputs==idx, axis=1))[0]
        inputs_new = np.zeros([num_aug,12,24,24])
        for i in range(num_aug):
            prop = np.random.random(1)*0.5
            j = np.random.choice(idx_aug, [num_aug, 2])
            inputs_new[i] = inputs[j[i,0]]*prop + inputs[j[i,1]]*(1-prop)
        return inputs_new, np.tile(idx, [num_aug,1])
    else:
        print("Length of inputs and outputs doesn't match.")
        
def TripartiteQDA(inputs, outputs):
    
    inputs_raw = np.copy(inputs)
    outputs_raw = np.copy(outputs)
    
    # 2 time
    inputs_new, outputs_new = data_aug_perm_3mode(inputs_raw, outputs_raw, 0)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    # 3 times
    inputs_new, outputs_new = data_aug_perm_3mode(inputs_raw, outputs_raw, 1)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
  
    # 4 times
    inputs_new, outputs_new = data_aug_perm_3mode(inputs_raw, outputs_raw, 2)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    # 5 times
    inputs_new, outputs_new = data_aug_perm_3mode(inputs_raw, outputs_raw, 0)
    inputs_new, outputs_new = data_aug_perm_3mode(inputs_new, outputs_new, 1)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    # 6 times
    inputs_new, outputs_new = data_aug_perm_3mode(inputs_raw, outputs_raw, 0)
    inputs_new, outputs_new = data_aug_perm_3mode(inputs_new, outputs_new, 2)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    return inputs, outputs
    
def QuadripartiteQDA(inputs, outputs):
    
    inputs_raw = np.copy(inputs)
    outputs_raw = np.copy(outputs)
    
    #6times
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 0)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 1)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 2)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 3)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 4)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    print(inputs.shape, outputs.shape)
    
    # 12times
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 5)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 0)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 1)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 0)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 4)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 0)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 5)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 1)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 0)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 1)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 3)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    print(inputs.shape, outputs.shape)
    
    # 18times
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 1)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 5)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 2)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 3)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 2)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 4)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 2)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 5)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 3)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 4)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 4)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 5)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    print(inputs.shape, outputs.shape)
    
    # 24times
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 0)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 2)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 3)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 3)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 2)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 3)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 0)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 4)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 5)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 1)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 0)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 5)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 1)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 0)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 4)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_raw, outputs_raw, 3)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 4)
    inputs_new, outputs_new = data_aug_perm_4mode(inputs_new, outputs_new, 2)
    inputs = np.append(inputs, inputs_new, axis=0)
    outputs = np.append(outputs, outputs_new, axis=0)
    
    return inputs, outputs
