import h5py
import numpy as np
import os

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
os.chdir(current_dir)

def TripartiteData(num_states=30000, include_mc=False):

    inputfs = np.array(h5py.File("3mode/state-3mode10000-inputs-111.h5", 'r')["inputs"])
    inputbi = np.array(h5py.File("3mode/state-3mode10000-inputs-12.h5", 'r')["inputs"])
    inputfe = np.array(h5py.File("3mode/state-3mode10000-inputs-3.h5", 'r')["inputs"])

    outputfs = np.tile([1, 0, 0], (inputfs.shape[0], 1))
    outputbi = np.tile([0, 1, 0], (inputbi.shape[0], 1))
    outputfe = np.tile([0, 0, 1], (inputfe.shape[0], 1))

    inputs_all = np.append(np.append(inputfs, inputbi, axis=0), inputfe, axis=0)
    outputs_all = np.append(np.append(outputfs, outputbi, axis=0), outputfe, axis=0)

    Ntrain = int(min(num_states, inputs_all.shape[0]) * 0.8)
    randomize = np.arange(inputs_all.shape[0])
    np.random.shuffle(randomize)
    inputs = inputs_all[randomize[0:Ntrain]]
    outputs = outputs_all[randomize[0:Ntrain]]
    
    inputs_test = inputs_all[randomize[Ntrain:]]
    outputs_test = outputs_all[randomize[Ntrain:]]

    if include_mc:
        inputfs_mc = np.array(h5py.File("3mode/MC data/state-3mode100-inputs-111-MC100000.h5", 'r')["inputs"])
        inputbi_mc = np.array(h5py.File("3mode/MC data/state-3mode100-inputs-12-MC100000.h5", 'r')["inputs"])
        inputfe_mc = np.array(h5py.File("3mode/MC data/state-3mode100-inputs-3-MC100000.h5", 'r')["inputs"])

        outputfs_mc = np.tile([1, 0, 0], (inputfs_mc.shape[0], 1))
        outputbi_mc = np.tile([0, 1, 0], (inputbi_mc.shape[0], 1))
        outputfe_mc = np.tile([0, 0, 1], (inputfe_mc.shape[0], 1))
        
        inputfs_mc_test = np.array(h5py.File("3mode/MC data/state-3mode33-inputs-111-MC100000.h5", 'r')["inputs"])
        inputbi_mc_test = np.array(h5py.File("3mode/MC data/state-3mode34-inputs-12-MC100000.h5", 'r')["inputs"])
        inputfe_mc_test = np.array(h5py.File("3mode/MC data/state-3mode33-inputs-3-MC100000.h5", 'r')["inputs"])

        outputfs_mc_test = np.tile([1, 0, 0], (inputfs_mc.shape[0], 1))
        outputbi_mc_test = np.tile([0, 1, 0], (inputbi_mc.shape[0], 1))
        outputfe_mc_test = np.tile([0, 0, 1], (inputfe_mc.shape[0], 1))

        inputs = np.append(np.append(np.append(inputfs_mc, inputbi_mc, axis=0), inputfe_mc, axis=0), inputs, axis=0)
        outputs = np.append(np.append(np.append(outputfs_mc, outputbi_mc, axis=0), outputfe_mc, axis=0), outputs, axis=0)

        inputs_test = np.append(np.append(np.append(inputfs_mc_test, inputbi_mc_test, axis=0), inputfe_mc_test, axis=0), inputs_test, axis=0)
        outputs_test = np.append(np.append(np.append(outputfs_mc_test, outputbi_mc_test, axis=0), outputfe_mc_test, axis=0), outputs_test, axis=0)

    return inputs, outputs, inputs_test, outputs_test

def QuadripartiteData(num_states=25000):

    input1 = np.array(h5py.File("4mode/state-4mode5000-inputs-1111.h5", 'r')["inputs"])
    input2 = np.array(h5py.File("4mode/state-4mode5000-inputs-121.h5", 'r')["inputs"])
    input3 = np.array(h5py.File("4mode/state-4mode5000-inputs-22.h5", 'r')["inputs"])
    input4 = np.array(h5py.File("4mode/state-4mode5000-inputs-13.h5", 'r')["inputs"])
    input5 = np.array(h5py.File("4mode/state-4mode5000-inputs-4.h5", 'r')["inputs"])

    output1 = np.tile([1, 0, 0, 0, 0], (input1.shape[0], 1))
    output2 = np.tile([0, 1, 0, 0, 0], (input2.shape[0], 1))
    output3 = np.tile([0, 0, 1, 0, 0], (input3.shape[0], 1))
    output4 = np.tile([0, 0, 0, 1, 0], (input4.shape[0], 1))
    output5 = np.tile([0, 0, 0, 0, 1], (input5.shape[0], 1))
    
    inputs_all = np.append(np.append(np.append(np.append(input1, input2, axis=0), input3, axis=0), input4, axis=0), input5, axis=0)
    outputs_all = np.append(np.append(np.append(np.append(output1, output2, axis=0), output3, axis=0), output4, axis=0), output5, axis=0)

    Ntrain = int(min(num_states, inputs_all.shape[0]) * 0.8)
    randomize = np.arange(inputs_all.shape[0])
    np.random.shuffle(randomize)
    inputs = inputs_all[randomize[0:Ntrain]]
    outputs = outputs_all[randomize[0:Ntrain]]
    
    inputs_test = inputs_all[randomize[Ntrain:]]
    outputs_test = outputs_all[randomize[Ntrain:]]

    return inputs, outputs, inputs_test, outputs_test

def CatData(num_states=15000):

    inputfs = np.array(h5py.File("CatStates/state-cat-3mode5000-inputs-111.h5", 'r')["inputs"])
    inputbi = np.array(h5py.File("CatStates/state-cat-3mode5000-inputs-12.h5", 'r')["inputs"])
    inputfe = np.array(h5py.File("CatStates/state-cat-3mode5000-inputs-3.h5", 'r')["inputs"])

    outputfs = np.tile([1, 0, 0], (inputfs.shape[0], 1))
    outputbi = np.tile([0, 1, 0], (inputbi.shape[0], 1))
    outputfe = np.tile([0, 0, 1], (inputfe.shape[0], 1))

    inputs_all = np.append(np.append(inputfs, inputbi, axis=0), inputfe, axis=0)
    outputs_all = np.append(np.append(outputfs, outputbi, axis=0), outputfe, axis=0)

    Ntrain = int(min(num_states, inputs_all.shape[0]) * 0.8)
    randomize = np.arange(inputs_all.shape[0])
    np.random.shuffle(randomize)
    inputs = inputs_all[randomize[0:Ntrain]]
    outputs = outputs_all[randomize[0:Ntrain]]
    
    inputs_test = inputs_all[randomize[Ntrain:]]
    outputs_test = outputs_all[randomize[Ntrain:]]

    return inputs, outputs, inputs_test, outputs_test

