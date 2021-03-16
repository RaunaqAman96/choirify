from __future__ import print_function, division, absolute_import, unicode_literals
import six
import argparse
import os
import numpy as np
import Model_RNN
import torch
import util
import sys
import librosa


if __name__=="__main__":
    blockSize = 4096
    hopSize = 2048

    if len(sys.argv) != 3:
        print("Usage:\n", sys.argv[0], "input_path output_path")
        exit(1) 

    #read the wav file
    x, fs = util.wavread(sys.argv[1])
    print(x.shape[0])
    #downmix to single channel
    x = np.mean(x,axis=-1)
    #perform stft
    S = util.stft_real(x, blockSize=  blockSize,hopSize=hopSize)
    magnitude = np.abs(S).astype(np.float32)
    angle = np.angle(S).astype(np.float32)

    #initialize the model
    model = Model_RNN.ModelSingleStep(blockSize)

    #load the pretrained model
    checkpoint = torch.load("savedModel_feedForward_best_RNN.pt", map_location=lambda storage, loc:storage)
    model.load_state_dict(checkpoint['state_dict'])

    #switch to eval mode
    model.eval()    
    ###################################
    #Model to obtain a mask
    ###################################
    output = model.process(magnitude)                   #Feed the spectrogram into the model
    magnitude_masked = np.multiply(magnitude,output)    #Multiply with original spectrogram
    ###################################


    #perform reconstruction
    y1 = util.istft_real(magnitude_masked* np.exp(1j* angle), blockSize=blockSize, hopSize=hopSize)
    y1 = y1/np.amax(y1)

    y_soprano = librosa.effects.pitch_shift(y1, fs, n_steps=5)
    y_soprano = y_soprano/np.amax(y_soprano)

    
    y_alto = librosa.effects.pitch_shift(y1, fs, n_steps=3)
    y_alto = y_alto/np.amax(y_alto)
 
    
    y_tenor = librosa.effects.pitch_shift(y1, fs, n_steps=-3)
    y_tenor = y_tenor/np.amax(y_tenor)
    
    
    y_bass = librosa.effects.pitch_shift(y1, fs, n_steps=-5)
    y_bass = y_bass/np.amax(y_bass)



    y_1 = y_soprano + y_alto + y_tenor + y_bass
    y_1 = y_1/np.amax(y_1)
    
    #save the result
    util.wavwrite(sys.argv[2], y_1,fs)

