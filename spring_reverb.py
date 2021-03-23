# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


class SpringReverb:
    
    def __init__(self, n, k, r):
        self.n = n ; self.k = k ; self.r = r
        self.ys = np.zeros(n + 2)
        self.dys_dt = np.zeros(n + 2)
        
    def reset(self):
        n = self.n
        self.ys = np.zeros(n + 2)
        self.dys_dt = np.zeros(n + 2)
        
    def step(self, x, dt):
        n = self.n ; k = self.k ; r = self.r  
        ys = self.ys ;dys_dt = self.dys_dt
        ys[0] = x
        dys_dt[1:n+1] += (-dt) * (k * (2.* ys[1:n+1] - ys[0:n] - ys[2:]) + r * dys_dt[1:n+1])
        ys[1:n+1] += dt * dys_dt[1:n+1]
        
    def __call__(self, signal, dt):
        n = self.n
        self.reset()
        output = np.empty((n, len(signal)))
        for i, x in enumerate(signal):
            self.step(x, dt)
            output[:, i] = self.ys[1:n+1]
        return output
    
    
def load_wav(path):
   samplerate, data = wavfile.read(path) 
   left_data = data[:, 0].reshape(-1)
   right_data = data[:, 1].reshape(-1)
   return samplerate, left_data, right_data
   

def write_wav(path, samplerate, left_signal, right_signal, normalize = None):
    
    if normalize is not None:
        max_values = max(np.max(np.abs(left_signal)),
                         np.max(np.abs(right_signal)))
        left_signal /= max_values
        right_signal /= max_values
    
    data = np.empty((len(left_signal), 2))
    data[:, 0] = left_signal
    data[:, 1] = right_signal
    wavfile.write(path, samplerate, data)
    
    
def test():
    
    samplerate, left_signal, right_signal = load_wav('./input3.wav')

    dt = 1. / samplerate
    
    nb_springs = 500
    
    reverb = SpringReverb(n = nb_springs,
                          k = 20000000.,
                          r = 5.)
  
    plt.plot(left_signal[50000:50500])
    
    for i in range(nb_springs):
        plt.plot(reverb(left_signal[50000:50500], dt)[i,:])    
    plt.show()
    
    left_outputs = reverb(left_signal, dt)
    right_outputs = reverb(right_signal, dt)
    
    for i in range(nb_springs):
        write_wav('./outputs/output' + str(i) + '.wav', 
                  samplerate, 
                  left_outputs[i, :], right_outputs[i, :],
                  normalize = 0.75) 
        
if __name__ == '__main__':
    test()    

 
    