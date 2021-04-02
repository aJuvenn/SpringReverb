# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd


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
        
    def __call__(self, signal, dt, only_returned_index = None):
        n = self.n
        self.reset()
        output = np.empty(((n if only_returned_index is None else 1), 
                           len(signal)))
        for i, x in enumerate(signal):
            self.step(x, dt)
            if only_returned_index is None:
                output[:, i] = self.ys[1:n+1]
            else:
                output[0, i] = self.ys[only_returned_index + 1]
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
    
    samplerate, left_signal, right_signal = load_wav('./input4.wav')

    dt = 1. / samplerate
    
    nb_springs = 2500
    
    reverb = SpringReverb(n = nb_springs,
                          k = 100000000.,
                          r = 0.)
  
    PLOT = False
    ONLY_RETURNED_INDEX = 0
    
    if PLOT:
        plt.plot(left_signal[10000:10500])
        
        for i in range(nb_springs):
            plt.plot(reverb(left_signal[10000:10500], dt)[i,:])    
        plt.show()
    
    left_outputs = reverb(left_signal, dt, 
                          only_returned_index = ONLY_RETURNED_INDEX)
    right_outputs = reverb(right_signal, dt,
                           only_returned_index = ONLY_RETURNED_INDEX)
    
    for i in range(nb_springs if ONLY_RETURNED_INDEX is None else 1):
        write_wav('./outputs/output' + str(i) + '.wav', 
                  samplerate, 
                  left_outputs[i, :], right_outputs[i, :],
                  normalize = 0.75) 

        sd.play(np.vstack([left_outputs[i], right_outputs[i]]).T,
                samplerate)
    


NOTES_TO_FREQ = {
        'C' : 261.626,
        'Db' : 277.183,
        'D' : 293.665,
        'Eb' : 311.127,
        'E' : 329.628,
        'F' : 349.228,
        'Gb' : 369.994,
        'G' : 391.995,
        'Ab' : 415.305,
        'A' : 440.,
        'Bb' : 466.164,
        'B' : 493.883
}


def get_frequence(note_name, octave_number = 4):
    return NOTES_TO_FREQ[note_name] * (2.**(octave_number - 4))
    


def save_frequences():
    
    NB_SPRINGS = 500
    RESISTANCE = 1.
    ONLY_RETURNED_INDEX = 0
    INPUT_PATH = './input5.wav'
    OCTAVE_NUMBER = -2
     
    samplerate, left_signal, right_signal = load_wav(INPUT_PATH)
    dt = 1. / samplerate
    
    for note_name in NOTES_TO_FREQ.keys():
        frequence = get_frequence(note_name, OCTAVE_NUMBER)
        k = (2. * (NB_SPRINGS + 1) * frequence) ** 2.
        print("Using k = ", k)
        reverb = SpringReverb(n = NB_SPRINGS,
                              k = k,
                              r = RESISTANCE)

        left_outputs = reverb(left_signal, dt, 
                              only_returned_index = ONLY_RETURNED_INDEX)
        
        right_outputs = reverb(right_signal, dt,
                               only_returned_index = ONLY_RETURNED_INDEX)
        
        output_path = ('./outputs/output' 
                       + str(ONLY_RETURNED_INDEX) 
                       + '_' + note_name + str(OCTAVE_NUMBER) + '.wav')
        
        write_wav(output_path, 
                  samplerate, 
                  left_outputs[0, :], right_outputs[0, :],
                  normalize = 0.95) 

    




def play():
    """
        Formula: f = sqrt(k) / (2 * (n+1))
        Or: k = (2 * (n + 1) * f)**2
    """
    
    
    samplerate, left_signal, right_signal = load_wav('./input2.wav')

    dt = 1. / samplerate
    
    FREQUENCE = 440. / 2.**5.
    NB_SPRINGS = 50
    RESISTANCE = 10.
    ONLY_RETURNED_INDEX = 0
  
    k = (2. * (NB_SPRINGS + 1) * FREQUENCE) ** 2.
    print("Using k = ", k)
    
    reverb = SpringReverb(n = NB_SPRINGS,
                          k = k,
                          r = RESISTANCE)
    
  
    for i in range(1):
        
        left_outputs = reverb(left_signal, dt, 
                              only_returned_index = ONLY_RETURNED_INDEX)
        
        right_outputs = reverb(right_signal, dt,
                               only_returned_index = ONLY_RETURNED_INDEX)
        
        playable_outputs = np.vstack([left_outputs[0], right_outputs[0]]).T
        playable_outputs /= np.max(np.abs(playable_outputs))
        sd.play(playable_outputs, samplerate, blocking = True)

        write_wav('./outputs/output' + str(ONLY_RETURNED_INDEX) + '.wav', 
                  samplerate, 
                  left_outputs[0, :], right_outputs[0, :],
                  normalize = 0.95) 

    
        
        
if __name__ == '__main__':
    save_frequences()    

 
    