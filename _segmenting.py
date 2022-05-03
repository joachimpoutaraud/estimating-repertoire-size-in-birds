### 2.SEGMENTING BIRD SYLLABLES WITH WAVELETS

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy
from scipy.signal import find_peaks, butter, lfilter
import pywt
import librosa
import warnings
warnings.filterwarnings('ignore')


### COMPUTING CWT COEFFICIENTS AND RMS ENERGY
def CWT(signal, sr=22050, nv=12, low_freq=2000):
    # Original scale function by Alexander Neergaard
    n = signal.size
    ds = 1 / nv
    # Smallest useful scale for Morlet wavelet
    s0 = 2
    # Determine longest useful scale for wavelet
    max_scale = n // (np.sqrt(2) * s0)
    if max_scale <= 1:
        max_scale = n // 2
    max_scale = np.floor(nv * np.log2(max_scale)) 
    a0 = 2 ** ds
    scales = s0 * a0 ** np.arange(0, max_scale + 1)
    
    # Filter out scales below low_freq
    fourier_factor = 6 / (2 * np.pi)
    frequencies = sr * fourier_factor / scales
    frequencies = frequencies[frequencies >= low_freq]
    scales = scales[0:len(frequencies)]

    # Compute Continuous Wavelet Transform (CWT)
    coef, freq = pywt.cwt(signal, scales, wavelet='morl')
    return coef, freq

def calculate_rms(cs, threshold_db):
    coefs = 20*np.log10(np.abs(cs))
    # Mask coefficient under threshold
    coefs[coefs < threshold_db] = 0
    # Calculate RMS
    coefs_rms = np.nanmean(np.sqrt(coefs**2), axis=0)
    return coefs_rms / max(coefs_rms)

def thresholding(rms, frame_length=1024):
    # Find high energy peaks
    peaks, _ = find_peaks(rms, prominence=0.3)
    threshold = np.zeros(len(rms))
    if len(peaks > 0):
        for i in range(len(peaks)):
            threshold[max(peaks[i]-frame_length, 0): min(peaks[i]+frame_length, len(rms))] = 1
    return threshold

### SEGMENTING
# Segment bird syllables with a minimum duration of 0.5 second
def segmenting(threshold, filename, min_duration=0.5, sr=22050):
    # Apply threshold mask to segment syllables
    segment = scipy.ndimage.find_objects(scipy.ndimage.label(threshold)[0])
    segments = []
    for r in segment:
        duration = round((r[0].stop-r[0].start)/sr, 3)
        if duration < min_duration:
            pass
        else:
            segments.append([filename, r[0].start, r[0].stop, duration])
    return segments

def segmenting_syllables(filename, fmin=2000, min_duration=0.5, threshold_db=-20, sr=22050):   
    # Load and filter audio file
    y, sr = librosa.load(filename, sr=sr, mono=True, offset=0)
    # Calculate CWT coefficients
    coefcwt = CWT(y, sr=sr)[0]
    # Calculate the root mean square of the CWT coefficients  
    rms = calculate_rms(coefcwt, threshold_db)
    # Define threshold for the root mean square values 
    threshold = thresholding(rms)    
    # Get the segmented syllables
    syllables = segmenting(threshold, filename, min_duration, sr)    
    return syllables

def syllables_to_csv(df, recordings_path, csv_name):    
    # Making sure to keep only the columns of interest
    df = df[['id','gen','en','cnt','type','file-name']]
    # Create new dataframe
    data = []
    new_df = pd.DataFrame(data, columns=['id','gen','en','cnt','type','file-name','start','end','duration'])
    columns = list(new_df)
    
    for i in tqdm(range(len(df.index))):
        
        filename = recordings_path +'/'+ df['file-name'][i]
        syllables = segmenting_syllables(filename)
               
        for x in syllables:
            row = df.loc[i, :].values.tolist()[:-1] + x
            zipped = zip(columns, row)
            dictionary = dict(zipped)
            data.append(dictionary)            
    
    df = new_df.append(data, True)
    df.to_csv(csv_name, index=False)
    return df

if __name__ == "__main__":
    
    df = pd.read_csv('dataset.csv')
    # Convert the segmented syllables and update dataset.csv file
    syllables_to_csv(df, 'data/recordings', 'dataset.csv')









