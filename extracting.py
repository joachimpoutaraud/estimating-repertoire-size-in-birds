### 3.SEGMENTING BIRD SYLLABLES WITH WAVELETS

import numpy as np
import pandas as pd
from tqdm import tqdm

import scipy
from scipy import fftpack
from scipy.signal import hilbert, lfilter, find_peaks

import pywt
import librosa
import warnings
warnings.filterwarnings('ignore')

### PREPROCESSING
def filtering(signal, wavelet='db6'):
    # Calculate decomposition and reconstruction filter values using Daubechies wavelet
    dec_lo, dec_hi, rec_lo, rec_hi = pywt.Wavelet(wavelet).filter_bank  
    # Apply high-pass decomposition filter along one-dimension
    y = lfilter(dec_hi, 1, signal) 
    return y

def moments(X, axis=0):
    return np.nanmean(X, axis), np.nanvar(X, axis)

if __name__ == "__main__":
    
    # Load csv file
    df = pd.read_csv('dataset.csv')
    # Keep syllables under 3 seconds 
    df = df.loc[df.duration <= 3]
    # Reset the dataframe index
    df = df.reset_index(drop=True)
    
    # Create new columns for the audio features
    ### MFCCs
    df['MFCC1'] = None
    df['MFCC2'] = None
    df['MFCC3'] = None
    df['MFCC4'] = None
    df['MFCC5'] = None
    df['MFCC6'] = None
    df['MFCC7'] = None
    df['MFCC8'] = None
    df['MFCC9'] = None
    df['MFCC10'] = None
    df['MFCC11'] = None
    df['MFCC12'] = None
    
    ### DESCRIPTIVE FEATURES (DF)
    # Time Domain Features (TDF)
    df['ENm'] = None
    df['ENv'] = None
    df['ZCRm'] = None
    df['ZCRv'] = None

    # Frequency Domain Features (FDF)
    df['SCm'] = None
    df['SCv'] = None
    df['SBm'] = None
    df['SBv'] = None
    df['SFm'] = None
    df['SFv'] = None
    df['SRm'] = None
    df['SRv'] = None
    df['SFMm'] = None
    df['SFMv'] = None

    for i in tqdm(range(len(df.index))): 

        # Load audio file 
        y, sr = librosa.load(df['file-name'][i], sr=22050)
        syllable = y[df.start[i]:df.end[i]]
        # Denoise syllable with Daubechies wavelet
        filtered = filtering(syllable)

        ### MFCCs
        mfcc = librosa.feature.mfcc(filtered, n_mfcc=13, n_mels=24, htk=True, n_fft=2048, hop_length=512)
        # Remove higher DCT coefficients because they represent fast changes in the filterbank energies and actually degrade ASR performance
        mfcc = mfcc[1:] 
        mfcc = mfcc.mean(axis=1)

        df['MFCC1'][i] = mfcc[0]
        df['MFCC2'][i] = mfcc[1]
        df['MFCC3'][i] = mfcc[2]
        df['MFCC4'][i] = mfcc[3]
        df['MFCC5'][i] = mfcc[4]
        df['MFCC6'][i] = mfcc[5]
        df['MFCC7'][i] = mfcc[6]
        df['MFCC8'][i] = mfcc[7]
        df['MFCC9'][i] = mfcc[8]
        df['MFCC10'][i] = mfcc[9]
        df['MFCC11'][i] = mfcc[10]
        df['MFCC12'][i] = mfcc[11]

        ### DESCRIPTIVE FEATURES (DF)
        # Time Domain Features (TDF)
        energy = np.abs(hilbert(filtered))
        EN = moments(energy)
        df['ENm'][i] = EN[0]
        df['ENv'][i] = EN[1]

        zcr = librosa.feature.zero_crossing_rate(filtered, frame_length=512, hop_length=256)[0]
        zcr = moments(zcr)
        df['ZCRm'][i] = zcr[0]
        df['ZCRv'][i] = zcr[1]

        # Frequency Domain Features (FDF)
        SC = librosa.feature.spectral_centroid(y=filtered, sr=sr, n_fft=2048, hop_length=512)[0]
        cent = moments(SC)
        df['SCm'][i] = cent[0]
        df['SCv'][i] = cent[1] 

        SB = librosa.feature.spectral_bandwidth(y=filtered, sr=sr,  n_fft=2048, hop_length=512)[0]
        band = moments(SB)
        df['SBm'][i] = band[0]
        df['SBv'][i] = band[1]

        SF = librosa.onset.onset_strength(y=filtered, sr=sr, n_fft=2048, hop_length=512)
        flux = moments(SF)
        df['SFm'][i] = flux[0]
        df['SFv'][i] = flux[1]   

        SR = librosa.feature.spectral_rolloff(y=filtered, sr=sr, n_fft=2048, hop_length=512)[0]
        roll = moments(SR)
        df['SRm'][i] = roll[0]
        df['SRv'][i] = roll[1]

        SFM = librosa.feature.spectral_flatness(y=filtered, n_fft=2048, hop_length=512)[0]
        flat = moments(SFM)
        df['SFMm'][i] = flat[0]
        df['SFMv'][i] = flat[1]

    # Updated and save the dataframe
    df.to_csv('dataset.csv', index=False)
