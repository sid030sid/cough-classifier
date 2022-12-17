import numpy as np

#Use old segmentation
def segment_cough(x,fs, cough_padding=0.2,min_cough_len=0.2, th_l_multiplier = 0.1, th_h_multiplier = 2):
    """Preprocess the data by segmenting each file into individual coughs using a hysteresis comparator on the signal power
    
    Inputs:
    *x (np.array): cough signal
    *fs (float): sampling frequency in Hz
    *cough_padding (float): number of seconds added to the beginning and end of each detected cough to make sure coughs are not cut short
    *min_cough_length (float): length of the minimum possible segment that can be considered a cough
    *th_l_multiplier (float): multiplier of the RMS energy used as a lower threshold of the hysteresis comparator
    *th_h_multiplier (float): multiplier of the RMS energy used as a high threshold of the hysteresis comparator
    
    Outputs:
    *coughSegments (np.array of np.arrays): a list of cough signal arrays corresponding to each cough
    cough_mask (np.array): an array of booleans that are True at the indices where a cough is in progress"""
                
    cough_mask = np.array([False]*len(x))
    

    #Define hysteresis thresholds
    rms = np.sqrt(np.mean(np.square(x)))
    seg_th_l = th_l_multiplier * rms
    seg_th_h =  th_h_multiplier*rms

    #Segment coughs
    coughSegments = []
    padding = round(fs*cough_padding)
    min_cough_samples = round(fs*min_cough_len)
    cough_start = 0
    cough_end = 0
    cough_in_progress = False
    tolerance = round(0.01*fs)
    below_th_counter = 0
    
    segment_indices = []
    
    for i, sample in enumerate(x**2):
        if cough_in_progress:
            if sample<seg_th_l:
                below_th_counter += 1
                if below_th_counter > tolerance:
                    cough_end = i+padding if (i+padding < len(x)) else len(x)-1
                    cough_in_progress = False
                    if (cough_end+1-cough_start-2*padding>min_cough_samples):
                        coughSegments.append(x[cough_start:cough_end+1])
                        segment_indices.append((cough_start,cough_end))
                        cough_mask[cough_start:cough_end+1] = True
            elif i == (len(x)-1):
                cough_end=i
                cough_in_progress = False
                if (cough_end+1-cough_start-2*padding>min_cough_samples):
                    coughSegments.append(x[cough_start:cough_end+1])
                    segment_indices.append((cough_start,cough_end))
            else:
                below_th_counter = 0
        else:
            if sample>seg_th_h:
                cough_start = i-padding if (i-padding >=0) else 0
                cough_in_progress = True
    
    starts = np.zeros(len(segment_indices))
    for i, ndx in enumerate(segment_indices):
        starts[i] = ndx[0]
    ends = np.zeros(len(segment_indices))
    for i, ndx in enumerate(segment_indices):
        ends[i] = ndx[1]
    peaks = []
    peak_locs = []
    for s, e in zip(starts,ends):
        sig = x[round(s):round(e)]
        pk = np.max(sig)
        loc = np.argmax(sig)
        peaks.append(pk)
        peak_locs.append(round(s)+loc)
    
    return coughSegments, cough_mask, starts, ends, peaks, peak_locs


def compute_SNR(x, cough_mask):
    """Compute the Signal-to-Noise ratio of the audio signal x (np.array) with sampling frequency fs (float)"""
    cough_idx = np.where(cough_mask)[0]
    start = cough_idx[0]
    end = cough_idx[-1]
    cough_sig = x[cough_mask]
    short_sig = x[start:end]
    short_mask = cough_mask[start:end]
    non_cough_sig = short_sig[~short_mask]
    RMS_signal = 0 if len(x[cough_mask])==0 else np.sqrt(np.mean(np.square(cough_sig)))
    RMS_noise = np.sqrt(np.mean(np.square(non_cough_sig)))
    SNR = 0 if (RMS_signal==0 or np.isnan(RMS_noise)) else 20*np.log10(RMS_signal/RMS_noise)
    return SNR