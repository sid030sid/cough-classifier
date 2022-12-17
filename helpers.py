from DSP import preprocess_cough
from enum import Enum
import os
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd
import numpy as np

FS_AUDIO = 16000
FS_IMU = 100

# Enums for easily accessing files
class Trial(str, Enum):
    # Trial number (1-3) of the experiment on a given subject
    ONE = '1'
    TWO = '2'
    THREE = '3'
    
class Movement(str, Enum):
    # Kinematic noise scenarios
    SIT = 'sit'
    WALK = 'walk'

class Noise(str, Enum):
    # Audio noise scenarios
    MUSIC = 'music'
    NONE = 'nothing'
    COUGH = 'someone_else_cough'
    TRAFFIC = 'traffic'
    
class Sound(str, Enum):
    # Sound that the subject performs
    COUGH = 'cough'
    LAUGH = 'laugh'
    TALK = 'talk'
    BREATH = 'deep_breathing'
    THROAT = 'throat_clearing'

class IMU_Signal(str, Enum):
    x = "Accel X"
    y = "Accel Y"
    z = "Accel Z"
    Y = "Gyro Y"
    P = "Gyro P"
    R = "Gyro R"

class IMU_Short(str, Enum):
    x = "x"
    y = "y"
    z = "z"
    Y = "Y"
    P = "P"
    R = "R"
    
def load_audio(folder, subject_id, trial, mov, noise, sound, start=1, end=0.5):
    
    fn = subject_id + '/' + subject_id + '_' + trial + '/mov_' + mov + '/background_noise_' + noise + '/' + sound + '/'
    
    files = os.listdir(folder + fn)
    audio_air_fn = ''
    audio_skin_fn = ''
    for file in files:
        fn_base = file.split('.')[0]
        if fn_base == 'mic_left':
            audio_skin_fn = file
        elif fn_base == 'mic_right':
            audio_air_fn = file
    try:        
        fs_aa, audio_air = wavfile.read(folder + fn + audio_air_fn)
    except FileNotFoundError as err:
        print("ERROR: Air mic file not found")

    try:        
        fs_as, audio_skin = wavfile.read(folder + fn + audio_skin_fn)
    except FileNotFoundError as err:
        print("ERROR: Skin mic file not found")
    
    if (fs_aa != fs_as):
        print("ERROR: Mismatching sampling rates")
   
    samp_trans = int(start*fs_aa)
    samp_trans_end = -int(end*fs_aa)
    
#     #Normalize recordings to [-1, +1] range
#     audio_air = preprocess_cough(audio_air[samp_trans:samp_trans_end],fs_aa, cutoff = 6000, normalize = True, filter_ = False, downsample = False)[0]
#     audio_skin = preprocess_cough(audio_skin[samp_trans:samp_trans_end],fs_aa, cutoff = 6000, normalize = True, filter_ = False, downsample = False)[0]
    # Normalize recordings based on maximum value
    max_val = 1<<29
    audio_air = audio_air[samp_trans:samp_trans_end]/max_val
    audio_skin = audio_skin[samp_trans:samp_trans_end]/max_val
    
    return fs_aa, audio_air, audio_skin

def get_audio_time(audio_sig):
    return np.arange(0,len(audio_sig)/FS_AUDIO,1/FS_AUDIO)

class IMU:
    fs = 100
    mask = [] #mask indicating time indices at which a cough occurs 
    times = [] #list of arrays of indices of segments of interest (ex. one cough or laugh burst)
    def __init__(self, Y,P,R,x,y,z):
        self.x=x
        self.y=y
        self.z=z
        self.Y=Y
        self.P=P
        self.R=R
    def normalize(self):
        self.x = self.x - np.mean(self.x)
        self.x = self.x/np.max(np.abs(self.x))
        self.y = self.y - np.mean(self.y)
        self.y = self.y/np.max(np.abs(self.y))
        self.z = self.z - np.mean(self.z)
        self.z = self.z/np.max(np.abs(self.z))
        self.Y = self.Y - np.mean(self.Y)
        self.Y = self.Y/np.max(np.abs(self.Y))
        self.P = self.P - np.mean(self.P)
        self.P = self.P/np.max(np.abs(self.P))
        self.R = self.R - np.mean(self.R)
        self.R = self.R/np.max(np.abs(self.R))
    def standardize(self):
        self.x = self.x - np.mean(self.x)
        self.x = self.x/np.std(self.x)
        self.y = self.y - np.mean(self.y)
        self.y = self.y/np.std(self.y)
        self.z = self.z - np.mean(self.z)
        self.z = self.z/np.std(self.z)
        self.Y = self.Y - np.mean(self.Y)
        self.Y = self.Y/np.std(self.Y)
        self.P = self.P - np.mean(self.P)
        self.P = self.P/np.std(self.P)
        self.R = self.R - np.mean(self.R)
        self.R = self.R/np.std(self.R)
    def get_time(self):
        if self.x is not None:
            time = np.arange(0,len(self.x)/self.fs,1/self.fs)
            if len(time) > len(self.x):
                return time[:-1]
            return time
    def plot(self):
        fig, axs = plt.subplots(6,1, figsize=(10,21))
        time = self.get_time()
        axs[0].plot(time,self.x, label='Accel X')
        axs[0].set_title("Accel X")
        axs[1].plot(time,self.y, label='Accel Y')
        axs[1].set_title("Accel Y")
        axs[2].plot(time,self.z, label='Accel Z')
        axs[2].set_title("Accel Z")
        axs[3].plot(time,self.Y, label='Gyro Y')
        axs[3].set_title("Gyro Y")
        axs[4].plot(time,self.P, label='Gyro P')
        axs[4].set_title("Gyro P")
        axs[5].plot(time,self.R, label='Gyro R')
        axs[5].set_title("Gyro R")
        axs[5].set_xlabel("Time (s)")
    def set_fs(self,fs_new):
        fs=fs_new
    def get_segments_and_snr(self,audio_segments):
        imu_time = self.get_time()
        self.mask = np.zeros(imu_time.size)
        imu_tolerance = 0.09
        cough_times = [];
        for seg in audio_segments:
            seg = np.array(seg)/fs
            cough_time = np.where((imu_time >= (seg[0] - imu_tolerance)) & (imu_time <= (seg[1] + imu_tolerance)))
            cough_times.append(cough_time)
            self.mask[cough_time] = True
#         snrs = self.get_snr()
        self.times = cough_times
#         return self.mask, cough_times, snrs
        return self.mask, cough_times
    def make_segment_df(self, time):
        df_cough = pd.DataFrame({})
        df_cough['x'] = self.x[time]
        df_cough['y'] = self.y[time]
        df_cough['z'] = self.z[time]
        df_cough['Y'] = self.Y[time]
        df_cough['P'] = self.P[time]
        df_cough['R'] = self.R[time]
        return df_cough
    
    def get_snr(self):
        snr_x = compute_SNR(self.x, self.mask==1)
        snr_y = compute_SNR(self.y, self.mask==1)
        snr_z = compute_SNR(self.z, self.mask==1)
        snr_Y = compute_SNR(self.Y, self.mask==1)
        snr_P = compute_SNR(self.P, self.mask==1)
        snr_R = compute_SNR(self.R, self.mask==1)
        return np.array([snr_x, snr_y, snr_z, snr_Y, snr_P, snr_R])

    
def delineate_imu(imu):
    imu_z = -imu.z
    deriv_imu = np.gradient(imu_z)
    fs_downsample = 1000
    b, a = butter(4, fs_downsample/FS_AUDIO, btype='lowpass') # 4th order butter lowpass filter
    deriv_imu_filt = filtfilt(b, a, deriv_imu)
    deriv_imu_filt = deriv_imu_filt/np.max(np.abs(deriv_imu_filt))
    deriv_imu = deriv_imu/np.max(np.abs(deriv_imu))
    second_deriv_imu = np.gradient(deriv_imu_filt)
    second_deriv_imu = second_deriv_imu/np.max(np.abs(second_deriv_imu))
    imu_valleys, _ = find_peaks(second_deriv_imu)
    imu_pks, _ = find_peaks(-second_deriv_imu)
    return imu_pks, imu_valleys, second_deriv_imu

def find_nth_closest_point(peak, f_points, pos='before',n=0):
    if pos == 'before':
        distances = peak - f_points
    elif pos == 'after':
        distances = f_points - peak
    distances[distances<0] = 10
    return f_points[np.argsort(distances)[n]]
    
def load_imu(folder, subject_id, trial, mov, noise, sound, start=1, end=0.5):
    
    fn = subject_id + '/' + subject_id + '_' + trial + '/mov_' + mov + '/background_noise_' + noise + '/' + sound + '/imu.csv'

    try:        
        df = pd.read_csv(folder + fn, header=None)
    except FileNotFoundError as err:
        print("ERROR: IMU file not found")
        return 0
    
    
    #samples to remove at the beginning due to microphone transient
    samp_trans = int(start*FS_IMU)
    samp_trans_end = int(end*FS_IMU)
    
    #Normalize recordings to [-1, +1] range
    Y = df[0].to_numpy()[samp_trans:-samp_trans_end]
    P = df[1].to_numpy()[samp_trans:-samp_trans_end]
    R = df[2].to_numpy()[samp_trans:-samp_trans_end]
    x = df[3].to_numpy()[samp_trans:-samp_trans_end]
    y = df[4].to_numpy()[samp_trans:-samp_trans_end]
    z = df[5].to_numpy()[samp_trans:-samp_trans_end]
    
    imu = IMU(x,y,z,Y,P,R) 
    
    return imu

def yamnet_test(yamnet_model, data, fs):
    scores, embeddings, spectrogram = yamnet_model(data)
    yamnet_window = 0.960*fs
    yamnet_step = 0.480*fs
    num_outputs = np.floor(len(data)/yamnet_step)
    if num_outputs != scores.shape[0]:
        print("Problem: predicted number of windows incorrect")
    idx_range = np.arange(yamnet_step*num_outputs + 1)
    scores_aggregated = np.zeros((len(idx_range),scores.shape[1]))
    for i, tstart in enumerate(range(0,int(idx_range[-1]),int(yamnet_step))):
        tend = tstart + yamnet_step
        idx_mask = (idx_range >= tstart) & (idx_range < tend)
        #First half window or last half window: no averaging
        if (i == 0) | ((tstart + yamnet_step) == idx_range[-1]):
            scores_aggregated[idx_mask,:] = scores[i,:]
        #Average currrent window with the next one 
        else:
            scores_aggregated[idx_mask,:] = (scores[i,:] + scores[i-1,:])/2
    return scores_aggregated, idx_range/fs