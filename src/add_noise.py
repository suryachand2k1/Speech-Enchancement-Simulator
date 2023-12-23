from src.data_tools import audio_files_to_numpy
from src.data_tools import blend_noise_randomly, numpy_audio_to_matrix_spectrogram
import librosa
import numpy as np
import os
from src.get_file_by_class import getNoiseByID

def add_noise_one(voice_file, noise_file, output_voice_file):
    sr = 8000
    y_voice, _ = librosa.load(voice_file,sr)
    y_noise, _ = librosa.load(noise_file,sr)
    y_noise = np.concatenate((y_noise,y_noise,y_noise,y_noise,y_noise))
    y_noise = y_noise[: len(y_voice)]
    ranlv = np.random.uniform(0.4, 0.6)
    noise_ranlv = y_noise*ranlv
    y_addednoise = y_voice + noise_ranlv
    librosa.output.write_wav(output_voice_file, y_addednoise, sr)

def add_noise(voice_dir='data/voice', noise_dir='data/noise', out_dir='data/added_noise', max_time=10, sr = 8000):
    noise_data = np.zeros((len(os.listdir(noise_dir)), max_time*sr))
    noise_file = os.listdir(noise_dir)
    if len(noise_file) > 100:
        noise_file[:100]
    for i,n in enumerate(noise_file):
        print('load noise:',n)
        y, _ = librosa.load(noise_dir + '/' + n, sr=sr)
        if len(y) < max_time*sr:
            y2 = np.concatenate(y,y)
            y = y2
        noise_data[i,:] = y[:max_time*sr]
    #print(noise_data.shape)

    for i, wav in enumerate(os.listdir(voice_dir)):
        print('add noise to voice:',wav)
        y, _ = librosa.load(voice_dir + '/' + wav, sr=sr)
        noise_ranid = np.random.randint(0,len(noise_data))
        noise_randata = noise_data[noise_ranid]
        level_noise = np.random.uniform(0.2, 0.8)
        noise_ranlv = level_noise * noise_randata
        if len(y) > max_time*sr:
            y= y[:max_time*sr]
        audio_noise = y + noise_ranlv[:len(y)]
        librosa.output.write_wav(out_dir + '/' + str(i) + '.wav', audio_noise, sr)
        print('output noise voice:' + out_dir + '/' + str(i) + '.wav')

#add_noise('data/voice','data/noise','data/added_noise')