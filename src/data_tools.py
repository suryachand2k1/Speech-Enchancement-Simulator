import librosa
import numpy as np
import os
import matplotlib.pyplot as plt

def audio_to_audio_frame_stack(sound_data, frame_length, hop_length_frame):
    """This function take an audio and split into several frame
       in a numpy matrix of size (nb_frame,frame_length)"""
    #print(sound_data, frame_length, hop_length_frame)
    sequence_sample_length = sound_data.shape[0]
    sound_data_array = []
    #for a in range(0, sequence_sample_length - frame_length + 1, hop_length_frame):
    #    print(a)
    sound_data_list = [sound_data[start:start + frame_length] for start in range(
    0, sequence_sample_length - frame_length + 1, hop_length_frame)]  # get sliding windows
    sound_data_array = np.vstack(sound_data_list)
    #print('sound_data_array:',sound_data_array.shape)
    return sound_data_array

def audio_files_to_numpy(audio_dir, list_audio_files, sample_rate, frame_length, hop_length_frame, min_duration):
    """This function take audio files of a directory and merge them
    in a numpy matrix of size (nb_frame,frame_length) for a sliding window of size hop_length_frame"""

    list_sound_array = []

    for i,file in enumerate(list_audio_files):
        print('load file {}/{}: {}'.format(i,len(list_audio_files),file))
        y, sr = librosa.load(audio_dir + file, sr=sample_rate)
        total_duration = librosa.get_duration(y=y, sr=sr)
        #print(total_duration)
        if (len(y) > frame_length):
            list_sound_array.append(audio_to_audio_frame_stack(y, frame_length, hop_length_frame))
        else:
            print(f"The following file {os.path.join(audio_dir,file)} is below the min duration")

    return np.vstack(list_sound_array)

def audio_file_to_numpy(filename, sample_rate, frame_length, hop_length_frame, min_duration):
    """This function take audio files of a directory and merge them
    in a numpy matrix of size (nb_frame,frame_length) for a sliding window of size hop_length_frame"""
    list_sound_array = []
    y, _ = librosa.load(filename, sr=sample_rate)
    if (len(y) > frame_length):
        list_sound_array.append(audio_to_audio_frame_stack(y, frame_length, hop_length_frame))
    else:
        print(f"The following file {filename} is below the min duration")

    return np.vstack(list_sound_array)

def blend_noise_randomly(voice, noise, nb_samples, frame_length):

    if nb_samples == -1:
        nb_samples = len(voice)

    prod_voice = np.zeros((nb_samples, frame_length))
    prod_noise = np.zeros((nb_samples, frame_length))
    prod_noisy_voice = np.zeros((nb_samples, frame_length))

    for i in range(nb_samples):
        id_voice = np.random.randint(0, voice.shape[0])
        id_noise = np.random.randint(0, noise.shape[0])
        level_noise = np.random.uniform(0.2, 0.8)
        prod_voice[i, :] = voice[id_voice, :]
        prod_noise[i, :] = level_noise * noise[id_noise, :]
        prod_noisy_voice[i, :] = prod_voice[i, :] + prod_noise[i, :]

    return prod_voice, prod_noise, prod_noisy_voice


def audio_to_magnitude_db_and_phase(n_fft, hop_length_fft, audio, phase):
    """This function takes an audio and convert into spectrogram,
       it returns the magnitude in dB and the phase"""
    if phase:
        stftaudio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft)
        stftaudio_magnitude, stftaudio_phase = librosa.magphase(stftaudio)
        stftaudio_magnitude_db = librosa.amplitude_to_db(stftaudio_magnitude, ref=np.max)
        #print(stftaudio_magnitude_db.shape)
        #plt.imshow(stftaudio_magnitude_db)
        #plt.savefig('stftaudio_magnitude_db.png')
        return stftaudio_magnitude_db, stftaudio_phase
    else:
        stftaudio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft)
        stftaudio_magnitude, _ = librosa.magphase(stftaudio)
        stftaudio_magnitude_db = librosa.amplitude_to_db(stftaudio_magnitude, ref=np.max)

        return stftaudio_magnitude_db


def numpy_audio_to_matrix_spectrogram(numpy_audio, dim_square_spec, n_fft, hop_length_fft, phase=False):
    nb_audio = numpy_audio.shape[0]
    m_mag_db = np.zeros((nb_audio, dim_square_spec, dim_square_spec))
    if phase:
        m_phase = np.zeros((nb_audio, dim_square_spec, dim_square_spec), dtype=complex)
        for i in range(nb_audio):
            m_mag_db[i, :, :], m_phase[i, :, :] = audio_to_magnitude_db_and_phase(n_fft, hop_length_fft, numpy_audio[i], phase)
        return m_mag_db, m_phase
    else:
        for i in range(nb_audio):
            m_mag_db[i, :, :] = audio_to_magnitude_db_and_phase(n_fft, hop_length_fft, numpy_audio[i], phase)
        return m_mag_db



def magnitude_db_and_phase_to_audio(frame_length, hop_length_fft, stftaudio_magnitude_db, stftaudio_phase):
    """This functions reverts a spectrogram to an audio"""

    stftaudio_magnitude_rev = librosa.db_to_amplitude(stftaudio_magnitude_db, ref=1.0)
    #print(stftaudio_magnitude_rev[0])
    #print(stftaudio_magnitude_rev.shape)
    #print(stftaudio_magnitude_rev.shape)
    #print(frame_length, hop_length_fft)
    # taking magnitude and phase of audio
    audio_reverse_stft = stftaudio_magnitude_rev * stftaudio_phase
    audio_reconstruct = librosa.core.istft(audio_reverse_stft, hop_length=hop_length_fft, length=frame_length)
    #audio_reconstruct = librosa.griffinlim(audio_reverse_stft, hop_length=hop_length_fft, length=frame_length-1)
    #exit()
    return audio_reconstruct

def matrix_spectrogram_to_numpy_audio(m_mag_db, m_phase, frame_length, hop_length_fft)  :
    """This functions reverts the matrix spectrograms to numpy audio"""
    list_audio = []
    nb_spec = m_mag_db.shape[0]
    for i in range(nb_spec):
        audio_reconstruct = magnitude_db_and_phase_to_audio(frame_length, hop_length_fft, m_mag_db[i], m_phase[i])
        list_audio.append(audio_reconstruct)
    return np.vstack(list_audio)

def scaled_in(matrix_spec):
    "global scaling apply to noisy voice spectrograms (scale between -1 and 1)"
    matrix_spec = (matrix_spec + 46)/50
    return matrix_spec

def scaled_ou(matrix_spec):
    "global scaling apply to noise models spectrograms (scale between -1 and 1)"
    matrix_spec = (matrix_spec -6 )/82
    return matrix_spec

def inv_scaled_in(matrix_spec):
    "inverse global scaling apply to noisy voices spectrograms"
    matrix_spec = matrix_spec * 50 - 46
    return matrix_spec

def inv_scaled_ou(matrix_spec):
    "inverse global scaling apply to noise models spectrograms"
    matrix_spec = matrix_spec * 82 + 6
    return matrix_spec
