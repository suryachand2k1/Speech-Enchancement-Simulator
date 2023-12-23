import librosa
from tensorflow.keras.models import model_from_json
from data_tools import scaled_in, inv_scaled_ou
from data_tools import audio_files_to_numpy, numpy_audio_to_matrix_spectrogram, matrix_spectrogram_to_numpy_audio
import matplotlib.pyplot as plt
import numpy as np
from src.autoencoder import ConvAutoEncoder

def decode(weights_path = args.weights_folder, name_model = args.name_model, audio_dir_prediction = args.audio_dir_prediction, dir_save_prediction = args.dir_save_prediction, audio_input_prediction = args.audio_input_prediction,
audio_output_prediction = args.audio_output_prediction, sample_rate = args.sample_rate, min_duration = args.min_duration, frame_length = args.frame_length, hop_length_frame = args.hop_length_frame, n_fft = args.n_fft, hop_length_fft = args.hop_length_fft):

    loaded_model = ConvAutoEncoder(weights_path = weights_path)
    loaded_model.load_weights()
    loaded_model.info()
    print("Loaded model from disk")
    audio = audio_files_to_numpy(audio_dir_prediction, audio_input_prediction, sample_rate, frame_length, hop_length_frame, min_duration)
    #Dimensions of squared spectrogram
    dim_square_spec = int(n_fft / 2) + 1
    # Create Amplitude and phase of the sounds
    m_amp_db_audio,  m_pha_audio = numpy_audio_to_matrix_spectrogram(audio, dim_square_spec, n_fft, hop_length_fft)

    data_compress = np.load('aaa.npy')
    print(data_compress.shape)
    decoded = loaded_model.decode(data_compress)
    #Rescale back the noise model
    inv_sca_X_pred = inv_scaled_ou(decoded)
    #Remove noise model from noisy speech
    X_denoise = m_amp_db_audio - inv_sca_X_pred[:,:,:,0]
    #Reconstruct audio from denoised spectrogram and phase
    print(X_denoise.shape)
    print(m_pha_audio.shape)
    print(frame_length)
    print(hop_length_fft)
    audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio, frame_length, hop_length_fft)
    #Number of frames
    nb_samples = audio_denoise_recons.shape[0]
    #Save all frames in one file
    denoise_long = audio_denoise_recons.reshape(1, nb_samples * frame_length)*10
    librosa.output.write_wav(dir_save_prediction + audio_output_prediction, denoise_long[0, :], sample_rate)
    print('saved audio decoded file in:', dir_save_prediction + audio_output_prediction)

decode()