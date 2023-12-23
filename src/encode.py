import librosa
from tensorflow.keras.models import model_from_json
from data_tools import scaled_in, inv_scaled_ou
from data_tools import audio_files_to_numpy, numpy_audio_to_matrix_spectrogram, matrix_spectrogram_to_numpy_audio
import matplotlib.pyplot as plt
import numpy as np
from autoencoder import ConvAutoEncoder
from args import parser
args = parser.parse_args()

def encode(weights_path = args.weights_folder, name_model = args.name_model, audio_dir_prediction = args.audio_dir_prediction, dir_save_prediction = args.dir_save_prediction, audio_input_prediction = args.audio_input_prediction,
audio_output_prediction = args.audio_output_prediction, sample_rate = args.sample_rate, min_duration = args.min_duration, frame_length = args.frame_length, hop_length_frame = args.hop_length_frame, n_fft = args.n_fft, hop_length_fft = args.hop_length_fft):
    loaded_model = ConvAutoEncoder(weights_path = weights_path)
    loaded_model.load_weights()
    loaded_model.info()
    print("Loaded model from:",weights_path)
    # Extracting noise and voice from folder and convert to numpy
    audio = audio_files_to_numpy(audio_dir_prediction, audio_input_prediction, sample_rate, frame_length, hop_length_frame, min_duration)
    #Dimensions of squared spectrogram
    dim_square_spec = int(n_fft / 2) + 1
    # Create Amplitude and phase of the sounds
    m_amp_db_audio,  m_pha_audio = numpy_audio_to_matrix_spectrogram(audio, dim_square_spec, n_fft, hop_length_fft)
    #global scaling to have distribution -1/1
    X_in = scaled_in(m_amp_db_audio)
    #Reshape for prediction
    X_in = X_in.reshape(X_in.shape[0],X_in.shape[1],X_in.shape[2],1)
    encoded = loaded_model.encode(X_in)
    #print(encoded)

    print('encoded.shape:'.encoded.shape)
    np.save('aaa',encoded)
    print('encoded file:', audio_dir_prediction + str(audio_input_prediction))
    print('save to: aaa.npy')

encode()