import librosa
import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)
from tensorflow.keras.models import model_from_json
from src.data_tools import scaled_in, inv_scaled_ou
from src.data_tools import audio_file_to_numpy, audio_files_to_numpy, numpy_audio_to_matrix_spectrogram, matrix_spectrogram_to_numpy_audio
import os
import scipy

def prediction(weights_path, name_model, audio_dir_prediction, dir_save_prediction, sample_rate, min_duration, frame_length, hop_length_frame, n_fft, hop_length_fft):

    # load json and create model
    json_file = open(weights_path+'/'+name_model+'.json', 'r')
    #print(json_file)
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    print('Load json done')
    # load weights into new model
    loaded_model.load_weights(weights_path+'/'+name_model+'.h5')
    print("Load weight done")
    for file in os.listdir(audio_dir_prediction):
        print('predict file: {}/{}'.format(audio_dir_prediction,[file]))
        # Extracting noise and voice from folder and convert to numpy
        audio = audio_files_to_numpy(audio_dir_prediction, [file], sample_rate, frame_length, hop_length_frame, min_duration)

        #Dimensions of squared spectrogram
        dim_square_spec = int(n_fft / 2) + 1
        #print(dim_square_spec)

        # Create Amplitude and phase of the sounds
        m_amp_db_audio,  m_pha_audio = numpy_audio_to_matrix_spectrogram(audio, dim_square_spec, n_fft, hop_length_fft)

        #global scaling to have distribution -1/1
        X_in = scaled_in(m_amp_db_audio)
        #Reshape for prediction
        X_in = X_in.reshape(X_in.shape[0],X_in.shape[1],X_in.shape[2],1)
        #Prediction using loaded network
        X_pred = loaded_model.predict(X_in)
        #Rescale back the noise model
        inv_sca_X_pred = inv_scaled_ou(X_pred)
        #Remove noise model from noisy speech
        X_denoise = m_amp_db_audio - inv_sca_X_pred[:,:,:,0]
        #Reconstruct audio from denoised spectrogram and phase
        #print(X_denoise.shape)
        #print(m_pha_audio.shape)
        #print(frame_length)
        #print(hop_length_fft)
        audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio, frame_length, hop_length_fft)
        #Number of frames
        nb_samples = audio_denoise_recons.shape[0]
        #Save all frames in one file
        denoise_long = audio_denoise_recons.reshape(1, nb_samples * (frame_length-1))*10
        #print(type(denoise_long[0, :][0]))
        #scipy.io.wavfile.write(dir_save_prediction +'/'+ file, sample_rate, denoise_long[0, :])
        librosa.output.write_wav(dir_save_prediction +'/'+ file, denoise_long[0, :], sample_rate)
        print('save predicted file: {}/{}'.format(dir_save_prediction,file))
        exit()

def predictOne(filename, ouput_filename):
    sr = 8000#/2
    model_dir = 'src'
    if sr == 8000:
        frame_length = 8064
        hop_length_frame = 8064
        n_fft = 255
        min_duration = 1.0
        hop_length_fft = 63
        dim_square_spec = 128
    elif sr == 16000:
        frame_length = 8063*2
        hop_length_frame = frame_length
        n_fft = 255*2
        min_duration = 1.0
        hop_length_fft = 63
        dim_square_spec = 256
    else:
        print('sr error!!!')
        exit()
    # load json and create model
    json_file = open(model_dir + '/weights/model.json', 'r')
    #print(json_file)
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    print('Load json done')
    # load weights into new model
    loaded_model.load_weights(model_dir+'/weights/model.h5')
    print("Load weight done")
    #print('aaaaa:',filename)
    audio = audio_files_to_numpy('',[filename], sr, frame_length, hop_length_frame, min_duration)
    print('audio:',audio.shape)
    # Create Amplitude and phase of the sounds
    m_amp_db_audio,  m_pha_audio = numpy_audio_to_matrix_spectrogram(audio, dim_square_spec, n_fft, hop_length_fft, phase=True)

    #global scaling to have distribution -1/1
    X_in = scaled_in(m_amp_db_audio)
    #Reshape for prediction
    X_in = X_in.reshape(X_in.shape[0],X_in.shape[1],X_in.shape[2],1)
    #Prediction using loaded network
    X_pred = loaded_model.predict(X_in)
    #Rescale back the noise model
    inv_sca_X_pred = inv_scaled_ou(X_pred)
    #Remove noise model from noisy speech
    X_denoise = m_amp_db_audio - inv_sca_X_pred[:,:,:,0]
    #Reconstruct audio from denoised spectrogram and phase
    audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio, frame_length, hop_length_fft)
    #Number of frames
    nb_samples = audio_denoise_recons.shape[0]
    #Save all frames in one file
    denoise_long = audio_denoise_recons.reshape(1, nb_samples * (frame_length))*10
    #print(type(denoise_long[0, :][0]))
    #scipy.io.wavfile.write(dir_save_prediction +'/'+ file, sample_rate, denoise_long[0, :])
    librosa.output.write_wav(ouput_filename, denoise_long[0, :], sr)