from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import datetime
import os
import keras
# from args import parser
# args = parser.parse_args()
# weights_path = args.weights_path

class ConvAutoEncoder:
    def __init__(self, weights_path, input_shape = (128,128,1), output_dim = 16, filters=[32, 64, 128, 256],
                 kernel=(3,3), stride=(1,1), strideundo=2, pool=(2,2),
                 optimizer="adamax", lossfn="mse"):
        # For now, assuming input_shape is mxnxc, and m,n are multiples of 2.

        self.input_shape = input_shape
        self.output_dim  = output_dim
        self.weights_path  = weights_path

        # define encoder architecture
        self.encoder = keras.models.Sequential()
        self.encoder.add(keras.layers.InputLayer(input_shape))
        for i in range(len(filters)):
            self.encoder.add(keras.layers.Conv2D(filters=filters[i], kernel_size=kernel, strides=stride, activation='elu', padding='same'))
            self.encoder.add(keras.layers.MaxPooling2D(pool_size=pool))
        self.encoder.add(keras.layers.Flatten())
        self.encoder.add(keras.layers.Dense(output_dim))

        # define decoder architecture
        self.decoder = keras.models.Sequential()
        self.decoder.add(keras.layers.InputLayer((output_dim,)))
        self.decoder.add(keras.layers.Dense(filters[len(filters)-1] * int(input_shape[0]/(2**(len(filters)))) * int(input_shape[1]/(2**(len(filters))))))
        self.decoder.add(keras.layers.Reshape((int(input_shape[0]/(2**(len(filters)))),int(input_shape[1]/(2**(len(filters)))), filters[len(filters)-1])))
        for i in range(1,len(filters)):
            self.decoder.add(keras.layers.Conv2DTranspose(filters=filters[len(filters)-i], kernel_size=kernel, strides=strideundo, activation='elu', padding='same'))
        self.decoder.add(keras.layers.Conv2DTranspose(filters=input_shape[2], kernel_size=kernel, strides=strideundo, activation=None, padding='same'))

        # compile model
        input         = keras.layers.Input(input_shape)
        code          = self.encoder(input)
        reconstructed = self.decoder(code)

        self.ae = keras.models.Model(inputs=input, outputs=reconstructed)
        self.ae.compile(optimizer=optimizer, loss=lossfn)

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=25):
        #Save best models to disk during training
        checkpoint = keras.callbacks.ModelCheckpoint(self.weights_path+'/model_best.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
        #tensorboard = TensorBoard(log_dir="log\\tensorboard\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        history = self.ae.fit(Xtrain, Ytrain, epochs=epochs, batch_size=10, validation_data=(Xtest, Ytest), callbacks=[checkpoint])
        self.mse = self.ae.evaluate(Xtest, Ytest)
        print('CAE MSE on validation data: ', self.mse)

        # loss = history.history['loss']
        # val_loss = history.history['val_loss']
        # epochs = range(1, len(loss) + 1)
        # plt.plot(epochs, loss, label='Training loss')
        # plt.plot(epochs, val_loss, label='Validation loss')
        # plt.yscale('log')
        # plt.title('Training and validation loss')
        # plt.legend()
        # plt.show()
    def info(self):
        self.ae.summary()

    def save_weights(self, path=None, prefix=""):
        model_json = self.ae.to_json()
        with open(self.weights_path + "/model.json", "w") as json_file:
            json_file.write(model_json)

        self.encoder.save_weights(self.weights_path + "/encoder_weights.h5")
        self.decoder.save_weights(self.weights_path + "/decoder_weights.h5")

    def load_weights(self, path=None, prefix=""):
        self.encoder.load_weights(self.weights_path + "/encoder_weights.h5")
        self.decoder.load_weights(self.weights_path + "/decoder_weights.h5")

    def encode(self, input):
        return self.encoder.predict(input)

    def decode(self, codes):
        return self.decoder.predict(codes)

    def predict(self, input):
        return self.ae.predict(input)