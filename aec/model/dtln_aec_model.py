import os
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Activation,
    Dense,
    LSTM,
    Dropout,
    Lambda,
    Input,
    Multiply,
    Layer,
    Conv1D,
)
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
)
import tensorflow as tf
from random import seed
import numpy as np

from aec_blood import audio_generator


class DTLN_model:
    """
    Class to create and train the DTLN model
    """

    def __init__(self):
        """
        Constructor
        """

        # defining default cost function
        self.cost_function = self.snr_cost
        # empty property for the model
        self.model = []
        # defining default parameters
        self.fs = 16000
        self.batchsize = 32
        self.len_samples = 15
        self.activation = "sigmoid"
        self.numUnits = 512
        self.numLayer = 2
        self.blockLen = 512
        self.block_shift = 128
        self.dropout = 0.25
        self.lr = 1e-3
        self.max_epochs = 200
        self.encoder_size = 256
        self.eps = 1e-7
        # reset all seeds to 42 to reduce invariance between training runs
        os.environ["PYTHONHASHSEED"] = str(42)
        seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        # some line to correctly find some libraries in TF 2.x
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, enable=True)

    @staticmethod
    def snr_cost(s_estimate, s_true):
        """
        Static Method defining the cost function.
        The negative signal to noise ratio is calculated here. The loss is
        always calculated over the last dimension.
        """

        # calculating the SNR
        snr = tf.reduce_mean(tf.math.square(s_true), axis=-1, keepdims=True) / (
            tf.reduce_mean(tf.math.square(s_true - s_estimate), axis=-1, keepdims=True)
            + 1e-7
        )
        # using some more lines, because TF has no log10
        num = tf.math.log(snr)
        denom = tf.math.log(tf.constant(10, dtype=num.dtype))
        loss = -10 * (num / (denom))
        # returning the loss
        return loss

    def loss_wrapper(self):
        """
        A wrapper function which returns the loss function. This is done to
        to enable additional arguments to the loss function if necessary.
        """

        def lossFunction(y_true, y_pred):
            # calculating loss and squeezing single dimensions away
            loss = tf.squeeze(self.cost_function(y_pred, y_true))
            # calculate mean over batches
            loss = tf.reduce_mean(loss)
            # return the loss
            return loss

        # returning the loss function as handle
        return lossFunction

    """
    In the following some helper layers are defined.
    """

    def stft_layer(self, x):
        """
        Method for an STFT helper layer used with a Lambda layer. The layer
        calculates the STFT on the last dimension and returns the magnitude and
        phase of the STFT.
        """

        # creating frames from the continuous waveform
        frames = tf.signal.frame(x, self.blockLen, self.block_shift)
        # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
        stft_dat = tf.signal.rfft(frames)
        # calculating magnitude and phase from the complex signal
        mag = tf.abs(stft_dat)
        phase = tf.math.angle(stft_dat)
        # returning magnitude and phase as list
        return [mag, phase]

    def fft_layer(self, x):
        """
        Method for an fft helper layer used with a Lambda layer. The layer
        calculates the rFFT on the last dimension and returns the magnitude and
        phase of the STFT.
        """
        # expanding dimensions
        frame = tf.expand_dims(x, axis=1)
        # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
        stft_dat = tf.signal.rfft(frame)
        # calculating magnitude and phase from the complex signal
        mag = tf.abs(stft_dat)
        phase = tf.math.angle(stft_dat)
        # returning magnitude and phase as list
        return [mag, phase]

    def ifft_layer(self, x):
        """
        Method for an inverse FFT layer used with an Lambda layer. This layer
        calculates time domain frames from magnitude and phase information.
        As input x a list with [mag,phase] is required.
        """

        # calculating the complex representation
        s1_stft = tf.cast(x[0], tf.complex64) * tf.exp(
            (1j * tf.cast(x[1], tf.complex64))
        )
        # returning the time domain frames
        return tf.signal.irfft(s1_stft)

    def overlap_add_layer(self, x):
        """
        Method for an overlap and add helper layer used with a Lambda layer.
        This layer reconstructs the waveform from a framed signal.
        """

        # calculating and returning the reconstructed waveform
        return tf.signal.overlap_and_add(x, self.block_shift)

    def seperation_kernel(self, num_layer, mask_size, x, stateful=False):
        """
        Method to create a separation kernel.
        !! Important !!: Do not use this layer with a Lambda layer. If used with
        a Lambda layer the gradients are updated correctly.

        Inputs:
            num_layer       Number of LSTM layers
            mask_size       Output size of the mask and size of the Dense layer
        """

        # creating num_layer number of LSTM layers
        for idx in range(num_layer):
            x = LSTM(self.numUnits, return_sequences=True, stateful=stateful)(x)
            # using dropout between the LSTM layer for regularization
            if idx < (num_layer - 1):
                x = Dropout(self.dropout)(x)
        # creating the mask with a Dense and an Activation layer
        mask = Dense(mask_size)(x)
        mask = Activation(self.activation)(mask)
        # returning the mask
        return mask

    def build_dtln_aec_model(self, norm_stft=False):
        """
        Method to build and compile the DTLN model. The model takes time domain
        batches of size (batchsize, len_in_samples) and returns enhanced clips
        in the same dimensions. As optimizer for the Training process the Adam
        optimizer with a gradient norm clipping of 3 is used.
        The model contains two separation cores. The first has an STFT signal
        transformation and the second a learned transformation based on 1D-Conv
        layer.
        """

        # input layer for time signal
        mic_time_dat = Input(batch_shape=(None, None))
        lpb_time_dat = Input(batch_shape=(None, None))
        # calculate STFT
        mic_mag, mic_angle = Lambda(self.stft_layer)(mic_time_dat)
        lpb_mag, lpb_angle = Lambda(self.stft_layer)(lpb_time_dat)
        # lpb_frames_1 = Lambda(self.ifft_layer)([lpb_mag,lpb_angle])
        lpb_frames_1 = tf.signal.frame(lpb_time_dat, self.blockLen, self.block_shift)
        # normalizing log magnitude stfts to get more robust against level variations
        if norm_stft:
            mag_norm = InstantLayerNormalization()(tf.math.log(mic_mag + 1e-7))
            lpb_mag_norm = InstantLayerNormalization()(tf.math.log(lpb_mag + 1e-7))
        else:
            # behaviour like in the paper
            mag_norm = mic_mag
            lpb_mag_norm = lpb_mag
        mag_norm = tf.concat([lpb_mag_norm, mag_norm], axis=-1)
        # predicting mask with separation kernel
        mask_1 = self.seperation_kernel(
            self.numLayer, (self.blockLen // 2 + 1), mag_norm
        )
        # multiply mask with magnitude
        estimated_mag = Multiply()([mic_mag, mask_1])
        # transform frames back to time domain
        estimated_frames_1 = Lambda(self.ifft_layer)([estimated_mag, mic_angle])
        # encode time domain frames to feature domain
        encoded_frames = Conv1D(self.encoder_size, 1, strides=1, use_bias=False)(
            estimated_frames_1
        )
        encoded_lpb = Conv1D(self.encoder_size, 1, strides=1, use_bias=False)(
            lpb_frames_1
        )
        # normalize the input to the separation kernel
        encoded_frames_norm = InstantLayerNormalization()(encoded_frames)
        encoded_lpb_norm = InstantLayerNormalization()(encoded_lpb)
        encoded_frames_concat = tf.concat(
            [encoded_lpb_norm, encoded_frames_norm], axis=-1
        )
        # predict mask based on the normalized feature frames
        mask_2 = self.seperation_kernel(
            self.numLayer, self.encoder_size, encoded_frames_concat
        )
        # multiply encoded frames with the mask
        estimated = Multiply()([encoded_frames, mask_2])
        # decode the frames back to time domain
        decoded_frames = Conv1D(self.blockLen, 1, padding="causal", use_bias=False)(
            estimated
        )
        # create waveform with overlap and add procedure
        estimated_sig = Lambda(self.overlap_add_layer)(decoded_frames)

        # create the model
        self.model = Model(inputs=[mic_time_dat, lpb_time_dat], outputs=estimated_sig)
        # show the model summary
        print(self.model.summary())

    def compile_model(self):
        """
        Method to compile the model for training

        """

        # use the Adam optimizer with a clipnorm of 3
        optimizerAdam = keras.optimizers.Adam(lr=self.lr, clipnorm=3.0)
        # compile model with loss function
        self.model.compile(loss=self.loss_wrapper(), optimizer=optimizerAdam)

    def create_saved_model(self, weights_file, target_name):
        """
        Method to create a saved model folder from a weights file

        """
        # check for type
        if weights_file.find("_norm_") != -1:
            norm_stft = True
        else:
            norm_stft = False
        # build model
        self.build_DTLN_model_stateful(norm_stft=norm_stft)
        # load weights
        self.model.load_weights(weights_file)
        # save model
        tf.saved_model.save(self.model, target_name)

    def train_model(
        self,
        runName,
        path_to_train_mix,
        path_to_train_mic,
        path_to_train_lpb,
        path_to_val_mix,
        path_to_val_mic,
        path_to_val_lpb,
    ):
        """
        Method to train the DTLN model.
        """

        # create save path if not existent
        savePath = "./models_" + runName + "/"
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        # create log file writer
        csv_logger = CSVLogger(savePath + "training_" + runName + ".log")
        # create callback for the adaptive learning rate
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=10 ** (-10), cooldown=1
        )
        # create callback for early stopping
        early_stopping = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=10,
            verbose=0,
            mode="auto",
            baseline=None,
        )
        # create model check pointer to save the best model
        checkpointer = ModelCheckpoint(
            savePath + runName + ".h5",
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="auto",
            save_freq="epoch",
        )

        # calculate length of audio chunks in samples
        len_in_samples = int(
            np.fix(self.fs * self.len_samples / self.block_shift) * self.block_shift
        )
        # create data generator for training data
        generator_input = audio_generator(
            path_to_train_mix,
            path_to_train_mic,
            path_to_train_lpb,
            len_in_samples,
            self.fs,
            train_flag=True,
        )
        dataset = generator_input.tf_data_set
        dataset = dataset.batch(self.batchsize, drop_remainder=True).repeat()
        # calculate number of training steps in one epoch
        steps_train = generator_input.total_samples // self.batchsize
        # create data generator for validation data
        generator_val = audio_generator(
            path_to_val_mix, path_to_val_mic, path_to_val_lpb, len_in_samples, self.fs
        )
        dataset_val = generator_val.tf_data_set
        dataset_val = dataset_val.batch(self.batchsize, drop_remainder=True).repeat()
        # calculate number of validation steps
        steps_val = generator_val.total_samples // self.batchsize
        # start the training of the model
        self.model.fit(
            x=dataset,
            batch_size=None,
            steps_per_epoch=steps_train,
            epochs=self.max_epochs,
            verbose=1,
            validation_data=dataset_val,
            validation_steps=steps_val,
            callbacks=[checkpointer, reduce_lr, csv_logger, early_stopping],
            max_queue_size=50,
            workers=4,
            use_multiprocessing=True,
        )
        # clear out garbage
        tf.keras.backend.clear_session()


class InstantLayerNormalization(Layer):
    """
    Class implementing instant layer normalization. It can also be called
    channel-wise layer normalization and was proposed by
    Luo & Mesgarani (https://arxiv.org/abs/1809.07454v2)
    """

    def __init__(self, **kwargs):
        """
        Constructor
        """
        super(InstantLayerNormalization, self).__init__(**kwargs)
        self.epsilon = 1e-7
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        """
        Method to build the weights.
        """
        shape = input_shape[-1:]
        # initialize gamma
        self.gamma = self.add_weight(
            shape=shape, initializer="ones", trainable=True, name="gamma"
        )
        # initialize beta
        self.beta = self.add_weight(
            shape=shape, initializer="zeros", trainable=True, name="beta"
        )

    def call(self, inputs):
        """
        Method to call the Layer. All processing is done here.
        """
        # calculate mean of each frame
        mean = tf.math.reduce_mean(inputs, axis=[-1], keepdims=True)
        # calculate variance of each frame
        variance = tf.math.reduce_mean(
            tf.math.square(inputs - mean), axis=[-1], keepdims=True
        )
        # calculate standard deviation
        std = tf.math.sqrt(variance + self.epsilon)
        # normalize each frame independently
        outputs = (inputs - mean) / std
        # scale with gamma
        outputs = outputs * self.gamma
        # add the bias beta
        outputs = outputs + self.beta
        return outputs


model = DTLN_model()
model.build_dtln_aec_model()
