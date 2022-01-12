import os, fnmatch

import tensorflow as tf
import soundfile as sf
from wavinfo import WavInfoReader
from random import shuffle
import numpy as np


class audio_generator:
    """
    Class to create a Tensorflow dataset based on an iterator from a large scale
    audio dataset. This audio generator only supports single channel audio files.
    """

    def __init__(
        self,
        path_to_input,
        path_to_mic,
        path_to_lpb,
        len_of_samples,
        fs,
        train_flag=False,
    ):
        """
        Constructor of the audio generator class.
        Inputs:
            path_to_input       path to the mixtures
            path_to_mic         path to the mic audio
            path_to_lpb         path to the lpb audio
            len_of_samples      length of audio snippets in samples
            fs                  sampling rate
            train_flag          flag for activate shuffling of files
        """
        # set inputs to properties
        self.path_to_input = path_to_input
        self.path_to_mic = path_to_mic
        self.path_to_lpb = path_to_lpb
        self.len_of_samples = len_of_samples
        self.fs = fs
        self.train_flag = train_flag
        # count the number of samples in your data set (depending on your disk,
        #                                               this can take some time)
        self.count_samples()
        # create iterable tf.data.Dataset object
        self.create_tf_data_obj()

    def count_samples(self):
        """
        Method to list the data of the dataset and count the number of samples.
        """

        # list .wav files in directory
        self.file_names = fnmatch.filter(os.listdir(self.path_to_input), "*.wav")
        # count the number of samples contained in the dataset
        self.total_samples = 0
        for file in self.file_names:
            info = WavInfoReader(os.path.join(self.path_to_input, file))
            self.total_samples = self.total_samples + int(
                np.fix(info.data.frame_count / self.len_of_samples)
            )

    def create_generator(self):
        """
        Method to create the iterator.
        """

        # check if training or validation
        if self.train_flag:
            shuffle(self.file_names)
        # iterate over the files
        for file in self.file_names:
            # read the audio files
            real_file_name = '-'.join(file.split("-")[0:3])
            mixed, fs_1 = sf.read(os.path.join(self.path_to_input, file))
            mic, fs_2 = sf.read(
                os.path.join(self.path_to_mic, real_file_name + "-mic.wav")
            )
            lpb, fs_3 = sf.read(
                os.path.join(self.path_to_lpb, real_file_name + "-lpb.wav")
            )
            # check if the sampling rates are matching the specifications
            if fs_1 != self.fs or fs_2 != self.fs or fs_3 != self.fs:
                raise ValueError("Sampling rates do not match.")
            if mixed.ndim != 1 or mic.ndim != 1 or lpb.ndim != 1:
                raise ValueError(
                    "Too many audio channels. The DTLN audio_generator \
                                 only supports single channel audio data."
                )
            # count the number of samples in one file
            num_samples = int(np.fix(mixed.shape[0] / self.len_of_samples))
            # iterate over the number of samples
            for idx in range(num_samples):
                # cut the audio files in chunks
                mixed_dat = mixed[
                    int(idx * self.len_of_samples) : int(
                        (idx + 1) * self.len_of_samples
                    )
                ]
                mic_dat = mic[
                    int(idx * self.len_of_samples) : int(
                        (idx + 1) * self.len_of_samples
                    )
                ]
                lpb_dat = lpb[
                    int(idx * self.len_of_samples) : int(
                        (idx + 1) * self.len_of_samples
                    )
                ]
                # yield the chunks as float32 data
                yield (
                    mixed_dat.astype("float32"),
                    mic_dat.astype("float32"),
                ), lpb_dat.astype("float32")

    def create_tf_data_obj(self):
        """
        Method to to create the tf.data.Dataset.
        """

        types = ((tf.float32,tf.float32),
          tf.float32)
        shapes = (([self.len_of_samples],[self.len_of_samples]),
                [self.len_of_samples])
        # creating the tf.data.Dataset from the iterator
        self.tf_data_set = tf.data.Dataset.from_generator(
            self.create_generator,
            types,
            output_shapes=shapes,
            args=None,
        )
