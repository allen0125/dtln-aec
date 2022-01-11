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

    def __init__(self, path_to_input, path_to_s1, len_of_samples, fs, train_flag=False):
        """
        Constructor of the audio generator class.
        Inputs:
            path_to_input       path to the mixtures
            path_to_s1          path to the target source data
            len_of_samples      length of audio snippets in samples
            fs                  sampling rate
            train_flag          flag for activate shuffling of files
        """
        # set inputs to properties
        self.path_to_input = path_to_input
        self.path_to_s1 = path_to_s1
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
            noisy, fs_1 = sf.read(os.path.join(self.path_to_input, file))
            speech, fs_2 = sf.read(os.path.join(self.path_to_s1, file))
            # check if the sampling rates are matching the specifications
            if fs_1 != self.fs or fs_2 != self.fs:
                raise ValueError("Sampling rates do not match.")
            if noisy.ndim != 1 or speech.ndim != 1:
                raise ValueError(
                    "Too many audio channels. The DTLN audio_generator \
                                 only supports single channel audio data."
                )
            # count the number of samples in one file
            num_samples = int(np.fix(noisy.shape[0] / self.len_of_samples))
            # iterate over the number of samples
            for idx in range(num_samples):
                # cut the audio files in chunks
                in_dat = noisy[
                    int(idx * self.len_of_samples) : int(
                        (idx + 1) * self.len_of_samples
                    )
                ]
                tar_dat = speech[
                    int(idx * self.len_of_samples) : int(
                        (idx + 1) * self.len_of_samples
                    )
                ]
                # yield the chunks as float32 data
                yield in_dat.astype("float32"), tar_dat.astype("float32")

    def create_tf_data_obj(self):
        """
        Method to to create the tf.data.Dataset.
        """

        # creating the tf.data.Dataset from the iterator
        self.tf_data_set = tf.data.Dataset.from_generator(
            self.create_generator,
            (tf.float32, tf.float32),
            output_shapes=(
                tf.TensorShape([self.len_of_samples]),
                tf.TensorShape([self.len_of_samples]),
            ),
            args=None,
        )
