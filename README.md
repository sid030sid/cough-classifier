# Cough classifier

## About
This project sets itself the goal to build, train and test a binary classifier identifying whether there is a cough in an audio file or not. The data comes from the Embeded Systems Lab at EPFL and thus is private. The data originates from 16 subjects whose audio signals were recorded by a device designed at EPFL.

The cough classifier is a neural network based on CNN. The chosen preprocessing of the audio signals is STFT. Due to the low amount of data, data augmentation is crucial. Therefore, more data is produced doing time shifting and splitting each audio file into sounds of 0.7 seconds. This window length was abritarily chosen. It, however, effects the performance of the model, thus one can do a subsequent study on finding the optimal window length.

## Tech stack
1. pytorch - for building, training and testing neural network