'''
Deep Learning FrameWork for atrial fibrillation (AF) classification. Here we
deal with the following questions:
    1. How can we harness the power of CNNs for 1d temporal signals such as ECG?
    2. How do we handle skewness and label imbalance in data?
    3. How do we train large tensors using data generators?
    4. How can we parallelize data transfer using generators?
CopyRight , Feb, 2019
Hooman Sedghamiz
'''

import os
import sys
import re
import random
from scipy.io import loadmat
from scipy.signal import stft
import matplotlib.pyplot as plt
import numpy as np
import pandas as pn
import keras
from shutil import copyfile
from DataGenerator import DataGenerator

class ProgressBar(object):
    '''
    Progress bar Class
    '''
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = 'Loading: %(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        self.current += 1
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        print('', file=self.output)

class DeepNetAF(object):
    ReadPath = ""
    FilesToLoad = []
    Signals = np.array([])
    partition = {'train': [], 'validation': []}
    labels = {}
    # Parameters
    params = {'dim': None,
          'batch_size': 40,
          'n_classes': 2,
          'n_channels': 3,
          'shuffle': True}

    def __init__(self, PathR = os.path.realpath(__file__), AnnotN=None):
        '''
        Constructor for ApneaDeepLearning:
        PathR : The Path to Annotation Files default is current filename path.
        Example :  A = ApneaDeepLearning('D:/heartbeat/polysomnography/annotations-events-nsrr/baseline')
        '''
        self.ReadPath = PathR
        self.AnnotF   = AnnotN
        self.listFiles('.mat')

    def listFiles(self,Ext):
        '''
        Loads a set of files with extention input EXT : 'mat','xml'
        Ext : Extension of the file e.g. ".mat"
        Example :  listFiles(".mat")
        '''
        for file in os.listdir(self.ReadPath):
            if file.endswith(Ext):
                self.FilesToLoad.append(file)

    def ImportAllSigs(self,y_index, Fs=300):
        '''
        Imports all of the recordings in PathLoad
        Returns:
        A panda dataframe containing all of the patients

        Returns: A Tensor N*T*F (Number of trials * time-samples * Features)
        '''
        LD = ProgressBar(len(self.FilesToLoad), fmt=ProgressBar.FULL)
        ShortSegs = []
        true_index = []
        counter = 0
        for j,i in enumerate(self.FilesToLoad):
            LD()
            if y_index[j]:
                temp = self.ReadMatFile(i)
                d = temp.shape
                if d[0] < d[1]:
                    temp = np.transpose(temp)

                if j==0:
                    self.Signals = np.zeros((np.sum(y_index),temp.shape[0],1)) # Nr trials, timesteps, data_dim
                # --- Truncate the signal ------ #
                if temp.shape[0] > self.Signals.shape[1]:
                    temp = temp[:self.Signals.shape[1]]

                if temp.shape[0] == self.Signals.shape[1]:
                    self.Signals[counter,:,:] = temp
                    counter +=1
                    true_index.append(True)
                    #------------------ compute STFT ---------------------------#
                    if counter == 1:
                        f, t, PSD = stft(self.Signals[counter,:,0], Fs, nperseg=Fs/2)
                        PSD_img = self.CreateImage(f,t,PSD)
                        self.params['dim'] = tuple(PSD_img.shape[:2])
                    else:
                        _, _, PSD = stft(self.Signals[counter,:,0], Fs, nperseg=Fs/2)
                        PSD_img = self.CreateImage(f,t,PSD)
                    
                    #------------------ Save PSD to disk -------------------------- #
                    np.save(os.path.join('AFdata', 'id-'+ str(counter)), PSD_img)
                else:
                    true_index.append(False)
                    ShortSegs.append(j)
        self.Signals = self.Signals[:counter]
        LD.done()
        print(len(true_index))
        return true_index,ShortSegs

    def ComputePSD(self,Fs=300):
        '''
        Computes and saves PSD of the signals in self.Signals
        '''
        if self.Signals.size == 0:
            raise ValueError('No sequence found. First load the signals.')
        # --- Compute the ST-ft in non-overlapping windows of 0.5 sec --- #
        LD = ProgressBar(self.Signals.shape[0], fmt=ProgressBar.FULL)
        for i in range(self.Signals.shape[0]):
            LD()
            if i == 0:
                f, t, PSD = stft(self.Signals[i,:,0], Fs, nperseg=Fs/2)
                PSD = self.CreateImage(f,t,PSD)
                # ------------- Instantiate Image matrix RGB -------------- #
                x_train = np.zeros((self.Signals.shape[0], PSD.shape[0], PSD.shape[1], 3))
                # ------------- Create plots from PSD --------------------- #
                x_train[i,:,:,:] = PSD
            else:
                # ------------- Instantiate Image matrix RGB -------------- #
                _, _, PSD = stft(self.Signals[i,:,0], Fs, nperseg=Fs/2)
                # ------------- Create plots from PSD --------------------- #
                x_train[i,:,:,:] = self.CreateImage(f,t,PSD)

        LD.done()
        return x_train


    def CreateImage(self,f,t,PSD):
        '''
        Accepts frequency (f), time (t) and PSD and
        creates an image, then returns the image as a matrix.
        '''
        fig = plt.figure(frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        plt.pcolormesh(t, f, np.abs(PSD), vmin=0)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        test = fig.canvas.tostring_rgb()
        mplimage = np.fromstring(test, dtype=np.uint8).reshape(height, width, 3)
        plt.close('all')

        return mplimage

    def ScalerF(self,data):
        '''
        Use SKlearn to standardize
        '''
        # train the standardization
        scaler = StandardScaler()
        scaler = scaler.fit(data)
        normalized = scaler.transform(data)
        return normalized, scaler

    def ReadMatFile(self,Fname):
        '''
        Read a .mat format file (Fname) and import to a np array
        '''
        # Read XML iteratively :  Only searches in ScoredEvents
        RF = os.path.join(self.ReadPath,Fname)
        # Import the signal
        Sig    = loadmat(RF)['val']

        return Sig

    def SplitData(self,InputSize, trainSize = 0.9):
        '''
        Splits the input into test and train and returns the indices of them as
        test_label: Indices of test data
        train_label: Indices of training data
        Mask_B: Binary mask where the true bits represent the training samples and false bits the test
        '''
        Mask_B = np.zeros(InputSize, dtype = bool)
        train_label = random.sample(np.arange(0, InputSize).tolist(), round(trainSize*float(InputSize)))
        Mask_B[train_label] = True
        test_label = np.where(Mask_B==False)[0].tolist()

        return test_label, train_label, Mask_B


    def ReadAnnot(self,AnnotF):
        '''
        Import the annotation csv file
        '''
        if AnnotF != None:
            dataframe = pn.read_csv(os.path.join(self.ReadPath,AnnotF),header = None,engine='python',usecols=[1],squeeze = True)
            y_index  = np.logical_or(dataframe.str.contains("N"),dataframe.str.contains("A"))
            dataframe = dataframe[y_index]
            y_train = np.zeros((dataframe.shape[0],))
            y_train[dataframe.str.contains("A")] = 1
        else:
            raise ValueError('No Annotation File Provided!')

        return y_train,y_index


    def UpSample(self,ind, factor, total_size_data, TrainFlag = 'train'):
        '''
        Given the index of files, this function upsamples them by factor and saves the results in the same folder 
        and new ID which is incremented based on the total size of data

        TrainFlag :  Set to 'validation' if the data upsampled is for testing otherwise leave as default
        '''
        for i in ind:
            for j in range(0,factor):
                copyfile(os.path.join('AFdata','id-'+ str(i+1) + '.npy'),os.path.join('AFdata','id-'+ str(total_size_data+1) + '.npy'))
                self.partition[TrainFlag].append('id-'+str(total_size_data+1))
                self.labels['id-'+ str(total_size_data+1)] = self.labels['id-'+ str(i+1)]
                total_size_data += 1

    def Create_Partition(self, ind, y_train, TrainFlag = 'train'):
        '''
        Packs the labels (y_train) and data in a dictionary for keras generator
        Note: y_train should contain all of the labels for data
        '''
        for i in ind:
            self.partition[TrainFlag].append('id-'+str(i+1))
            self.labels['id-'+str(i+1)] = y_train[i]








if __name__ == "__main__":
    '''
        Helper to test out the classes.
    '''
    # -------------- Initialize the class ---------- #
    data = DeepNetAF('C:/Users/hooma/Documents/Visual Studio 2015/Projects/DL-AtrialD/training2017')
    # --------------Import labels ------------------ #
    y_train,y_index = data.ReadAnnot('REFERENCE-original.csv')

    # --------------Import ECG segments ------------ #
    y_index = data.ImportAllSigs(y_index)
 
    # --------Prepare training and validation------- #
    y_train = y_train[y_index[0][:]]
    normal_test_ind, normal_train_ind, _ = data.SplitData(sum(y_train==0))
    print('Nr of Test= %d , Nr Training= %d' % (len(normal_test_ind), len(normal_train_ind)))
    Afib_test_ind, Afib_train_ind, _ = data.SplitData(sum(y_train==1))
    print('Nr of Test= %d , Nr Training= %d' % (len(Afib_test_ind), len(Afib_train_ind)))
 
    #--------------- Partition Data ----------------#
    data.Create_Partition(normal_test_ind,y_train,TrainFlag = 'validation')

    data.Create_Partition(normal_train_ind,y_train)

    data.Create_Partition(Afib_test_ind,y_train,TrainFlag = 'validation')
    data.Create_Partition(Afib_train_ind,y_train)

    #------------- Upsample ---------------- #
    data.UpSample(Afib_train_ind, 7, len(y_train))
    data.UpSample(Afib_test_ind, 7, len(y_train),TrainFlag = 'validation')

    
    # ---------------- Generators ----------------------- #
    training_generator = DataGenerator(data.partition['train'], data.labels, **data.params)
    validation_generator = DataGenerator(data.partition['validation'], data.labels, **data.params)


    #------------- Create a simple VGG network----------- #
    model = keras.models.Sequential()
    # input: 100x100 images with 3 channels -> (640, 480, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    INP = list(data.params['dim'])
    INP.append(data.params['n_channels'])
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=tuple(INP)))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    hist = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)
    # -------------- Print Validation Accuracy --------------- #
    print(hist.history)
