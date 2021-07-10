#!/usr/bin/env Python
# coding=utf-8

import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras import optimizers
from decimal import *
# import seaborn as sns
import scipy
import math

getcontext().prec = 18

def pca(dataMat, alpha):
    """
    Calculate first k principal components of each subsequence.

    Args:
        dataMat (arr): the multivariate time series subsequence
        percentage (float): the percentage of extracted eigenvectors

    Returns:
        redEigVects (matrix): first k eigenvectors of principal components
    """

    meanRemoved = (dataMat-np.mean(dataMat))
    covMat = np.cov(meanRemoved,rowvar=False)
    try:
        eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    except:
        covMat[np.isnan(covMat)] = 0
        eigVals, eigVects = np.linalg.eig(np.mat(covMat))

    # Extract first k eigenvalues and eigenvectors of principal components
    eigValInd = eigVals.argsort()
    eigValInd = eigValInd[:-(alpha + 1):-1]
    redEigVects = eigVects[:, eigValInd]
    return redEigVects

def datatrans(T,window_size,dimension,step,alpha):
    """
    Subsequence segmentation and dissimilarity sequence calculation.

    Args:
        T (arr): the whole multivariate time series sequence
        window_size (int): the length of time window
        dimension (int): the dimension of time series subsequence
        step (int): the step of sliding time window
        alpha (int): the number of extracted components

    Returns:
        sequence_i (arr): the dissimilarity sequence
    """

    meanRemoved = T
    TT = meanRemoved
    d = dimension
    num = T.shape[0]
    df_subspace = []

    nc = Decimal(1)
    # Subsequence segmentation
    for ii in range(0,num - window_size - 1,step):
        TT_i = TT[ii:ii + window_size, :]
        # Calculate first k principal components of each subsequence
        recon_TT = pca(TT_i, alpha)
        subspace = recon_TT
        df_subspace.append(subspace)
    neighbor = 5
    cc = np.zeros(len(df_subspace) - neighbor)

    # Calculate dissimilarity of adjacent subsequences with principal components
    for t in range(neighbor,len(df_subspace)):
        c_sum = 0
        for t1 in range(1, neighbor+1):
            # Calculation of adjacent subsequence matrix: C = V2^T*V1*V1^T*V2
            V1_subspace = df_subspace[t-t1]
            V2_subspace = df_subspace[t]
            V1t = V1_subspace.transpose()
            V2t = V2_subspace.transpose()
            C = np.dot(np.dot(np.dot(V2t, V1_subspace), V1t), V2_subspace)
            # Calculate minimum eigenvalues of C
            eigVals, eigVects = np.linalg.eig(C)
            min_lambda = min(eigVals).real
            if nc > Decimal(min_lambda) and Decimal(min_lambda) != 0:
                print(min_lambda)
                nc = min_lambda
            c_sum = c_sum + min_lambda
        # lambda_avg: average of neighbor min_lambda
        cc[t - neighbor] = c_sum/neighbor

    # Dissimilarity sequence claculation
    sequence_i = np.zeros(len(cc))
    for tt in range(len(cc)):
        # dissimilarity = |1-lambda_avg|
        sequence_i[tt] = Decimal(abs(1 - cc[tt]))
    scaler = MinMaxScaler()
    sequence_i = scaler.fit_transform(sequence_i.reshape(-1,1).tolist())
    sequence_i = sequence_i.reshape(1,-1)[0]
    return sequence_i

def train_and_test_set_split(data, split_length):
    """
    Normalization and trian and test set split

    Args:
        data (arr): the sequence of dissimilarity
        split_length (int): the length of each input sequence

    Returns:
        x_train, y_train, x_test, y_test (list): the input and output of train and test set
    """
    sequence_lenghth = split_length + 1
    result = []
    for index in range(data.shape[0] - sequence_lenghth):
        result.append(data[index: index + sequence_lenghth])
    result = np.array(result)
    row = round(0.3 * result.shape[0])
    train = result[:int(row), :].copy()
    # repeat to multiply train set
    train = np.repeat(train, 3, axis=0)

    test = result
    plt_train = train[:, -1]
    result_unshuffle = result[:,-1]
    plt.subplot(2, 1, 1)
    plt.plot(result_unshuffle, label='True Data',color ='black')
    plt.xlim(0, len(result_unshuffle))
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(plt_train, label='True Data',color ='black')
    plt.xlim(0, len(plt_train))
    plt.legend()
    plt.show()
    # pd.DataFrame(result_unshuffle).to_excel('Result/normalized_similarity_video.xlsx')

    x_test = test[:, :-1]
    y_test = test[:, -1]
    # disorder the train set
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_train, y_train, x_test, y_test

def LSTM_modeling(x_train,y_train,x_test):
    """
    LSTM modeling and predicting anomaly score point_by_point

    Args:
        x_train, y_train, x_test (arr): the input and output of train and test set

    Returns:
        anomaly_score (arr): the anomaly score of dissimilarity sequence

    """
    model = Sequential()
    model.add(LSTM(input_dim = 1, output_dim=split_length, return_sequences=True))
    model.add(LSTM(20, return_sequences= False))
    model.add(Dense(output_dim = 1))
    model.add(Activation('tanh'))
    rmsprop = optimizers.RMSprop(lr=0.003)
    model.compile(loss='mae', optimizer= rmsprop)
    model.summary()
    model.fit(x_train, y_train, batch_size=256, epochs=100, validation_split=0.1)

    # LSTM prediction
    predictions = model.predict(x_test)
    predictions = np.reshape(predictions, (predictions.size,))
    # pd.DataFrame(predictions).to_excel('Result/LSTMprediction_video.xlsx')

    # plotting prediction
    plot_results(predictions, y_test)
    anomaly_score = abs(y_test - predictions)

    return anomaly_score

def plot_results(predicted_data, real_data):
    """
    Dissimilarity prediction and real value plotting

    Args:
        predicted_data (arr): prediction of dissimilarity test set
        real_data (arr): real value of dissimilarity test set

    """
    plt.subplot(2, 1, 1)
    plt.plot(real_data, label='True Data',color ='black', zorder=1)
    plt.plot(predicted_data, label='Prediction',color ='red', zorder=2)
    plt.xlim(0, len(predicted_data))
    plt.legend()
    plt.show()

def anomaly_detection(anomaly_score,dataset,win,step,split_length):
    """
    Anomaly detection and matching abnormal dissimilarity values
        with raw multivariate time series

    Args:
        anomaly_score (arr): deviation of dissimilarity test set
        dataset (arr): the whole multivariate time series sequence
        win (int): the length of time window
        step (int): the step of sliding time window
        split_length (int): length of input sequence

    Returns:
        anomaly_result (arr): labeled multivariate time series
    """
    # threshold identification under Chebyshev's inequality
    mean = anomaly_score.mean()
    std = anomaly_score.std()
    threshold = mean+3*std
    print(threshold)
    anomaly_identify = np.zeros(len(anomaly_score))
    anomaly_result = []
    for i in range(dim):
        anomaly_result.append(np.zeros(len(dataset)))
    binary_label = np.zeros(len(dataset))
    # matching anomaly points to raw multivariate time series
    for i in range(len(anomaly_score)):
        if anomaly_score[i] >= threshold:
            anomaly_identify[i] = anomaly_score[i]
            start = (i+split_length) * step
            for j in range(win):
                if start+j < len(anomaly_result[0]):
                    if anomaly_result[0][start + j] == 0:
                        binary_label[start + j] = 1
                        for i in range(dim):
                            anomaly_result[i][start + j] = dataset[start + j, i]
    dt = []
    for i in range(dim):
        dt.append(anomaly_result[i][np.nonzero(anomaly_result[i])])

    # anomaly score plotting
    ano = anomaly_identify[np.nonzero(anomaly_identify)]
    plt.subplot(2, 1, 1)
    plt.plot(range(len(anomaly_score)), anomaly_score, color='black', zorder=1)
    plt.plot(np.nonzero(anomaly_identify)[0],ano ,1, color='red', zorder=2)
    plt.xlim(0, len(anomaly_score))
    plt.show()
    # output the abrupt change points
    anomaly_score_label = np.vstack((anomaly_score, anomaly_identify))
    anomaly_score_label = anomaly_score_label.transpose()
    anomaly_score_label = pd.DataFrame(anomaly_score_label, columns=['anomaly_score1', 'anomaly_score2'])
    # pd.DataFrame(anomaly_score_label).to_excel('Result/anomaly_score_video.xlsx')

    # plotting anomaly in raw dataset
    for i in range(dim):
        plt.subplot(dim, 1, i+1)
        plt.plot(range(len(dataset)), dataset[:,i],color ='black', zorder=1)
        plt.scatter(np.nonzero(anomaly_result[i])[0], dt[i], 1, color ='red', zorder=2)
        plt.xlim(0, len(dataset))
    plt.show()
    return anomaly_result

if __name__ == '__main__':
    # initial parameters
    dataset = pd.read_excel('C:/Users/65702/Desktop/code and datasets/Dataset/SD3-grouting.xlsx', encoding='utf-8')
    dataset = np.array(dataset.iloc[:,1:])
    # determine the extracted number k of principal component
    dim = dataset.shape[1]
    alpha = dim-1
    win = 140   # set time window length
    step = math.ceil(win / 10)  # set the step of sliding time window
    split_length = 10  # length of input sequence

    # transforming into dissimilarity sequence
    X_scaled = datatrans(dataset, win, dim, step,alpha)

    # train and testing set segmentation
    x_train, y_train, x_test, y_test = train_and_test_set_split(X_scaled, split_length)
    print('shape_x_train', np.array(x_train).shape)
    print('shape_y_train', np.array(y_train).shape)
    print('shape_x_test', np.array(x_test).shape)
    print('shape_y_test', np.array(y_test).shape)

    # LSTM modeling and prediction
    anomaly_score = LSTM_modeling(x_train,y_train,x_test)

    # Anomaly score plot comparison
    AD_result = anomaly_detection(anomaly_score,dataset,win,step,split_length)

    # Anomaly-labeled dataset output
    AD_result_ful = np.zeros((dim,len(dataset)))
    for i in range(dim):
        AD_result_ful[i] = AD_result[i]
    AD_result_ful = AD_result_ful.transpose()
    column_name = []
    for i in range(dim):
        column_name.append(''.join(('anomaly',str(i))))
    AD_result_ful = pd.DataFrame(AD_result_ful,columns=column_name)
    dataset2 = pd.concat([pd.DataFrame(dataset), AD_result_ful], axis=1)
    pd.DataFrame(dataset2).to_excel('Result/SD3-grouting1.xlsx')