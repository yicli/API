import numpy as np
import pickle
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale
from sklearn.datasets import load_svmlight_file as svm
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def loadtxt(in_name, out_name, sep, target_last, scale_x=True, header=None, onehot=None):
    print("preprocessing", out_name)
    data = pandas.read_csv("Project/datasets/"+in_name, sep=sep, header=header)
    maxrow = min(data.shape[0], 2000)
    data = data.sample(maxrow)
    if target_last:
        xs = np.array(data.iloc[:, 0:-1])
        ys = np.array(data.iloc[:, -1])
    else:
        xs = np.array(data.iloc[:, 1:])
        ys = np.array(data.iloc[:, 0])
    if onehot is not None:
        enc = OneHotEncoder(sparse=False)
        x_cat = enc.fit_transform(xs[:, onehot])
        xs = xs[:, [col not in onehot for col in range(xs.shape[1])]]
    if scale_x:
        xs = scale(xs)
    if onehot is not None:
        xs = np.concatenate((xs, x_cat), axis=1)
    with open("Project/processed_data/"+out_name, "wb") as file:
        pickle.dump([xs, ys], file)


def unpick(pick_file):
    with open("Project/processed_data/"+pick_file, "rb") as file:
        xs, ys = pickle.load(file)
    return xs, ys


if __name__ == "__main__":
    # Aloi  # TODO: randomise sample and convert to ndarray
    x, y = svm("Project/datasets/aloi.scale")
    x = x[0:2000, :].todense()
    y = y[0:2000]
    with open("Project/processed_data/aloi.pkl", "wb") as f:
        pickle.dump([x, y], f)
    del [x, y]

    # Isolet
    loadtxt("isolet5.data", "isolet.pkl", ", ", True, scale_x=False)

    # Letter Recognition
    loadtxt("letter-recognition.data", "letter.pkl", ",", False)
    with open("Project/processed_data/letter.pkl", "rb") as f:
        x, y = pickle.load(f)
    y = np.array([ord(item)-ord('A') for item in y])
    with open("Project/processed_data/letter.pkl", "wb") as f:
        pickle.dump([x, y], f)
    del [x, y]

    # Sensorless drive
    loadtxt("Sensorless_drive_diagnosis.txt", "sensorless.pkl", " ", True)

    # Year Prediction
    loadtxt("YearPredictionMSD.txt", "year.pkl", ",", False)

    # Boston Housing
    (x, y), _ = tf.keras.datasets.boston_housing.load_data()
    x = x.astype(np.float32)
    print("scaling boston")
    x = scale(x)
    with open("Project/processed_data/boston.pkl", "wb") as f:
        pickle.dump([x, y], f)
    del [x, y]

    # CCPP
    loadtxt("CCPP.csv", "ccpp.pkl", ",", True, header=0)

    # Forest Fire
    loadtxt("forestfires.csv", "forest.pkl", ",", True, header=0, onehot=[2, 3])

    # Physioco
    loadtxt("physioco.csv", "phsioco.pkl", ",", False, header=0)

    # CT Slice
    loadtxt("slice_localization_data.csv", "ctslice.pkl", ",", True, header=0)

    # MNIST
    (x, y), (_, _) = tf.keras.datasets.mnist.load_data()
    rows = np.random.choice(len(x), 2000, False)
    x = x[rows, :]
    y = y[rows]
    x = x.astype('float32')/255
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    with open("Project/processed_data/mnist.pkl", "wb") as f:
        pickle.dump([x, y], f)
    del [x, y, rows]

    # CIFAR 10
    (x, y), (_, _) = tf.keras.datasets.cifar10.load_data()
    rows = np.random.choice(len(x), 2000, False)
    x = x[rows, :]
    y = y[rows].reshape(-1)
    x = x.astype('float32')/255
    with open("Project/processed_data/cifar10.pkl", "wb") as f:
        pickle.dump([x, y], f)
    del [x, y, rows]

    # CIFAR 100
    (x, y), (_, _) = tf.keras.datasets.cifar100.load_data()
    rows = np.random.choice(len(x), 2000, False)
    x = x[rows, :]
    y = y[rows].reshape(-1)
    x = x.astype('float32')/255
    with open("Project/processed_data/cifar100.pkl", "wb") as f:
        pickle.dump([x, y], f)
    del [x, y, rows]

    # Crowd Flower
    dat = pandas.read_csv("Project/datasets/text_emotion.csv", sep=',', header=0)
    dat = dat.sample(2000)
    x = dat.iloc[:, 3]
    y = dat.iloc[:, 1]
    tfidf_cf = TfidfVectorizer(use_idf=True,
                               dtype=np.float32,
                               max_features=2000,
                               smooth_idf=True,
                               max_df=0.99)
    x = tfidf_cf.fit_transform(x).todense()
    x = np.array(x)
    le = LabelEncoder()
    y = le.fit_transform(y)
    with open("Project/processed_data/crowdflower.pkl", "wb") as f:
        pickle.dump([x, y], f)
    del [x, y, dat]

    # IMDB
    (x, y), _ = tf.keras.datasets.imdb.load_data()
    rows = np.random.choice(len(x), 2000, False)
    x = x[rows]
    y = y[rows]
    # convert sequences of word indices to string
    for i in range(2000):
        x[i] = " ".join(map(str, x[i]))
    tfidf = TfidfVectorizer(use_idf=True,
                            dtype=np.float32,
                            max_features=2000,
                            smooth_idf=True,
                            max_df=0.99)
    x = tfidf.fit_transform(x).todense()
    x = np.array(x)
    with open("Project/processed_data/imdb.pkl", "wb") as f:
        pickle.dump([x, y], f)
    del [x, y, rows]
