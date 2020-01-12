import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import pydot
import os


def unpick(pick_file):
    with open("Project/processed_data/"+pick_file, "rb") as file:
        xs, ys = pickle.load(file)
    return xs, ys


RANDOM_STATE = 2020

def saveTree(tree):
    export_graphviz(tree, out_file='tree.dot', rounded=True,
                    precision=1)  # Use dot file to create a graph
    (graph,) = pydot.graph_from_dot_file('tree.dot')  # Write graph to a png file
    graph.write_png('tree.png')


def encode_targets(y):
    enc = LabelEncoder()
    cy = enc.fit_transform(y)
    return tf.keras.utils.to_categorical(cy, len(np.unique(cy)))


def transform_targets(y):
    ymean = np.mean(y)
    ystd = np.std(y)
    return [(tg - ymean) / ystd for tg in y]


def mse_clf(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(criterion='entropy', n_jobs=-1, n_estimators=100, random_state=RANDOM_STATE)
    rf = rf.fit(X_train, y_train)
    yhat = rf.predict(X_test)
    col = np.argmax(y_test, axis=1)
    row = range(len(y_test))
    return mean_squared_error(y_test[row, col], yhat[row, col])

def mse_reg(X_train, X_test, y_train, y_test):
    rf = RandomForestRegressor(criterion='mse', n_jobs=-1, n_estimators=100, random_state=RANDOM_STATE)
    rf = rf.fit(X_train, y_train)
    yhat = rf.predict(X_test)
    return mean_squared_error(y_test, yhat)


def run_rf(xs, ys):
    cy = encode_targets(ys)
    ry = transform_targets(ys)

    cX_train, cX_test, cy_train, cy_test = train_test_split(xs, cy, test_size=.5, shuffle=False)
    rX_train, rX_test, ry_train, ry_test = train_test_split(xs, ry, test_size=.5, shuffle=False)

    c_mse = mse_clf(cX_train, cX_test, cy_train, cy_test)
    r_mse = mse_reg(rX_train, rX_test, ry_train, ry_test)

    print('---------')
    print('Dataset: ', filename)
    print('MSE clf: ', c_mse)
    print('MSE reg: ', r_mse)



if __name__ == '__main__':
    filenames = []
    for file in os.listdir('Project/processed_data'):
        filenames.append(file)

    for filename in filenames:
        xs, ys = unpick(filename)

        shape = xs.shape
        if len(shape) > 1:
            d = shape[1:]
            xs = xs.reshape(shape[0], np.prod(d))

        run_rf(xs, ys)

