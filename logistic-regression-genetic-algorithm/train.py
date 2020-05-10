"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def get_cifar10():
    """Retrieve the CIFAR dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 64
    input_shape = (3072,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(50000, 3072)
    x_test = x_test.reshape(10000, 3072)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def get_mnist():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 128
    input_shape = (784,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def get_ILPD():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes = 3
    batch_size = 16
    input_shape = (10,)

    # Get the data.

    dataset = pd.read_csv('dataset.csv')

    # print (dataset.head)

    from sklearn.impute import SimpleImputer
    dataset = dataset.replace(" ",np.NaN)
    imp = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
    imp.fit(dataset)
    dataset = pd.DataFrame(imp.transform(dataset))

    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
    X.columns = ['AGE','GENDER', 'TB','DB','ALKPHOS','SGPT','SGOT','TP','ALB','A/G']
    X = X.drop(['ALKPHOS','TB','TP','A/G'], axis = 1)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 2)

    from sklearn import metrics, preprocessing 
    from sklearn.impute import SimpleImputer

    scaler = preprocessing.StandardScaler()
    x_train = scaler.fit_transform(X_train)
    x_test = scaler.transform(X_test)

   

    return x_train, x_test, y_train, y_test  

def compile_model(network, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def train_and_score(network, dataset):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    if dataset == 'cifar10':
        x_train, x_test, y_train, y_test = get_ILPD()
    elif dataset == 'mnist':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_mnist()

    
    model = LogisticRegression(C = network['C'], tol = network['tol'], penalty = network['penalty'])

    model.fit(x_train, y_train)

    score = model.score(x_test, y_test)

    return score  # 1 is accuracy. 0 is loss.
