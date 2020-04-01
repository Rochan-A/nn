from nn.network import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import r2_score

plt.style.use('seaborn-whitegrid')

def dataset_norm_scale(path):
    """Read data forom excel file (.xls)

    Args:
        path: path to excel file

    Returns:
        X_train: features for training
        y_train: labels for training
        X_test: features for testing
        y_test: labels for testing
        sc_y: labels scaler object
    """

    # Read dataset and normalize
    dataset = pd.read_excel(path)

    X = dataset.iloc[0:8300,0:13].values
    y = dataset.iloc[0:8300:, 13:14].values
    X = X.astype('float32')
    y = y.astype('float32')

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                            random_state = 0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    sc_y = MaxAbsScaler()
    y_train = sc_y.fit_transform(y_train)
    y_test = sc_y.transform(y_test)

    return X_train, y_train, X_test, y_test, sc_y

def train_network(X_train,
        y_train,
        X_test,
        y_test,
        model,
        epoch=100,
        batch_size=1
        ):
    """Training loop

    Args:
        X_train : features for training
        X_test : features for testing
        y_train : labels for training
        y_test : labels for testing
        model: Neural Network object
        epoch (Default=100): Number of epochs
        batch_size (Default=1): Batch Size

    Returns:
        loss_history: Array of training loss
    """

    loss_history = []
    for i in range(EPOCH):
        for k in range(0, len(X_train), batch_size):
            end = min(k+batch_size, len(X_train))
            output = model.forward(X_train[k:end,:])
            model.backwards(X_train[k:end,:], \
                    y_train[k:end,:])

        loss = 0
        for j, val in enumerate(X_test):
            val = np.reshape(val,(1,len(val)))
            output = model.predict(val)
            try:
                output = output[0][0].item()
            except:
                a = 0
            loss += (((y_test[j][0] - output))**2)*0.5
        loss /= len(y_test)
        loss_history.append(loss)

        print("Epoch: ", i," Validation Loss: ", loss)

    np.savetxt('loss_history.csv', loss_history, delimiter=',')

    return loss_history

def scores(model, X_test,y_test, sc_y, path):
    """Compute MAPE and RSQ.

    Args:
        model: Neural Network object
        X_test: features for testing
        y_test: labels fro testing
        sc_y: labels scaler object
        path: location to save predicted labels
    """

    y_pred = model.predict(X_test)

    y_pred = sc_y.inverse_transform(y_pred)
    y_test = sc_y.inverse_transform(y_test)

    Data = np.concatenate((y_test,y_pred), axis = 1 )
    df = pd.DataFrame(Data, columns = ['Real Values', 'Predicted Values'])
    df.to_excel(path, index = False, header=True)

    mape = 0
    cnt = 0
    for i,_ in enumerate(y_test):
        mape += abs(y_pred[i] - y_test[i])/abs(y_test[i])
        cnt += 1

    print("RSQ: ", r2_score(y_test, y_pred))
    print("MAPE: ", (mape*100)/cnt)

if __name__ == '__main__':
    # Neural Network Parameters
    INPUT = 13
    OUTPUT = 1

    # Training Parameters
    EPOCH = 400
    BATCH_SIZE = 4

    # Initialize Neural Network
    nn = pytorch_network(bias=True, batch_size=BATCH_SIZE)

    # Add layers
    nn.add_linear(INPUT, 50, True)
    nn.add_linear(50, OUTPUT)

    # Read data and transform
    X_train, y_train, X_test, y_test, sc_y = \
        dataset_norm_scale('../dataset/mpp_dataset_v1_13-inputs.xls')

    # Train
    loss_history = train_network(X_train,y_train,X_test,y_test, model = nn,
                                        epoch=EPOCH, batch_size=BATCH_SIZE)

    # Compute MAPE and RSQ value
    save_path = r'C:\Users\Zakaria\Desktop\Inbox\export_dataframe_2.xlsx'
    scores(nn, X_test, y_test, sc_y, save_path)

    # Plot loss history
    plt.scatter(np.arange(len(loss_history)), loss_history, alpha=0.5)
    plt.show()
