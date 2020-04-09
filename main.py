from nn.network import pytorch_network,_objective
import torch
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

    X = dataset.iloc[0:8300, 0:13].values
    y = dataset.iloc[0:8300:, 13:14].values
    X = X.astype('float32')
    y = y.astype('float32')

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=0)

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
                  epoch=100
                  ):
    """Training loop

    Args:
        X_train : features for training
        X_test : features for testing
        y_train : labels for training
        y_test : labels for testing
        model: Neural Network object
        epoch (Default=100): Number of epochs

    Returns:
        loss_history: Array of training loss
    """

    loss_history = []
    for i in range(EPOCH):

            
        model.backwards_de(X_train,
                               y_train)

        loss = torch.mean(_objective(torch.tensor(model.predict(X_test)), torch.tensor(y_test))).item()
        loss_history.append(loss)

        print("Epoch: ", i, " Validation Loss: ", loss)

    np.savetxt('loss_history.csv', loss_history, delimiter=',')

    return loss_history


def scores(model, X_test, y_test, sc_y, path):
    """Compute MAPE and RSQ.

    Args:
        model: Neural Network object
        X_test: features for testing
        y_test: labels for testing
        sc_y: labels scaler object
        path: location to save predicted labels
    """

    y_pred = model.predict(X_test)

    y_pred = sc_y.inverse_transform(y_pred)
    y_test = sc_y.inverse_transform(y_test)

    Data = np.concatenate((y_test, y_pred), axis=1)
    df = pd.DataFrame(Data, columns=['Real Values', 'Predicted Values'])
    df.to_excel(path, index=False, header=True)

    mape = 0
    cnt = 0
    for i, _ in enumerate(y_test):
        mape += abs(y_pred[i] - y_test[i]) / abs(y_test[i])
        cnt += 1

    print("RSQ: ", r2_score(y_test, y_pred))
    print("MAPE: ", (mape * 100) / cnt)


if __name__ == '__main__':
    # Neural Network Parameters
    INPUT = 13
    OUTPUT = 1

    # Differential Evolution Parameters
    POP_SIZE = 3
    K_VAL = 0.8
    CROSS_PROB = 0.75

    # Training Parameters
    EPOCH = 30
    BATCH_SIZE = 8300

    # Initialize Neural Network
    nn = pytorch_network(bias=True, batch_size=BATCH_SIZE, pop_size=POP_SIZE,
                         k_val=K_VAL, cross_prob=CROSS_PROB)

    # Add layers
    nn.add_linear(INPUT, 50, True)
    nn.add_linear(50, OUTPUT)

    # Read data and transform
    X_train, y_train, X_test, y_test, sc_y = \
        dataset_norm_scale('../dataset/mpp_dataset_v1_13-inputs.xls')

    # Train
    loss_history = train_network(X_train, y_train, X_test, y_test, model=nn,
                                 epoch=EPOCH, batch_size=BATCH_SIZE)

    # Add candidate losses to plot
    for idx in range(POP_SIZE):
        plt.plot(
            np.arange(nn.candidate_loss[idx].shape[0]), nn.candidate_loss[idx])

    # Compute MAPE and RSQ value
    save_path = r'C:\Users\Zakaria\Desktop\Inbox\export_dataframe_2.xlsx'
    scores(nn, X_test, y_test, sc_y, save_path)

    # Plot loss history
    plt.show()
    
    plt.plot(range(len(loss_history)),loss_history)
    plt.title('The Validation loss of the best candidate in each gen')
    plt.xlabel('Generation')
    plt.ylabel('Validation Loss')
    plt.show()
