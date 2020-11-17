import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.data_utils import get_features_from_match_data
from utils import tracab_dat_reader as reader
from models.LSTM import initialize_lstm
from sklearn.metrics import classification_report


# Function to predict the labels for the test set
def predict_ball_state(model, X):
    # X = X.reshape(1, 1, len(X))
    y_pred = model.predict(X)
    return y_pred


# Function to plot the learning curves
def plot_performance(model, type="loss", show_plot=True):
    plt.clf()
    loss = model.history[type]
    val_loss = model.history['val_' + type]
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'g', label='Training ' + type)
    plt.plot(epochs, val_loss, 'y', label='Validation ' +type)
    plt.title('Training and validation ' + type)
    plt.xlabel('Epochs')
    plt.ylabel(type)
    plt.legend()
    if show_plot:
        plt.show()


# Function to create the sequential dataset for an LSTM
def create_sequential_dataset(data, frame_delay):
    x_data, y_data = [], []
    for i in range (0, len(data) - frame_delay - 1):
        # rows contain features plus last column as the target i.e. y
        x_data.append(data.iloc[i:(i+frame_delay), :-1])
        y_data.append(data.iloc[i+frame_delay+1, -1])

    return np.array(x_data), np.array(y_data)


# Get the data from the files and generate a dataframe
def get_parsed_data_as_dataframe(files):
    match_dataframe_all = []

    for dat_file in files:
        dat_filename_to_csv = dat_file.split('/')[-1].split('.')[0] + '.csv'
        preprocessed_file_directory = os.path.join(os.getcwd(), 'data_preprocessed')
        preprocessed_file_path = os.path.join(preprocessed_file_directory, dat_filename_to_csv)
        if not os.path.exists(preprocessed_file_directory):
            os.makedirs(preprocessed_file_directory)

        # Check if the CSV file exists.
        if not os.path.exists(preprocessed_file_path):
            match_data = reader.read_match_file(dat_file)
            match_dataframe = get_features_from_match_data(match_data)
            match_dataframe.to_csv(preprocessed_file_path, index=False)

        # If the file exists, we can directly read from it!
        match_dataframe = pd.read_csv(preprocessed_file_path)
        match_dataframe_all.extend(match_dataframe.values)
        # pd.concat([a, b], ignore_index=True)

    match_dataframe_all = pd.DataFrame(match_dataframe_all, columns=match_dataframe.columns)

    return match_dataframe_all


def main():
    # get the location for all the dat files
    dat_files = glob.glob(os.path.join(os.getcwd(), 'data/*.dat'))
    # parse the .dat files into a dataframe.
    match_dataframe_all = get_parsed_data_as_dataframe(dat_files)

    x_data, y_data = create_sequential_dataset(match_dataframe_all, frame_delay=10)

    model = initialize_lstm(input_shape=x_data.shape, layer_sizes=2)

    # Split into train/test and reshape if required
    seq_data_len = len(x_data)
    train_split = 0.8
    x_train, y_train = x_data[:int(train_split*seq_data_len)], y_data[:int(train_split*seq_data_len)]
    x_test, y_test = x_data[int(train_split * seq_data_len) + 1:], y_data[int(train_split * seq_data_len) + 1:]
    y_test = y_test.reshape(len(y_test), 1)

    # compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the model and store performance details in lstm_model
    lstm_model = model.fit(x_train, y_train, validation_split=0.2, epochs=5)

    # Save the model weights
    model.save('my_model.h5')
    # Plot curves for loss and accuracy on the training/validation set
    plot_performance(lstm_model, show_plot=False)
    plot_performance(lstm_model, type="accuracy", show_plot=False)

    # Prediction Accuracy
    y_pred = predict_ball_state(model, x_test)

    print(classification_report(y_test, y_pred))

    print('done')


if __name__ == "__main__":
    # data_folder = "/data/1061290.dat"
    main()











