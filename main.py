import os
import glob
import numpy as np
from utils.data_utils import get_features_from_match_data
from utils import tracab_dat_reader as reader
from models.LSTM import initialize_lstm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# np.random.seed(7)
import pandas as pd
import matplotlib.pyplot as plt

# def print_length_example(filename):
#     data_file = os.getcwd() + filename
#     data = reader.read_match_file(data_file)
#     # print("Number of frames: ", data[0].Length())
#     return data
#
# def consume_data(frames):
#     df = np.zeros((len(frames), 5))
#     target = np.zeros(len(frames))
#     for idx, frame in enumerate(frames):
#         df[idx] = [frame.ball.height, frame.ball.speed, frame.ball.pos.x, frame.ball.pos.z, frame.ball_possession_team.value]
#         target[idx] = frame.ball_state.value
#
#     return df, target

# create a differenced series
# def diff(data, interval=1):
#     diff_list = list()
#     for i in range(interval, len(data)):
#         value = data[i] - data[i - interval]
#         diff_list.append(value)
#
#     return np.array(diff_list)

# Define/Initialize the LSTM model and train it with the given training data
# def lstm_fit(training_data, target, batch, epochs):
#     X_train = training_data
#     y_train = target
#     n_features = len(X_train[0])
#     X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
#     model = Sequential()
#     # model.add(Input(shape=(5, 100)))
#     model.add(LSTM(8, input_shape=(len(X_train), n_features)))
#     model.add(Dropout(0.2))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     # print(model.summary())
#     for epoch in range(epochs):
#         model.fit(X_train, y_train, epochs=1, batch_size=100)
#         # model.reset_states()
#     # Add IO exception handling
#     model.save('my_model.h5')
#     return model

# # Function for predicting the ball state
# def predict_ball_state_lstm(model, X):
#     X = X.reshape(1, 1, len(X))
#     y_pred = model.predict_classes(X)
#     return y_pred[0]

# # Function updates the model with the training data accumulated over 100 frames
# def update_model(X, y, batch_size, epochs):
#     # X, y = train[:, 0:-1], train[:, -1]
#     # Add exception handling
#     model = load_model('my_model.h5')
#     X = X.reshape(X.shape[0], 1, X.shape[1])
#     for i in range(epochs):
#         model.fit(X,y, epochs=1, batch_size=100)
#         # model.reset_states()
#     model.save('my_model.h5')


def plot_performance(model, type="loss"):
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
    plt.show()


def create_sequential_dataset(data, frame_delay):
    x_data, y_data = [], []
    for i in range (0, len(data) - frame_delay - 1):
        # rows contain features plus last column as the target i.e. y
        x_data.append(data.iloc[i:(i+frame_delay), :-1])
        y_data.append(data.iloc[i+frame_delay+1, -1])

    return np.array(x_data), np.array(y_data)


# For now lets have folder as a file and we'll iterate later
def main():
    dat_files = glob.glob(os.path.join(os.getcwd(), 'data/*.dat'))
    for dat_file in dat_files:
        dat_filename_to_csv = dat_file.split('/')[-1].split('.')[0]+'.csv'
        preprocessed_file_directory = os.path.join(os.getcwd(), 'data_preprocessed')
        preprocessed_file_path = os.path.join(preprocessed_file_directory, dat_filename_to_csv)
        if not os.path.exists(preprocessed_file_directory):
            os.makedirs(preprocessed_file_directory)

        if not os.path.exists(preprocessed_file_path):
            match_data = reader.read_match_file(dat_file)
            match_dataframe = get_features_from_match_data(match_data)
            match_dataframe.to_csv(preprocessed_file_path, index=False)
        else:
            # TODO: stack the match dataframes on top of each other.
            match_dataframe = pd.read_csv(preprocessed_file_path)
        break

    x_data, y_data = create_sequential_dataset(match_dataframe, frame_delay=10)

    # model = initialize_lstm(input_shape=x_data.shape, layer_sizes=[])
    model = initialize_lstm(input_shape=x_data.shape, layer_sizes=2)
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    lstm_model = model.fit(x_data, y_data, validation_split=0.2, epochs=5)

    model.save('my_model.h5')
    plot_performance(lstm_model)
    plot_performance(lstm_model, type="accuracy")

    print('done')


if __name__ == "__main__":
    # data_folder = "/data/1061290.dat"
    main()

    # data = print_length_example("/data/1061290.dat")
    #
    # X, y = consume_data(data[0].frames)
    #
    # # df = pd.DataFrame(X, columns=['height', 'speed', 'pos_x', 'pos_z', 'possession'])
    # # df['ball_state'] = pd.Series(y)
    # # df.to_csv('./match_data.csv', index = False)
    #
    # # arr = diff(X)
    # accuracy_list = []

    # for k in range(len(X)-101):
    #     arr = diff(X[k:k+101])
    #     # X_train, X_test, y_train, y_test = train_test_split(arr, y[:len(arr)], test_size=0.2, shuffle=False)
    #     X_train, y_train = arr[1:], y[k+1:k+100]
    #     # print(k)
    #     X_test = np.array(arr[0]).reshape(1, len(arr[0]))
    #     y_test = np.array(y[0]).reshape(1, 1)
    #     if k == 0:
    #         ball_state_model = lstm_fit(X_train, y_train, 1, 5)
    #     else:
    #         if (k % 100) == 0:
    #             print("Batch Accuracy: %4.2f" % (np.sum(accuracy_list) / len(accuracy_list)))
    #             update_model(X_train, y_train, 1, 5)
    #
    #     # Just the one record in test
    #     for i in range(len(X_test)):
    #         new_val = np.zeros((1, len(X_test)))
    #         y_pred = predict_ball_state_lstm(ball_state_model, X=X_test[i])
    #         #     update the model?
    #         accuracy_list.append(y_pred[0])
    #         print('Ball_state at data point %d: Label: %d, Predicted: %d' %(k, y_test, y_pred))
    #
    #         # Update the model with the predicted data
    #         new_val = X_test[i]
    #         X_train = np.vstack((X_train, new_val))
    #         y_train = np.append(y_train, y_test[i - 1].reshape(1), axis=0)
    #
    #











