from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Dropout
from keras.models import Model


# add dims are arguments etc.
def initialize_lstm(input_shape, layer_sizes, hidden_act_func='tanh', o_act_func='sigmoid'):

    data_input = Input(shape=input_shape[1:], )

    lstm_layer = LSTM(layer_sizes, activation=hidden_act_func, return_sequences=False)(data_input)
    dense = Dense(1, activation=o_act_func)(lstm_layer)
    # model = Sequential()
    # # model.add(Input(shape=(5, 100)))
    # model.add(LSTM(8, input_shape=(len(X_train), n_features)))
    # model.add(Dropout(0.2))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model = Model(inputs=data_input, outputs=dense)
    print(model.summary())

    return model
