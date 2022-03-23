import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import numpy as np
np.random.seed(4)
import tensorflow 
tensorflow.random.set_seed(4)
from util import csv_to_dataset, history_points
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.regularizers import L1L2



# dataset
#devo fare in modo che i dati vengano prima slittati e poi normalizzati
ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset('GOGL_daily.csv')

test_split = 0.9
n = int(ohlcv_histories.shape[0] * test_split)

ohlcv_train = ohlcv_histories[:n]
tech_ind_train = technical_indicators[:n]
y_train = next_day_open_values[:n]

ohlcv_test = ohlcv_histories[n:]
tech_ind_test = technical_indicators[n:]
y_test = next_day_open_values[n:]

unscaled_y_test = unscaled_y[n:]
print(tech_ind_train.shape)
print(tech_ind_test.shape)

print(ohlcv_train.shape)
print(ohlcv_test.shape)


# model architecture

# define two sets of inputs
lstm_input = Input(shape=(history_points, 5), name='lstm_input')
dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')

# the first branch operates on the first input
x = LSTM(50, name='lstm_0')(lstm_input)
x = Dropout(0.3, name='lstm_dropout_0')(x)
lstm_branch = Model(inputs=lstm_input, outputs=x)
# new arch

# the second branch opreates on the second input
#y = Dense(60, name='tech_dense_0')(dense_input)
#y = Activation("relu", name='tech_relu_0')(y)
y = LSTM(50, name='lstm_0')(dense_input)
y = Dropout(0.3, name='lstm_dropout_0')(y)
y = Dropout(0.3, name='tech_dropout_0')(y)
y = LSTM(units = 50, activation = 'relu', return_sequences = True)(y)
y = Dropout(0.2)(y)
y = LSTM(units = 60, activation = 'relu', return_sequences = True)(y)
y = Dropout(0.3)(y)
y = LSTM(units = 80, activation = 'relu', return_sequences = True)(y)
y = Dropout(0.4)(y)
y = LSTM(units = 120, activation = 'relu')(y)
y = Dropout(0.5)(y)


technical_indicators_branch = Model(inputs=dense_input, outputs=y)

# combine the output of the two branches
combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
z = Dense(1, activation="linear", name='dense_out')(z)

# our model will accept the inputs of the two branches and
# then output a single value
model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
adam = tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=adam, loss='mse')

#early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
#mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

model.fit(x=[ohlcv_train, tech_ind_train], y=y_train, batch_size=32, epochs=120, shuffle=True, validation_split=0.1, callbacks=[es])


#earlyStop=EarlyStopping(monitor="val_loss",verbose=2,mode='min',patience=3)
#history=model.fit(xTrain,yTrain,epochs=100,batch_size=10,validation_data=(xTest,yTest) ,verbose=2,callbacks=[earlyStop])
# load the saved model
#saved_model = load_model('best_model.h5')
## evaluate the saved model
#_, train_acc = saved_model.evaluate(ohlcv_train, tech_ind_train, y_train, verbose=0)
#_, test_acc = saved_model.evaluate(ohlcv_test, tech_ind_test, y_test, verbose=0)
#print('maxACC MODEL -> Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# evaluate the current model
train_acc = model.evaluate([ohlcv_train, tech_ind_train], y_train)
print(train_acc)
#_, test_acc = model.evaluate(ohlcv_test, tech_ind_test, y_test, verbose=0)
#print('CURRENT MODEL -> Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# evaluation

y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
y_predicted = model.predict([ohlcv_histories, technical_indicators])
y_predicted = y_normaliser.inverse_transform(y_predicted)
assert unscaled_y_test.shape == y_test_predicted.shape
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
print(scaled_mse)

import matplotlib.pyplot as plt

plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')

# real = plt.plot(unscaled_y[start:end], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])

plt.show()

from datetime import datetime
model.save(f'technical_model.h5')
