import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D,Flatten, LeakyReLU, Dropout, GlobalMaxPooling1D
from keras.optimizers import Adam



class BaseModelMultisteps:
    def __init__(self, model_name, in_steps, out_steps):
        self.model = self.build_model(in_steps, out_steps)
        self.model_root = 'models/'
        self.model_name = model_name
        self.model_path = self.model_root + self.model_name

    def build_model(self, in_steps, out_steps):
        raise NotImplementedError("Subclasses must implement the build_model() method.")

    def train(self, train_X, train_y, test_X, test_y, verbose=1, epochs=100):
        model = self.model
        checkpoint = ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True, save_weights_only=True)
        model.fit(train_X, train_y, epochs=epochs, verbose=verbose, validation_data=(test_X, test_y), callbacks=[checkpoint])
        self.model = model

    def evaluate(self, test_X, test_y):
        model = self.model
        model.load_weights(self.model_path)
        # Assuming you have your true values in 'y_true' and predicted values in 'y_pred'
        pred_y = model.predict(test_X)
        RMSE_scores = []
        R2_scores = []
        MAE_scores = []
        for i in range(test_y.shape[1]):
            mse = mean_squared_error(test_y[:, i], pred_y[:,  i])
            rmse = np.sqrt(mse)
            RMSE_scores.append(rmse)

        for i in range(test_y.shape[1]):
            r2 = r2_score(test_y[:, i], pred_y[:,  i])
            R2_scores.append(r2)

        for i in range(test_y.shape[1]):
            mae = mean_absolute_error(test_y[:, i], pred_y[:,  i])
            MAE_scores.append(mae)

        total_score = 0
        for row in range(test_y.shape[0]):
            for col in range(pred_y.shape[1]):
                total_score += (test_y[row, col] - pred_y[row, col] ** 2)
                total_score = np.sqrt(total_score / (test_y.shape[0] * test_y.shape[1]))

        return RMSE_scores, MAE_scores, R2_scores, total_score, pred_y
    
    def plot(self, in_steps = 27, out_steps = 7):
        """
        this works only for non CNN models only!
        
        """
        df = pd.read_csv('data.csv')
        df.drop('Unnamed: 0', inplace = True, axis = 1)
        df.columns = ['Date', 'sfu']
        df.Date = pd.to_datetime(df.Date)
        lstm2_multistep = self.build_model(in_steps= in_steps, out_steps=out_steps)
        lstm2_multistep.load_weights(self.model_path)
        
        start = 600
        last_value = df['sfu'][start]
        input = df['sfu'][start-in_steps:start]

        input = np.array(input).reshape(1, in_steps, 1)
        predictions = lstm2_multistep.predict(input)


        # predictions = np.array(predictions.tolist().insert(0, last_value))

        plt.subplot(3,2,1)
        start = 600
        last_value = df['sfu'][start]
        input = df['sfu'][start-in_steps:start]

        input = np.array(input).reshape(1, in_steps, 1)
        predictions = lstm2_multistep.predict(input)
        plt.plot(df[['Date']][start-50:start+1], df[['sfu']][start-50:start+1])
        plt.plot(df[['Date']][start:start+7], df[['sfu']][start:start+7])
        plt.plot(df[['Date']][start:start+7],predictions.reshape(out_steps, 1), color = 'red')

        plt.subplot(3,2,2)
        start = 100
        last_value = df['sfu'][start]
        input = df['sfu'][start-in_steps:start]

        input = np.array(input).reshape(1, in_steps, 1)
        predictions = lstm2_multistep.predict(input)
        plt.plot(df[['Date']][start-50:start+1], df[['sfu']][start-50:start+1])
        plt.plot(df[['Date']][start:start+7], df[['sfu']][start:start+7])
        plt.plot(df[['Date']][start:start+7],predictions.reshape(out_steps, 1), color = 'red')

        plt.subplot(3,2,3)
        start = 200
        last_value = df['sfu'][start]
        input = df['sfu'][start-in_steps:start]

        input = np.array(input).reshape(1, in_steps, 1)
        predictions = lstm2_multistep.predict(input)
        plt.plot(df[['Date']][start-50:start+1], df[['sfu']][start-50:start+1])
        plt.plot(df[['Date']][start:start+7], df[['sfu']][start:start+7])
        plt.plot(df[['Date']][start:start+7],predictions.reshape(out_steps, 1), color = 'red')

        plt.subplot(3,2,4)
        start = 654
        last_value = df['sfu'][start]
        input = df['sfu'][start-in_steps:start]

        input = np.array(input).reshape(1, in_steps, 1)
        predictions = lstm2_multistep.predict(input)
        plt.plot(df[['Date']][start-50:start+1], df[['sfu']][start-50:start+1])
        plt.plot(df[['Date']][start:start+7], df[['sfu']][start:start+7])
        plt.plot(df[['Date']][start:start+7],predictions.reshape(out_steps, 1), color = 'red')


        plt.subplot(3,2,5)
        start = 156
        last_value = df['sfu'][start]
        input = df['sfu'][start-in_steps:start]

        input = np.array(input).reshape(1, in_steps, 1)
        predictions = lstm2_multistep.predict(input)
        plt.plot(df[['Date']][start-50:start+1], df[['sfu']][start-50:start+1])
        plt.plot(df[['Date']][start:start+7], df[['sfu']][start:start+7])
        plt.plot(df[['Date']][start:start+7],predictions.reshape(out_steps, 1), color = 'red')


        plt.subplot(3,2,6)
        start = 546
        last_value = df['sfu'][start]
        input = df['sfu'][start-in_steps:start]

        input = np.array(input).reshape(1, in_steps, 1)
        predictions = lstm2_multistep.predict(input)
        plt.plot(df[['Date']][start-50:start+1], df[['sfu']][start-50:start+1])
        plt.plot(df[['Date']][start:start+7], df[['sfu']][start:start+7])
        plt.plot(df[['Date']][start:start+7],predictions.reshape(out_steps, 1), color = 'red')
        
        
    def plot_CNN(self, in_steps = 27, out_steps = 7):
        """
        this works only for  CNN models only!
        
        """
        df = pd.read_csv('data.csv')
        df.drop('Unnamed: 0', inplace = True, axis = 1)
        df.columns = ['Date', 'sfu']
        df.Date = pd.to_datetime(df.Date)
        lstm2_multistep = self.build_model(in_steps= in_steps, out_steps=out_steps)
        lstm2_multistep.load_weights(self.model_path)
        
        start = 600
        last_value = df['sfu'][start]
        input = df['sfu'][start-in_steps:start]

        input = np.array(input).reshape(1, 1, in_steps, 1)
        predictions = lstm2_multistep.predict(input)


        # predictions = np.array(predictions.tolist().insert(0, last_value))

        plt.subplot(3,2,1)
        start = 600
        last_value = df['sfu'][start]
        input = df['sfu'][start-in_steps:start]

        input = np.array(input).reshape(1, 1, in_steps, 1)
        predictions = lstm2_multistep.predict(input)
        plt.plot(df[['Date']][start-50:start+1], df[['sfu']][start-50:start+1])
        plt.plot(df[['Date']][start:start+7], df[['sfu']][start:start+7])
        plt.plot(df[['Date']][start:start+7],predictions.reshape(out_steps, 1), color = 'red')

        plt.subplot(3,2,2)
        start = 100
        last_value = df['sfu'][start]
        input = df['sfu'][start-in_steps:start]

        input = np.array(input).reshape(1, 1, in_steps, 1)
        predictions = lstm2_multistep.predict(input)
        plt.plot(df[['Date']][start-50:start+1], df[['sfu']][start-50:start+1])
        plt.plot(df[['Date']][start:start+7], df[['sfu']][start:start+7])
        plt.plot(df[['Date']][start:start+7],predictions.reshape(out_steps, 1), color = 'red')

        plt.subplot(3,2,3)
        start = 200
        last_value = df['sfu'][start]
        input = df['sfu'][start-in_steps:start]

        input = np.array(input).reshape(1, 1, in_steps, 1)
        predictions = lstm2_multistep.predict(input)
        plt.plot(df[['Date']][start-50:start+1], df[['sfu']][start-50:start+1])
        plt.plot(df[['Date']][start:start+7], df[['sfu']][start:start+7])
        plt.plot(df[['Date']][start:start+7],predictions.reshape(out_steps, 1), color = 'red')

        plt.subplot(3,2,4)
        start = 654
        last_value = df['sfu'][start]
        input = df['sfu'][start-in_steps:start]

        input = np.array(input).reshape(1, 1, in_steps, 1)
        predictions = lstm2_multistep.predict(input)
        plt.plot(df[['Date']][start-50:start+1], df[['sfu']][start-50:start+1])
        plt.plot(df[['Date']][start:start+7], df[['sfu']][start:start+7])
        plt.plot(df[['Date']][start:start+7],predictions.reshape(out_steps, 1), color = 'red')


        plt.subplot(3,2,5)
        start = 156
        last_value = df['sfu'][start]
        input = df['sfu'][start-in_steps:start]

        input = np.array(input).reshape(1, 1, in_steps, 1)
        predictions = lstm2_multistep.predict(input)
        plt.plot(df[['Date']][start-50:start+1], df[['sfu']][start-50:start+1])
        plt.plot(df[['Date']][start:start+7], df[['sfu']][start:start+7])
        plt.plot(df[['Date']][start:start+7],predictions.reshape(out_steps, 1), color = 'red')


        plt.subplot(3,2,6)
        start = 546
        last_value = df['sfu'][start]
        input = df['sfu'][start-in_steps:start]

        input = np.array(input).reshape(1, 1, in_steps, 1)
        predictions = lstm2_multistep.predict(input)
        plt.plot(df[['Date']][start-50:start+1], df[['sfu']][start-50:start+1])
        plt.plot(df[['Date']][start:start+7], df[['sfu']][start:start+7])
        plt.plot(df[['Date']][start:start+7],predictions.reshape(out_steps, 1), color = 'red')

class MLP2_multistep(BaseModelMultisteps):
    def build_model(self, in_steps, out_steps):   
        
        model = Sequential()
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(out_steps))
        
        optimizer = Adam(lr=0.001)
        model.compile(optimizer=optimizer, loss='mse')
        print("LSTM2_multistep model built!")

        return model


class LSTM_multistep(BaseModelMultisteps):
    def build_model(self, in_steps, out_steps):   
        
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(in_steps, 1)))
        model.add(Dense(out_steps))
        
        optimizer = Adam(lr=0.001)
        model.compile(optimizer=optimizer, loss='mse')
        print("LSTM2_multistep model built!")

        return model



#so far this has been observed to be the best model

class LSTM2_multistep(BaseModelMultisteps):
    def build_model(self, in_steps, out_steps):   
        
        model = Sequential()
        model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(in_steps, 1)))
        model.add(LSTM(32, activation='relu'))
        model.add(Dense(out_steps))
        
        optimizer = Adam(lr=0.001)
        model.compile(optimizer=optimizer, loss='mse')
        print("LSTM2_multistep model built!")

        return model
    

class LSTM3_multistep(BaseModelMultisteps):
    def build_model(self, in_steps, out_steps):   
        
        model = Sequential()
        model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(in_steps, 1)))
        model.add(LSTM(64, activation='relu',  return_sequences= True))
        model.add(LSTM(32, activation='relu'))
        model.add(Dense(out_steps))
        
        optimizer = Adam(lr=0.001)
        model.compile(optimizer=optimizer, loss='mse')
        print("LSTM2_multistep model built!")

        return model
    
    
    
class LSTMCNN_multistep(BaseModelMultisteps):
        
    def build_model(self, in_steps, out_steps):
        # Define model
        
        model_cnn_lstm = Sequential()
        model_cnn_lstm.add(TimeDistributed(Conv1D(filters=16, kernel_size=14, activation='relu'), input_shape=(None, in_steps, 1)))
        model_cnn_lstm.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model_cnn_lstm.add(TimeDistributed(Conv1D(filters=32, kernel_size=4, activation='relu')))
        model_cnn_lstm.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model_cnn_lstm.add(TimeDistributed(Flatten()))
        model_cnn_lstm.add(Dense(128))
        model_cnn_lstm.add(LSTM(64, activation='relu', return_sequences= True))
        model_cnn_lstm.add(LSTM(32, activation='relu'))
        model_cnn_lstm.add(Dense(out_steps))
        model_cnn_lstm.compile(loss='mse', optimizer='adam')

        optimizer = Adam(lr=0.001)
        model_cnn_lstm.compile(optimizer=optimizer, loss='mse')
        return model_cnn_lstm
       


