from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler



main = tkinter.Tk()
main.title("An In-Depth Historical Examination of NIFTY Indices for Strategic Investment Decision-Making")
main.geometry("1000x650")
title = tkinter.Label(main, text="An In-Depth Historical Examination of NIFTY Indices for Strategic Investment Decision-Making",justify='center')

title.grid(column=0, row=0)
font=('times', 15, 'bold')
title.config(bg='skyblue', fg='white')
title.config(font=font)
title.config(height=3,width=100)
title.place(x=50,y=5)

global df,df_rev
global x_train,y_train,X_train,y_test
global model,scaled_data
import pandas as pd
def upload():
    global df,df_rev
    path  = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0',tkinter.END)
    df=pd.read_csv(path)
    text.insert(tkinter.END,path+'Loaded\n\n')
    text.insert(tkinter.END,str(df.head()))

def preprocessing():
    global df,df_rev
    df=df.drop(['Index Name','Open','High','Low'],axis=1)
    df_rev = df.iloc[::-1].reset_index(drop=True) 
    ma100=df_rev.Close.rolling(100).mean()
    text.insert(tkinter.END,'\n\n------preprocess done sucessfully-----')
    text.insert(tkinter.END,str(df_rev))
    plt.figure(figsize=(12,6))
    plt.plot(df_rev.Close)
    plt.plot(ma100,'red')
    plt.show()

from sklearn.preprocessing import MinMaxScaler       # import lib
scaler = MinMaxScaler(feature_range=(0,1)) 
def splitting():
    global x_train,y_train
    df_train= pd.DataFrame(df_rev['Close'][0:int(len(df_rev)*0.70)])
    df_test = pd.DataFrame(df_rev['Close'][int(len(df_rev)*0.70):int(len(df_rev))])
    # Scale down the data
    df_train_arr= scaler.fit_transform(df_train)
    x_train = []
    y_train = []

    for i in range(100,df_train_arr.shape[0]):        # df_train_arr[101:1]   # here we take only from df_train
        x_train.append(df_train_arr[i-100: i])
        y_train.append(df_train_arr[i,0])
    x_train,y_train = np.array(x_train),np.array(y_train)
    text.insert(tkinter.END,'\n\n----------------splitting --------------')
    text.insert(tkinter.END,'\n x train',str(x_train))
    text.insert(tkinter.END,'\n y train',str(y_train))
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
def model_file():
    global model,scaled_data
    global x_train,X_test,y_train,X_train,y_test,model_path

    model_path = 'model/keras_model.h5'

    if os.path.exists(model_path):
        # Load the model if it exists
        model = load_model(model_path)
        print("Model loaded successfully.")

    else:

        data = df['Close'].values.reshape(-1, 1)
        # 3. Normalize the data (LSTM performs better with normalized data)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        # 4. Prepare the training and testing data
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        def create_dataset(data, time_step=60):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)
        # Reshape the data for the LSTM model
        time_step = 60  # LSTM will look at the last 60 days to predict the next
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        # Reshape the input to be [samples, time_steps, features] for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # 5. Build the LSTM Model
        model = Sequential()

        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))  # Dropout for regularization
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))  # Prediction of the next stock price
        # 6. Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        # 7. Train the model
        model.fit(X_train, y_train, batch_size=64, epochs=20)
        model.save('model/keras_model.h5')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def predict():
    global X_test,model
    model_path = 'model/keras_model.h5' 
    model = load_model(model_path)
    predicted_stock_price = model.predict(X_test)
    # 9. Undo the scaling to bring predicted prices back to original scale
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    # 10. Visualize the results
    plt.figure(figsize=(14, 5))
    plt.plot(df.index[-len(y_test_actual):], y_test_actual, color='blue', label='Actual Stock Price')
    plt.plot(df.index[-len(predicted_stock_price):], predicted_stock_price, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    # 11. Predict for the next 60 days
    last_60_days = scaled_data[-60:]  # Take the last 60 days from the original dataset
    X_next = [last_60_days]
    # Reshape the input for LSTM (same format as training data)
    X_next = np.array(X_next).reshape(1, 60, 1)
    # Predict the next day stock price
    predicted_price_60_days = []
    for i in range(60):
        next_price = model.predict(X_next)
        predicted_price_60_days.append(next_price)
        # Update the X_next input by removing the first element and adding the predicted next price
        X_next = np.append(X_next[:, 1:, :], next_price.reshape(1, 1, 1), axis=1)
    # Convert predictions back to original scale
    predicted_price_60_days = scaler.inverse_transform(np.array(predicted_price_60_days).reshape(-1, 1))
    text.insert(tkinter.END,predicted_price_60_days)
    
font=('times', 15, 'bold')
uploadButton = Button(main, text="Upload Dataset",command=upload)
uploadButton.config(bg='pink', fg='Black')
uploadButton.place(x=50,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="pre Processing ",command=preprocessing)
uploadButton.config(bg='pink', fg='Black')
uploadButton.place(x=220,y=100)
uploadButton.config(font=font)



uploadButton = Button(main, text="Splitting",command=splitting)
uploadButton.config(bg='pink', fg='Black')
uploadButton.place(x=390,y=100)
uploadButton.config(font=font)


uploadButton = Button(main, text="LSTM",command=model_file)
uploadButton.config(bg='pink', fg='Black')
uploadButton.place(x=500,y=100)
uploadButton.config(font=font)


uploadButton = Button(main, text="Predict",command=predict)
uploadButton.config(bg='pink', fg='Black')
uploadButton.place(x=610,y=100)
uploadButton.config(font=font)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=200)
text.config(font=font1)

main.mainloop()
    