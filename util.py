import pandas as pd
from sklearn import preprocessing
import numpy as np
import csv
import datetime
from datetime import date, timedelta, datetime # Date Functions



history_points = 50


def csv_to_dataset(csv_path):
    today = date.today()
    date_today = today.strftime("%Y-%m-%d")
    data = pd.read_csv(csv_path)
    #data = data.drop('Adj Close', axis=1)
    #data = data.drop(data.shape[0] -1, axis=0)
    data.rename(columns = {'Open':'1. open', 'High':'2. high', 'Low':'3. low', 'Close':'4. close', 'Volume':'5. volume'}, inplace = True)

    data[['1. open', '2. high', '3. low', '4. close', '5. volume']]
#    columns_titles = ["2. high","3. open"]
#    data=data.reindex(columns=columns_titles)
#    columns_titles = ["3. low","2. high"]
#    data=data.reindex(columns=columns_titles)
#    data.iloc[::-1]
    
    print(data)
    #data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

    #data = data.loc[(data['date'] >= '2012-02-22') & (data['date'] < date_today)]

    #data = data.drop('date', axis=1)
    #data = data.drop(data.shape[0] -1, axis=0)


    #tech = csv.reader(open('dataset.csv', 'rb'), delimiter=",", quotechar='|')
    #index_col=False
    
    
    tech = pd.read_csv("GOGL_ds.csv", sep=',',header = 0)
    tech.iloc[::-1]
    tech = tech.loc[(tech['date'] >= '2012-02-22')  & (tech['date'] <= date_today)]
    tech = tech.iloc[49:]
    
    #tech.reindex(index=tech.index[::-1])

    #print(tech.to_string())

    #tech = pd.DataFrame(tech)
    #columns = list(tech)

    data = data.values
    print(data)
    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(data)

    # using the last {history_points} open close high low volume data points, predict the next open value
    ohlcv_histories_normalised = np.array([data_normalised[i:i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    print(ohlcv_histories_normalised)
    next_day_open_values_normalised = np.array([data_normalised[:, 0][i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

    next_day_open_values = np.array([data[:, 0][i + history_points].copy() for i in range(len(data) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(next_day_open_values)

    def calc_ema(values, time_period):
        # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
        sma = np.mean(values[:, 3])
        ema_values = [sma]
        k = 2 / (1 + time_period)
        for i in range(len(his) - time_period, len(his)):
            close = his[i][3]
            ema_values.append(close * k + ema_values[-1] * (1 - k))
        return ema_values[-1]
    technical_indicators = []
    technical_indicatorss = []
    #for his in ohlcv_histories_normalised:
    wma= np.array(tech["WMA_14"])
    rsi= np.array(tech["RSI_14"])
    cci= np.array(tech["CCI_10_0.015"])
    sign= np.array(tech["SIGNAL"])
    sma10= np.array(tech["SMA_10"])
    mom= np.array(tech["MOM"])
    stock= np.array(tech["STOCHk_10_10_10"])
    stocd= np.array(tech["STOCHd_10_10_10"])
    ad= np.array(tech["AD"])
    #print(wma)
    #technical_indicators.append([wma, lwr])
    #print(technical_indicators)
    for his in ohlcv_histories_normalised:
        #print(i)
        # note since we are using his[3] we are taking the SMA of the closing price
        sma = np.mean(his[:, 3])
        #print(sma)
        macd = calc_ema(his, 12) - calc_ema(his, 26)
        #technical_indicators.append(np.array([sma]))
        technical_indicatorss.append(np.array([sma,macd,]))
        #print(technical_indicators)
    technical_indicators = np.c_[ sma10 ]
    technical_indicatorss = np.array(technical_indicatorss)
    #technical_indicators = np.array(wma)
    #np.c_[ wma , lwr ]
    print(technical_indicatorss)
    print(technical_indicators)
    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)
    print(technical_indicators_normalised)
    print(ohlcv_histories_normalised.shape[0])
    print(next_day_open_values_normalised.shape[0])
    print(technical_indicators_normalised.shape[0])
    assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0] == technical_indicators_normalised.shape[0]
    return ohlcv_histories_normalised, technical_indicators_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser


def multiple_csv_to_dataset(test_set_name):
    import os
    ohlcv_histories = 0
    technical_indicators = 0
    next_day_open_values = 0
    for csv_file_path in list(filter(lambda x: x.endswith('daily.csv'), os.listdir('./'))):
        if not csv_file_path == test_set_name:
            print(csv_file_path)
            if type(ohlcv_histories) == int:
                ohlcv_histories, technical_indicators, next_day_open_values, _, _ = csv_to_dataset(csv_file_path)
            else:
                a, b, c, _, _ = csv_to_dataset(csv_file_path)
                ohlcv_histories = np.concatenate((ohlcv_histories, a), 0)
                technical_indicators = np.concatenate((technical_indicators, b), 0)
                next_day_open_values = np.concatenate((next_day_open_values, c), 0)

    ohlcv_train = ohlcv_histories
    tech_ind_train = technical_indicators
    y_train = next_day_open_values

    ohlcv_test, tech_ind_test, y_test, unscaled_y_test, y_normaliser = csv_to_dataset(test_set_name)

    return ohlcv_train, tech_ind_train, y_train, ohlcv_test, tech_ind_test, y_test, unscaled_y_test, y_normaliser
