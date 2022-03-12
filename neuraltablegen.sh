#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gst
i="GOGL"
dir="/home/gabriele/stock"
mkdir -p /home/gabriele/stock/LSTM/

function prepro() { #still has a missin value in foist row

python - << END_SCRIPT
from sklearn import preprocessing
import pandas as pd
date = pd.read_csv('/home/gabriele/stock/LSTM/dataset.csv', usecols = [0])
print(date)
stock = pd.read_csv("/home/gabriele/stock/LSTM/dataset.csv", usecols = [1,2,3,4,5,6,7,8,9,10])
scaler = preprocessing.MinMaxScaler() #use of min max normalization
names = stock.columns
d = scaler.fit_transform(stock)
scaled_df = pd.DataFrame(d, columns=names)
scaled_df.head()

#d = preprocessing.normalize(stock, axis=0) #axis 1 to normalize per sample. (row)
#scaled_df = pd.DataFrame(d, columns=names)
#scaled_df.head()
scaled_df = scaled_df.join(date)
df1 = scaled_df.pop('date') # remove column b and store it in df1
scaled_df.insert(0, "date", df1)

scaled_df.to_csv(r'/home/gabriele/stock/LSTM/NORMdataset.csv', index=False)
END_SCRIPT
echo "end of post processing"
}

function joininonecsv {

touch /home/gabriele/stock/LSTM/dataset.csv
wma_path=$(find /home/gabriele/stock/LSTM/ -type f -print | grep -Eo  "wma14_[0-9]*_[0-9]*.csv" | head -n 1)
sma_path=$(find /home/gabriele/stock/LSTM/ -type f -print | grep -Eo  "sma10_[0-9]*_[0-9]*.csv" | head -n 1)
cci_path=$(find /home/gabriele/stock/LSTM/ -type f -print | grep -Eo  "cci_[0-9]*_[0-9]*.csv" | head -n 1)
stoch_path=$(find /home/gabriele/stock/LSTM/ -type f -print | grep -Eo  "stoch_[0-9]*_[0-9]*.csv" | head -n 1)
rsi_path=$(find /home/gabriele/stock/LSTM/ -type f -print | grep -Eo  "rsi_[0-9]*_[0-9]*.csv" | head -n 1)
macd_path=$(find /home/gabriele/stock/LSTM/ -type f -print | grep -Eo  "macd_[0-9]*_[0-9]*.csv" | head -n 1)
ad_path=$(find /home/gabriele/stock/LSTM/ -type f -print | grep -Eo  "ad_[0-9]*_[0-9]*.csv" | head -n 1)


python - << END_SCRIPT
import csv
import pandas as pd
csv1 = pd.read_csv('/home/gabriele/stock/LSTM/${wma_path}', index_col=False)
csv1.head()
csv2 = pd.read_csv('/home/gabriele/stock/LSTM/${sma_path}', index_col=False)
csv2.head()
csv3 = pd.read_csv('/home/gabriele/stock/LSTM/${cci_path}', index_col=False)
csv3.head()
csv4 = pd.read_csv('/home/gabriele/stock/LSTM/${stoch_path}', index_col=False)
csv4.head()
csv5 = pd.read_csv('/home/gabriele/stock/LSTM/${rsi_path}', index_col=False)
csv5.head()
csv6 = pd.read_csv('/home/gabriele/stock/LSTM/${macd_path}', index_col=False)
csv6.head()
csv7 = pd.read_csv('/home/gabriele/stock/LSTM/${ad_path}', index_col=False)
csv7.head()

merged_data = csv1.merge(csv2,on=["date"]).merge(csv5,on=["date"]).merge(csv3,on=["date"]).merge(csv4,on=["date"]).merge(csv6,on=["date"]).merge(csv7,on=["date"])


del merged_data['Unnamed: 0_x']
del merged_data['Unnamed: 0_y']
del merged_data['Adj Close_y']
del merged_data['Adj Close_x']
del merged_data['Unnamed: 0']



merged_data.to_csv(r'/home/gabriele/stock/LSTM/dataset.csv', index=False)
END_SCRIPT


echo "data has been generated correctly"
}

function signal() {
signal="10"
fast="12"
slow="26"
output=$(./terminal.py "/stocks/load "${1}" -s 2012-01-01/ta/macd -f "$fast" -s "$slow" --signal "$signal" --export csv/exit")

path=$(echo "$output" | grep -Eo  "${HOME}/GamestonkTerminal/exports/stocks/technical_analysis/macd_[0-9]*_[0-9]*.csv" | head -n 1)
echo "${path}"
mv "${path}" "${dir}"/LSTM/

macd_path=$(find /home/gabriele/stock/LSTM/ -type f -print | grep -Eo  "macd_[0-9]*_[0-9]*.csv" | head -n 1)

python - << END_SCRIPT

import pandas as pan
df = pan.read_csv('/home/gabriele/stock/LSTM/${macd_path}',index_col=False)
df['SIGNAL'] = df['MACDs_12_26_10'].shift(1)*(1-(2/(10+1))) + df['MACD_12_26_10']*(2/(10+1))
del df['MACDh_12_26_10']
del df['MACDs_12_26_10']
del df['MACD_12_26_10']
df.to_csv(r'/home/gabriele/stock/LSTM/${macd_path}')
END_SCRIPT
}

function lwr() {

wma14_path=$(find /home/gabriele/stock/LSTM/ -type f -print | grep -Eo  "wma14_[0-9]*_[0-9]*.csv" | head -n 1)
echo "$wma14_path"
python - << END_SCRIPT
import pandas as pan
df = pan.read_csv('/home/gabriele/stock/LSTM/${wma14_path}',index_col=False)
df['max']=df['Adj Close'].rolling(window=10).max().shift(1)
df['min']=df['Adj Close'].rolling(window=10).min().shift(1)
df['LWR']= ((( df['max']-df['Adj Close']) / ( df['max']-df['min'] )) *100)
del df['min']
del df['max']
df.to_csv(r'/home/gabriele/stock/LSTM/${wma14_path}')
END_SCRIPT
}


function mom() {

sma10_path=$(find /home/gabriele/stock/LSTM/ -type f -print | grep -Eo  "sma10_[0-9]*_[0-9]*.csv" | head -n 1)

python - << END_SCRIPT
import csv
import pandas as pan
df = pan.read_csv('/home/gabriele/stock/LSTM/${sma10_path}',index_col=False)
df['MOM'] = df['Adj Close'].shift(9) - df['Adj Close']
df.to_csv(r'/home/gabriele/stock/LSTM/${sma10_path}')
END_SCRIPT
}

function sma10(){ 

#Moving Averages are used to smooth the data in an array to help eliminate noise and identify trends. The Simple
#Moving Average is literally the simplest form of a moving average. Each output value is the average of the
#previous n values. In a Simple Moving Average, each value in the time period carries equal weight, and values
#outside of the time period are not included in the average. This makes it less responsive to recent changes in
#the data, which can be useful for filtering out those changes.

window="10"
offset="0"
output=$(./terminal.py "/stocks/load "${1}" -s 2012-01-01/ta/sma -l "$window" -o "$offset" --export csv/exit")

path=$(echo "$output" | grep -Eo  "${HOME}/GamestonkTerminal/exports/stocks/technical_analysis/sma10_[0-9]*_[0-9]*.csv" | head -n 1)
echo "${path}"

mv "${path}" "${dir}"/LSTM/
}

function wma14(){

#A Weighted Moving Average puts more weight on recent data and less on past data. This is done by multiplying
#each barâ€™s price by a weighting factor. Because of its unique calculation, WMA will follow prices more closely
#than a corresponding Simple Moving Average.

window="14"
offset="0"
output=$(./terminal.py "/stocks/load "${1}" -s 2012-01-01/ta/wma -l "$window" -o "$offset" --export csv/exit")

path=$(echo "$output" | grep -Eo  "${HOME}/GamestonkTerminal/exports/stocks/technical_analysis/wma14_[0-9]*_[0-9]*.csv" | head -n 1)
echo "${path}"

mv "${path}" "${dir}"/LSTM/
}

function cci(){ 

#The CCI is designed to detect beginning and ending market trends. The range of 100 to -100 is the normal trading
#range. CCI values outside of this range indicate overbought or oversold conditions. You can also look for price
#divergence in the CCI. If the price is making new highs, and the CCI is not, then a price correction is likely.

len="10"
#scalar="0.015"
output=$(./terminal.py "/stocks/load "${1}" -s 2012-01-01/ta/cci -l "$len" --export csv/exit")

path=$(echo "$output" | grep -Eo  "${HOME}/GamestonkTerminal/exports/stocks/technical_analysis/cci_[0-9]*_[0-9]*.csv" | head -n 1)
echo "${path}"

mv "${path}" "${dir}"/LSTM
}

function rsi(){ 

#The Relative Strength Index (RSI) calculates a ratio of the recent upward price movements to the absolute price movement. The RSI ranges from 0 to 100. The RSI is
#interpreted as an overbought/oversold indicator when the value is over 70/below 30. You can also look for divergence with price. If the price is making new highs/lows,
#and the RSI is not, it indicates a reversal.

len="14"
scalar="100"
drift="1"
output=$(./terminal.py "/stocks/load "${1}" -s 2012-01-01/ta/rsi -l "$len" -s "$scalar" -d "$drift" --export csv/exit")

path=$(echo "$output" | grep -Eo  "${HOME}/GamestonkTerminal/exports/stocks/technical_analysis/rsi_[0-9]*_[0-9]*.csv" | head -n 1)
echo "${path}"

mv "${path}" "${dir}"/LSTM
}



function stoch(){ 


#The Stochastic Oscillator measures where the close is in relation to the recent trading range. The values range from zero to 100. %D values over 75 indicate an
#overbought condition; values under 25 indicate an oversold condition. When the Fast %D crosses above the Slow %D, it is a buy signal; when it crosses below, it is a
#sell signal. The Raw %K is generally considered too erratic to use for crossover signals.

fast="10"
slowd="10"
slowk="10"
output=$(./terminal.py "/stocks/load "${1}" -s 2012-01-01/ta/stoch -k "$fast" -d "$slowd" --slowkperiod "$slowk" --export csv/exit")

path=$(echo "$output" | grep -Eo  "${HOME}/GamestonkTerminal/exports/stocks/technical_analysis/stoch_[0-9]*_[0-9]*.csv" | head -n 1)
echo "${path}"

mv "${path}" "${dir}"/LSTM
}

function ad(){ 
#The Accumulation/Distribution Line is similar to the On Balance Volume (OBV), which sums the
#volume times +1/-1 based on whether the close is higher than the previous close. The
#Accumulation/Distribution indicator, however multiplies the volume by the close location value
#(CLV). The CLV is based on the movement of the issue within a single bar and can be +1, -1 or
#zero. The Accumulation/Distribution Line is interpreted by looking for a divergence in the
#direction of the indicator relative to price. If the Accumulation/Distribution Line is
#trending upward it indicates that the price may follow. Also, if the Accumulation/Distribution
#Line becomes flat while the price is still rising (or falling) then it signals an impending
#flattening of the price.

output=$(./terminal.py "/stocks/load "${1}"  -s 2012-01-01/ta/ad --export csv/exit")
path=$(echo "$output" | grep -Eo  "${HOME}/GamestonkTerminal/exports/stocks/technical_analysis/ad_[0-9]*_[0-9]*.csv" | head -n 1)
echo "${path}"

mv "${path}" "${dir}"/LSTM
}




#sma10 "${i}"
#wma14 "${i}"
#rsi "${i}"
#stoch "${i}"
#signal "${i}"
#lwr
#mom
#ad "${i}"
#cci "${i}"
#joininonecsv
prepro




