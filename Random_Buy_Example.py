import random
import requests
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras.backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 取得股票資訊
# Input:
#   stock_code: 股票ID
#   start_date: 開始日期，YYYYMMDD
#   stop_date: 結束日期，YYYYMMDD
# Output: 持有股票陣列


def Get_Stock_Informations(stock_code, start_date, stop_date):
    information_url = ('http://140.116.86.242:8081/stock/' +
                       'api/v1/api_get_stock_info_from_date_json/' +
                       str(stock_code) + '/' +
                       str(start_date) + '/' +
                       str(stop_date)
                       )
    try:
        result = requests.get(information_url).json()
    except:
        return dict([])
    if(result['result'] == 'success'):
        return result['data']
    return dict([])

# 取得持有股票
# Input:
#   account: 使用者帳號
#   password: 使用者密碼
# Output: 持有股票陣列


def Get_User_Stocks(account, password):
    data = {'account': account,
            'password': password
            }
    search_url = 'http://140.116.86.242:8081/stock/api/v1/get_user_stocks'
    result = requests.post(search_url, data=data).json()
    if(result['result'] == 'success'):
        return result['data']
    return dict([])

# 預約購入股票
# Input:
#   account: 使用者帳號
#   password: 使用者密碼
#   stock_code: 股票ID
#   stock_shares: 購入張數
#   stock_price: 購入價格
# Output: 是否成功預約購入(True/False)


def Buy_Stock(account, password, stock_code, stock_shares, stock_price):
    print('Buying stock...')
    data = {'account': account,
            'password': password,
            'stock_code': stock_code,
            'stock_shares': stock_shares,
            'stock_price': stock_price}
    buy_url = 'http://140.116.86.242:8081/stock/api/v1/buy'
    result = requests.post(buy_url, data=data).json()
    print('Result: ' + result['result'] + "\nStatus: " + result['status'])
    return result['result'] == 'success'

# 預約售出股票
# Input:
#   account: 使用者帳號
#   password: 使用者密碼
#   stock_code: 股票ID
#   stock_shares: 售出張數
#   stock_price: 售出價格
# Output: 是否成功預約售出(True/False)


def Sell_Stock(account, password, stock_code, stock_shares, stock_price):
    print('Selling stock...')
    data = {'account': account,
            'password': password,
            'stock_code': stock_code,
            'stock_shares': stock_shares,
            'stock_price': stock_price}
    sell_url = 'http://140.116.86.242:8081/stock/api/v1/sell'
    result = requests.post(sell_url, data=data).json()
    print('Result: ' + result['result'] + "\nStatus: " + result['status'])
    return result['result'] == 'success'


# 隨機購入或出售
# Input: None
# Output: None
def Random_Buy_Or_Sell():

    account = '帳號'  # 使用者帳號
    password = '密碼'  # 使用者密碼

    today = datetime.today().strftime('%Y%m%d')  # 今日日期，YYYYMMDD

    action = random.randrange(0, 10000000) & 1  # 決定操作為隨機購買或售出，0=購買、1=售出
    user_stocks = Get_User_Stocks(account, password)  # 取得使用者持有股票S
    if(len(user_stocks) == 0):  # 若使用者不持有任何股票
        action = 0  # 指定操作為購買股票
    if(action == 0):  # 如果操作為購買股票
        target_stocks_id = [2330]  # 購買股票清單，隨機購買將於該清單內隨機挑選一個股票來進行購買

        selected_stock_id = target_stocks_id[random.randrange(
            0, 100000000) % len(target_stocks_id)]  # 於購買清單隨機挑選一張股票

        today_stock_information = Get_Stock_Informations(
            selected_stock_id, '20200101', today)  # 取得選定股票往日資訊
        if(len(today_stock_information) == 0):  # 若選定股票沒有任何資訊
            print('未曾開市')
            return  # 結束操作
        today_stock_information = today_stock_information[0]  # 取得選定股票最新資訊
        # 股票隨機浮動範圍設定為最高價-最低價
        random_trading_price_offset_range = today_stock_information['high'] - \
            today_stock_information['low']

        buy_shares = random.randrange(1, 10)  # 股票隨機數量選定1~10張

        #購買價格以收盤價格-浮動範圍*(隨機浮點數0.0~1.0)
        buy_price = today_stock_information['close'] - random.random() * random_trading_price_offset_range
        # buy_price = today_stock_information['high']  # 購買價格設定為股票最高價

        Buy_Stock(account, password, selected_stock_id,
                  buy_shares, buy_price)  # 購買股票
    else:  # 操作為售出股票
        selected_stock = user_stocks[random.randrange(
            0, 100000000) % len(user_stocks)]  # 隨機選定一個使用者持有的股票

        selected_stock_id = selected_stock['stock_code_id']  # 取得選定的股票ID

        today_stock_information = Get_Stock_Informations(
            selected_stock_id, '20200101', today)  # 取得選定股票最新資訊
        if(len(today_stock_information) == 0):  # 若選定股票沒有任何資訊
            print('未曾開市')
            return  # 結束操作
        today_stock_information = today_stock_information[0]  # 取得選定股票最新資訊
        # 股票隨機浮動範圍設定為最高價-最低價
        random_trading_price_offset_range = today_stock_information['high'] - \
            today_stock_information['low']

        keeping_shares = selected_stock['shares']  # 取得使用者在選定股票所持有的張數
        sell_shares = random.randrange(
            1, keeping_shares)  # 隨機取得售出張數(1~使用者持有張數)
        sell_price = today_stock_information['close'] + random.random(
        ) * random_trading_price_offset_range  # 購買價格以收盤價格+浮動範圍*(隨機浮點數0.0~1.0)

        Sell_Stock(account, password, selected_stock_id,
                   sell_shares, sell_price)  # 售出股票

#print(Get_Stock_Informations(2325,"20230309","20230311"))
#Random_Buy_Or_Sell()

#write stock info as json formatted string into files.
def scanStocks():
    for i in range(1000, 10000):
        dictionary = Get_Stock_Informations(i, "20160222", "20230312")
        if dictionary != dict([]):
            with open('sample' + str(i) + '.json', 'w') as outfile:
                outfile.write(json.dumps(dictionary, indent=4))

def readIndividualStocksIntoLargeFiles(list):
    dicList = []
    for i in list:#range(1101, 10000):
        try:
            dicList.append(json.loads(open("samples\\sample"+str(i)+".json", 'r').read()))
        except:
            continue
        if len(dicList) == 10:
            break
    print(dicList[0][0]['date'])
    with open('samples//customsamples.json','w') as outfile:
        outfile.write(json.dumps(dicList, indent=4))

def loadStocks(file = 'samples//first10samples.json') -> list[list[dict]]:
    dat = json.loads(open(file,'r').read())
    return dat

def extractFeature(feature : str, nested_list : list) -> list:
    if isinstance(nested_list, dict):
        return nested_list[feature] # replace dict with a single feature
    elif isinstance(nested_list, list):
        return [extractFeature(feature, item) for item in nested_list] # recursive call for each item in the list
    else:
        return nested_list # base case: return non-list and non-dict items unchanged

#276 for tsmc

def evenOutData(data):
    minlength = 99999999999
    for i in range(0, len(data)):
        if len(data[i]) < minlength:
            minlength = len(data[i])
    for i in range(0, len(data)):
        data[i] = data[i][0:minlength]
        data[i].reverse()
    return data


def remove_rows_with_zeros(matrix1, matrix2):
    # Find the rows in matrix1 that contain a 0
    zero_rows = np.any(matrix1 == 0.0, axis=1)
    
    # Remove the zero rows from both matrices
    matrix1 = matrix1[~zero_rows]
    matrix2 = matrix2[~zero_rows]
    
    # Return the modified matrices and the indices of the removed rows
    return matrix1, matrix2, np.where(zero_rows)[0]
# from keras.preprocessing.sequence import TimeseriesGenerator
# closedata = extractFeature('close', loadStocks())
# minlength = 99999999
# for i in range(0, len(closedata)):
#     if len(closedata[i]) < minlength:
#         minlength = len(closedata[i])

# print(minlength)
# for i in range(0, len(closedata)):
#     closedata[i] = closedata[i][0:minlength]
#     closedata[i].reverse()

# array = np.array(closedata).transpose() #make sure dim 0 is time
# #print(array)
# generator = TimeseriesGenerator(array, array, 50, batch_size=1)
# X, y = generator[0]
# print(X)
# print(y)
# input()
# print(array[1702])

# def loss(y_true : tf.Tensor, y_pred : tf.Tensor):
#     print(type(y_true))
#     print(tf.size(y_pred))
#     print(y_true[0])
#     input()
#     return tf.divide(tf.reduce_sum(tf.pow(tf.subtract(tf.nn.softmax(y_true), y_pred),2.0)), 10.0)

# model = Sequential()
# model.add(LSTM(128, activation='relu', input_shape=(50, 10))) #time batch length, 10 stocks
# model.add(Dense(10,activation='softmax'))
# model.compile(optimizer='adam', loss=loss) #loss = fn(y_true, y_pred) where y_true are ground truth vals and y_pred are model predictions. lost can be list of losses for each output! 

# model.summary()
# model.fit(generator,epochs=2)

# testme = array[1652:1702].reshape(1, 50, 10)
# print(model.predict(testme))
# print('actual value: ', tf.nn.softmax(array[1702]))



#readIndividualStocksIntoLargeFiles()