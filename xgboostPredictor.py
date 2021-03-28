from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv


listNEstimators = []

for i in range(10, 2010, 10):
    listNEstimators.append(i)

MAEList = []
predList = []
BESTimator = [0]
maeBest = 100

fileName = '/Users/aviralmehrotra/Downloads/Code/Python/Machine Learning/Projects/Stocks/Files/AAPL.csv'

for numberEstimators in listNEstimators:
    data = pd.read_csv(fileName)
    r = csv.reader(open(fileName))
    lines = list(r)


    for items in lines:
        replaced = items[0].replace('-', '')
        items[0] = replaced

    writer = csv.writer(open(fileName, 'w'))
    writer.writerows(lines)
    
    data = data[['Date', 'Open', 'Close']]

    x = np.array(data.drop(['Close'], 1)) 
    y = np.array(data['Close'])

    for i in range(len(y)):
        y[i]= float(y[i])
    for items in x:
        items[1]= float(items[1])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

    model = XGBRegressor(n_estimators = numberEstimators, learning_rate = 0.1)
    model.fit(x_train, y_train, early_stopping_rounds = 5, eval_set = [(x_test, y_test)], verbose = False)

    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_pred, y_test)
    
    print(str(numberEstimators) + ' Estimators ' + '= ' + str(mae) + ' MAE')

    if mae < 20:
        MAEList.append(mae)

    if mae < maeBest:
        maeBest = mae
        BESTimator.append(numberEstimators)
        BESTimator.pop(0)

    pred_data = [20210326, 120.39]
    pred_data = np.array(pred_data).reshape((1,-1))
    y_future = model.predict(pred_data)
    print('Future Price Prediction: ' + str(*y_future))

    predList.append(y_future)


with open('predictions.csv', 'w', newline = '') as file:
    writer = csv.writer(file)
    writer.writerow(['Date', 'Pred Close'])
    i = 0
    for items in y_pred: 
        writer.writerow([x_test[i][0], y_pred[i], y_test[i]])
        i += 1


lowestMAE = min(MAEList)
highestMAE = max(MAEList)
avgMAE = sum(MAEList) / len(MAEList)

print()
print('Lowest MAE' + ' = ' + str(lowestMAE))
print('Highest MAE' + ' = ' + str(highestMAE))
print()
print('Average MAE' + ' = ' + str(avgMAE))

print('Best Number of Estimators: ', *BESTimator)

lowestPred = min(predList)
highestPred = max(predList)
avgPred = sum(predList) / len(predList)

print()
print('Lowest Prediction' + ' = ' + str(*lowestPred))
print('Highest Prediction' + ' = ' + str(*highestPred))
print()
print('Average Prediction' + ' = ' + str(*avgPred))


