import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import datetime
import calendar
from xgboost import XGBRegressor as xgb
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score,GridSearchCV,train_test_split

import random


#EDA

data = pd.read_csv(r"C:\Users\arvin\Desktop\DOcs\flightrate\Data_Train.csv")

#Airline opertaional counts
count_airlines = pd.DataFrame(data['Airline'].value_counts())
count_airlines = count_airlines.reset_index(drop=False)
count_airlines.columns = ['Airline','Count']
plt.bar(count_airlines['Airline'],count_airlines['Count'])
plt.title('Airline carriers Vs Counts')
plt.xticks(rotation=90)
plt.show()

data.Price.describe()


#Average price for each airline
airline_price = pd.DataFrame(data.groupby(['Airline']).mean())
plt.bar(airline_price.index,airline_price.Price)
plt.xticks(rotation=90)

#Boxplot for each airline vs Price
data.boxplot(by='Airline',column='Price',vert=False,figsize=(10,10))

#Checking for the outliers
q1 = data['Price'].quantile(0.25)
q3 = data['Price'].quantile(0.75)

iqr = q3 - q1

lower_bound = q1 -(1.5 * iqr) 
upper_bound = q3 +(1.5 * iqr)

outliers = data[data['Price']>upper_bound]

outliers.boxplot(by='Airline',column='Price',vert=False)
outliers.Airline.value_counts()
data.Airline.value_counts()


## Source and destination Vs Price boxplot
data.boxplot(by='Source',column='Price',vert=False)
data.boxplot(by='Destination',column='Price',vert=False)

#Combining Airline, Source and destination Vs Price
data.boxplot(by=['Source','Destination'],column='Price',vert=False)
#Outliers Price comaparison
outliers.boxplot(by=['Airline','Source','Destination'],column='Price',vert=False)




#Feature Egineering

data = pd.read_csv(r"C:\Users\arvin\Desktop\DOcs\flightrate\Data_Train.csv")
#Wrong record
data = data.drop([6474])

data1 = data
encoder = preprocessing.LabelEncoder()
one_hotencoding = preprocessing.OneHotEncoder()


#Splitting duration, arrival and dept time column to hours and mins
data['Duration'] = data['Duration'].map(lambda x: x.split(' '))
data['Dept_hr'] = data['Dep_Time'].map(lambda x: x.split(':')).str[0].astype('int')
data['Dept_min'] = data['Dep_Time'].map(lambda x: x.split(':')).str[1].astype('int')
data['Arr_hr'] = data['Arrival_Time'].map(lambda x: x.split(' ')).str[0].map(lambda x: x.split(':')).str[0].astype('int')
data['Arr_min'] = data['Arrival_Time'].map(lambda x: x.split(' ')).str[0].map(lambda x: x.split(':')).str[1].astype('int')

#Extracting day of month and month from Dept date
data['dayofmonth'] = data['Date_of_Journey'].map(lambda x : str(x).split('/')).str[0].astype(int)
data['month'] = data['Date_of_Journey'].map(lambda x : str(x).split('/')).str[1].astype(int)

#Extracting routes taken in between and filling Nan with None
data['route1'] = data['Route'].map(lambda x: str(x).split(' ? ')).str[0]
data['route2'] = data['Route'].map(lambda x: str(x).split(' ? ')).str[1]
data['route3'] = data['Route'].map(lambda x: str(x).split(' ? ')).str[2]
data['route4'] = data['Route'].map(lambda x: str(x).split(' ? ')).str[3]
data['route5'] = data['Route'].map(lambda x: str(x).split(' ? ')).str[4]

data['route1'] = data['route1'].fillna("None")
data['route2'] = data['route2'].fillna("None")
data['route3'] = data['route3'].fillna("None")
data['route4'] = data['route4'].fillna("None")
data['route5'] = data['route5'].fillna("None")

#Extract only integer value from duration
temp_hr =[]
temp_min =[]
for i in range(0,len(data)):
    try:
        temp = data['Duration'][i][0]
        temp = str(temp).split('h')[0]
        temp_hr.append(temp) 
    except:
        temp_hr.append('0')
for i in range(0,len(data)):
   try:
        temp = data['Duration'][i][1]
        temp = str(temp).split('m')[0]
        temp_min.append(temp)
   except:
        temp_min.append('0')
        
data['temp_hr'] = temp_hr
data['temp_min'] = temp_min

data['temp_hr'] = data['temp_hr'].astype('int')
data['temp_min'] = data['temp_min'].astype('int')

#Weekday from date
data['departure'] = data['Date_of_Journey'].map(str)+' '+data['Dep_Time']
data['departure'] = data['departure'].map(lambda x: datetime.datetime.strptime(x, "%d/%m/%Y %H:%M"))
data['weekday'] = data.departure.dt.dayofweek
data['weekday'] = data['weekday'].map(lambda x:calendar.day_name[x])


#Price Vs weekday
data.boxplot(by='weekday',column='Price',vert=False)
plt.show()
#Airlines Count vs Weekday
weekdays = data.groupby('weekday').count()
plt.bar(weekdays.index,weekdays['Airline'])
plt.show()

#Creating dummy variables from following columns
data_new = data[['Airline','Source','Destination','Additional_Info','Total_Stops']]

data_source =pd.get_dummies(data['Source'],drop_first=True)
data_source.columns = ['source_'+ i for i in data_source.columns]
data_Destination =pd.get_dummies(data['Destination'],drop_first=True)
data_Destination.columns = ['source_'+ i for i in data_Destination.columns]
data_Additional_Info =pd.get_dummies(data['Additional_Info'])
data_Additional_Info = data_Additional_Info.drop(['No Info','No info'],axis=1)
data_Total_Stops =pd.get_dummies(data['Total_Stops'],drop_first=True)
data_Airline =pd.get_dummies(data['Airline'],drop_first=True)
data_weekdays = pd.get_dummies(data['weekday'],drop_first=True)
data_route1 = pd.get_dummies(data['route1'],drop_first=True)
data_route2 = pd.get_dummies(data['route2'],drop_first=True)
data_route3 = pd.get_dummies(data['route3'],drop_first=True)
data_route4 = pd.get_dummies(data['route4'],drop_first=True)
data_route5 = pd.get_dummies(data['route5'],drop_first=True)


#Merging all the columns
final_data = data[['Price','temp_hr','temp_min','Arr_hr','Arr_min','Dept_hr', 'Dept_min','dayofmonth', 'month']]

final_data = pd.concat([final_data,data_Airline,data_source,data_Destination,data_weekdays,
                        data_Additional_Info,data_Total_Stops,data_route1,data_route2,data_route3
                        ,data_route4,data_route5],axis=1)


#Scaling the data before splitiing to train and test
scaler = StandardScaler()
X = final_data.drop('Price',axis=1)
X_scaled  = pd.DataFrame(scaler.fit_transform(X))
Y = final_data[['Price']]


r2_train = pd.DataFrame()
rmse_train = pd.DataFrame()
r2_test = pd.DataFrame()
rmse_test = pd.DataFrame()
shuffle = []
for x in range(20):
   shuffle.append( random.randint(1,101))
for i in shuffle:
    X_train,X_test,Y_train,Y_test = train_test_split(X_scaled,Y,test_size=0.2, random_state=i)
    
    
    clf = xgb()
    xgbmodel = clf.fit(X_train,Y_train)
    train_pred = xgbmodel.predict(X_train)
    print("Train")
    temp = float(r2_score(Y_train,train_pred)*100)
    r2_train = r2_train.append([temp])
    temp = np.sqrt(mean_squared_error(Y_train,train_pred))
    rmse_train = rmse_train.append([temp])
    y_pred = xgbmodel.predict(X_test)
    print("test")
    temp = r2_score(Y_test,y_pred)*100
    r2_test = r2_test.append([temp])
    temp = np.sqrt(mean_squared_error(Y_test,y_pred))
    rmse_test = rmse_test.append([temp])
    

metrics = pd.concat([r2_train,r2_test,rmse_train,rmse_test],axis=1)
metrics.columns = ['R2_Train','R2_Test','RMSE_Train','RMSE_Test']


#Grid search
params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],
'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [2,3,4]}

# Initialize XGB and GridSearch
xgb_model = xgb(nthread=-1) 

grid = GridSearchCV(xgb_model, params)
grid.fit(X_train, Y_train)

print("Train")
print("Accuracy = ",r2_score(Y_train, grid.best_estimator_.predict(X_train)))
print("RMSE = ",np.sqrt(mean_squared_error(Y_train,grid.best_estimator_.predict(X_train)))) 
print("Test")
print("Accuracy = ",r2_score(Y_test, grid.best_estimator_.predict(X_test))) 
print("RMSE = ",np.sqrt(mean_squared_error(Y_test,grid.best_estimator_.predict(X_test))))


#Best parameters
params = {'colsample_bytree': 1.0,
 'gamma': 0.3,
 'max_depth': 4,
 'min_child_weight': 4,
 'subsample': 0.7}

#Final model
xgb = xgb(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bytree=1.0, gamma=0.3, learning_rate=0.1,
             max_delta_step=0, max_depth=4, min_child_weight=4, missing=None,
             n_estimators=100, n_jobs=1, nthread=-1, objective='reg:linear',
             random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
             seed=None, silent=True, subsample=0.7)

