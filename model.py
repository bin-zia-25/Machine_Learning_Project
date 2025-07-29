import numpy as np
import pandas as pd
df = pd.read_csv("Housing.csv")

#
yes_no_cols = ['mainroad','guestroom','basement','airconditioning','prefarea']
for col in yes_no_cols:
    df[col] = df[col].map({'yes':1,'no':0})

#
df['furnishingstatus'] = df['furnishingstatus'].map({'furnished':1,'unfurnished':0,'semi-furnished':0.5})
x = df.drop(['price','hotwaterheating'],axis=1)
# print(df['price'])
y = np.log(df['price'])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)

#
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

#
from sklearn.model_selection import cross_val_score
 
Cross_Val_Score = cross_val_score(lr,x_train,y_train, scoring='neg_mean_squared_error',cv=10)
neg_mean_sq = np.mean(Cross_Val_Score)

#
y_predict = lr.predict(x_test)
# print("The value of Y predict is :",np.exp(y_predict))

from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

mse = mean_squared_error(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)


# print("MSE:", mse)
# print("MAE:", mae)
# print("r2:", r2)

# # Custom value to check model performance
# custom_data = np.array([[30000,4,2,3,1,0,0,1,2,1,1.0]])
# std_custom_data = sc.transform(custom_data)
# prediction = lr.predict(std_custom_data)
# print("The price is ",np.exp(prediction[0]))



import joblib as jl
jl.dump(lr,'Linear_Regression_Model.pkl')
jl.dump(sc,'scaler.pkl')


