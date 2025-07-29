# Libraries
import numpy as np
import pandas as pd
df = pd.read_csv("Housing.csv")

# Converting to numbers (0,1)
yes_no_cols = ['mainroad','guestroom','basement','airconditioning','prefarea']
for col in yes_no_cols:
    df[col] = df[col].map({'yes':1,'no':0})

# same
df['furnishingstatus'] = df['furnishingstatus'].map({'furnished':1,'unfurnished':0,'semi-furnished':0.5})
x = df.drop(['price','hotwaterheating'],axis=1)
# print(df['price'])
y = np.log(df['price'])

# Spliting Test and Train Data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)


# Importing model 
from sklearn.ensemble import RandomForestRegressor
random_forest_model = RandomForestRegressor(n_estimators=80,random_state=40,max_depth=10)
random_forest_model.fit(x_train,y_train)

# Predicting
y_predict = random_forest_model.predict(x_test)

# Finding MSE,MAE,r2
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
mse = mean_squared_error(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)


# print("MSE:", mse)
# print("MAE:", mae)
# print("r2:", r2)


# Testing Model on Random Data.
# custom_data = np.array([[16200,5,3,2,1,0,0,0,0,0,0.0]])
# prediction = random_forest_model.predict(custom_data)
# print("The price is ",np.exp(prediction[0]))




# Saving Model
import joblib as jl
jl.dump(random_forest_model,'Trained_Rf_model.pkl')



# import matplotlib.pyplot as plt
# # Get feature importances from your model
# importances = random_forest_model.feature_importances_

# # Create a DataFrame for better visualization
# feature_importance_df = pd.DataFrame({
#     'Feature': x.columns,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)

# # Print top features
# print(feature_importance_df)

# # Plot feature importances
# plt.figure(figsize=(10,6))
# plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
# plt.gca().invert_yaxis()  # Highest at top
# plt.title('Feature Importance')
# plt.xlabel('Importance')
# plt.tight_layout()
# plt.show()

