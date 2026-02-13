<img width="882" height="461" alt="Screenshot 2026-02-13 083607" src="https://github.com/user-attachments/assets/984060af-ed26-46c4-8757-2234fb771122" /># Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Aswath Ragavan MG
RegisterNumber:  25013229
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"C:\Users\acer\Downloads\student_scores.csv")

print(df.head(10))

plt.scatter(df['Hours'], df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Hours vs Scores')
plt.show()


X = df[['Hours']]   
y = df['Scores']    

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lr = LinearRegression()
lr.fit(X_train, Y_train)

single_pred = lr.predict(X_test.iloc[0].values.reshape(1, 1))
print("Single Prediction:", single_pred)

plt.scatter(df['Hours'], df['Scores'])
plt.plot(X_train, lr.predict(X_train), color='red')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Regression Line')
plt.show()

print("Coefficient:", lr.coef_)
print("Intercept:", lr.intercept_)

y_pred = lr.predict(X_test)

mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)
```

## Output:
<img width="911" height="624" alt="image" src="https://github.com/user-attachments/assets/a911c0d4-5559-44bc-86e1-1dd9f8a048f7" />
<img width="882" height="461" alt="Screenshot 2026-02-13 083607" src="https://github.com/user-attachments/assets/76f5a5ee-82a2-4fd4-9006-e777a6743ff4" />
<img width="877" height="633" alt="Screenshot 2026-02-13 083632" src="https://github.com/user-attachments/assets/0ab85426-0230-4500-aa6e-448c3a4954d6" />










## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
