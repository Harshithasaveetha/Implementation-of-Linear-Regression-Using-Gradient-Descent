# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:HARSHITHA D 
RegisterNumber:212224040110


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
print(data.head())
print("\n")
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print("\n")
print(X1_Scaled)
print("\n")
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
  
*/

```

## Output:

<img width="275" height="884" alt="494687591-d4bfd6b9-14e4-4ac6-a1c5-b7d01a658609" src="https://github.com/user-attachments/assets/7d054100-33e2-4b65-8482-fb3c7a647ef0" />\n

<img width="425" height="883" alt="494687690-e4d48c04-5374-4d1f-9c2c-b4eb4c7cf843" src="https://github.com/user-attachments/assets/054bf8d1-4639-478b-b908-00bcb0f008c0" />\n

<img width="299" height="50" alt="494687872-5d35eb0f-4a7b-4d7a-8547-9156d5466683" src="https://github.com/user-attachments/assets/609a2ba0-cc3e-4381-b709-93d8f532174f" />





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
