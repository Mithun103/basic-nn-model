# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain).

These input/output units are interconnected and each connection has a weight associated with it. 

Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables.

Regression models work well only when the regression equation is a good fit for the data.

Most regression models will not fit the data perfectly

Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 2 hidden layer(1st hidden layer contains 7 neurons and 2nd hidden layer contains 14 neurons) with activation layer as relu and with their nodes in them.

Now we will fit our dataset and then predict the value.

## Neural Network Model

![Screenshot 2024-02-26 081404](https://github.com/Pavithraramasaamy/basic-nn-model/assets/118596964/226899ce-f9b5-4d2f-bbd2-51b51edf1c40)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:MITHUN MS
### Register Number:212222240067
```python
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('deepdata').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])

df = pd.DataFrame(rows[1:], columns=rows[0])
df

df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})
df.head()

x=df[['Input']].values
y=df[['Output']].values

x
y

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size = 0.4, random_state =35)


Scaler = MinMaxScaler()
Scaler.fit(x_train)
X_train1 = Scaler.transform(x_train)

#Create the model
ai_brain = Sequential([
    Dense(13,activation='relu'),
    Dense(10,activation='relu'),
    Dense(1)
])

#Compile the model
ai_brain.compile(optimizer = 'rmsprop' , loss = 'mse')

ai_brain.fit(X_train1 , y_train,epochs = 3000)

loss_df = pd.DataFrame(ai_brain.history.history)

loss_df.plot()

X_test1 =Scaler.transform(x_test)
ai_brain.evaluate(X_test1,y_test)

X_n1=[[11]]
X_n1_1=Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)

```
## Dataset Information
![image](https://github.com/Mithun103/basic-nn-model/assets/118344695/be477158-8924-46ed-866f-b951492bae4d)


## OUTPUT:

### Training Loss Vs Iteration Plot

![image](https://github.com/Mithun103/basic-nn-model/assets/118344695/2d93ab21-2cae-4f21-bfa0-3c8e0ae7e036)

### Test Data Root Mean Squared Error


![image](https://github.com/Mithun103/basic-nn-model/assets/118344695/39576f2e-6eb4-40b7-919e-f19d4b772f12)


![image](https://github.com/Mithun103/basic-nn-model/assets/118344695/bc6b7a70-9363-4442-8f90-1fbb16fe8a5e)


### New Sample Data Prediction

![image](https://github.com/Mithun103/basic-nn-model/assets/118344695/ade2b04d-b9c3-43dd-bfe8-784c17b6ba92)


## RESULT

Thus a Neural Network regression model for the given dataset is written and executed successfully
