import pandas as pd

data = {
    "Hours_Studied": [1,2,3,4,5,6,7,8,9],
    "Marks": [35,40,50,60,65,70,75,85,90]
}

df = pd.DataFrame(data)
df
X = df[['Hours_Studied']]  # input
y = df['Marks']            # output
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred
import matplotlib.pyplot as plt

plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Student Score Prediction using Linear Regression")
plt.show()
hours = [[7]]
predicted_marks = model.predict(hours)
predicted_marksprint("Predicted marks for 7 hours:", predicted_marks)