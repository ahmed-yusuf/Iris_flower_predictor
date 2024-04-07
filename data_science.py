import streamlit as st
import pandas as pd
import seaborn as sns
iris=sns.load_dataset('iris')
print(iris.head())
print(iris.isnull().sum())

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
X=iris.drop('species',axis=1)
y=iris['species']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=103)
dt.fit(X_train,y_train)

with st.form('My form'):
    st.title('Iris Flower Predictions')
    st.subheader('Enter the details of your Flower')
    sepal_length=st.number_input('Enter the Sepal length',min_value=0.0,max_value=20.0)
    sepal_width=st.number_input('Enter the Sepal width',min_value=0.0,max_value=20.0)
    petal_length=st.number_input('Enter the Petal length',min_value=0.0,max_value=20.0)
    petal_width=st.number_input('Enter the Petal width',min_value=0.0,max_value=20.0)
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    pred = dt.predict(features)
    final=st.form_submit_button('click to predict flower type')


if final:
    st.success(f'it is a {pred[0]} flower')

