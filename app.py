# importing necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

# X and y for storing features and target values
X = iris.data
y = iris.target

# training the decision tree model
decision_tree = DecisionTreeClassifier()
decision_tree = decision_tree.fit(X,y)

# Displaying title
st.title("Classification using Decision Tree ")

st.write("The classification task is implemented on the iris dataset using Decision Tree Classifier from scikit learn library")

st.subheader("Input Features")

# Sliders for inserting values for prediction
sepal_length = st.slider("Sepal Length",min_value=float(X[:,:1].min()),max_value=float(X[:,:1].max()))
Sepal_width = st.slider("Sepal Width",min_value=float(X[:,1:2].min()),max_value=float(X[:,1:2].max()))
petal_length = st.slider("Petal Length",min_value=float(X[:,2:3].min()),max_value=float(X[:,2:3].max()))
petal_width = st.slider("Petal Width",min_value=float(X[:,3:4].min()),max_value=float(X[:,3:4].max()))

input_data = [[sepal_length,Sepal_width,petal_length,petal_width]]

predicted_class = decision_tree.predict(input_data)

class_map = {0:'Setosa',1:'Versicolor',2:'Virginica'}

# displaying prediction
st.subheader(f"Predicted Class : {class_map[predicted_class[0]]}")