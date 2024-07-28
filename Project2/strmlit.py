import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
# Load the pre-trained model
model = joblib.load('model.pkl')
df=pd.read_csv("C:\\Users\\saipr\\Downloads\\SaYoPillow.csv")
#splitting the data into training and testing sets
df.rename(columns={'sr.1': 'sr_1'}, inplace=True)

X=df.drop('sl',axis=1)
y=df['sl']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X_train,y_train)

# Streamlit app
st.title("Model Prediction")

# User inputs
st.sidebar.header("Input Features")

def user_input_features():
    sr= st.sidebar.slider("sr", min_value=0.0, max_value=100.0, value=1.0)
    rr = st.sidebar.slider("rr", min_value=0.0, max_value=100.0, value=4.0)
    t = st.sidebar.slider("t", min_value=0.0, max_value=100.0, value=7.0)
    lm = st.sidebar.slider("lm", min_value=0.0, max_value=100.0, value=13.0)
    bo = st.sidebar.slider("bo", min_value=0.0, max_value=100.0, value=13.0)
    rem = st.sidebar.slider("rem", min_value=0.0, max_value=100.0, value=16.0)
    sr_1=st.sidebar.slider("sr_1",min_value=0.0,max_value=100.0,value=1.55)
    hr = st.sidebar.slider("hr", min_value=0.0, max_value=100.0, value=22.0)
    return pd.DataFrame({
        'sr': [sr],
        'rr': [rr],
        't': [t],
        'lm': [lm],
        'bo': [bo],
        'rem': [rem],
        'sr_1':[sr_1],
        'hr': [hr]
    })


new_data = user_input_features()

# Standardize the new data point
new_data_scaled = scaler.transform(new_data)

# Make predictions
prediction = model.predict(new_data_scaled)

# Display results
st.subheader("Prediction")
st.write("Predicted Class:", prediction[0])
