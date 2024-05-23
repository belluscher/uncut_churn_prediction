import streamlit as st
import pandas as pd
import pickle as p
from sklearn.datasets import  fetch_california_housing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#import xgboost as xgb
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime, timedelta

st.title("Churn prediction")

st.write("We are looking at the user activity since 2021-12-27")

st.write("---")
st.header("Lets see some data")

uncut_clean=pd.read_csv("uncut_clean.csv")

user_input = st.text_input("Please enter the current date (YYYY-MM-DD): ")
current_date = pd.to_datetime(user_input)
Last_Logged_In_At = pd.to_datetime(uncut_clean['Last Logged In At'])
uncut_clean['is_churned'] = (current_date - Last_Logged_In_At) > timedelta(days=90)

target = uncut_clean["is_churned"]
features = uncut_clean[["Uncut Collectors #", "Followers #", "ArtX Balance", "ArtX Total Earned", "signup_lastlogin"]]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.20, random_state = 0)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
normalizer = RobustScaler()
normalizer.fit(X_train)
X_train_norm = normalizer.transform(X_train)
X_test_norm = normalizer.transform(X_test)
model = AdaBoostRegressor(DecisionTreeRegressor(max_depth= 10,
 max_leaf_nodes = 500))
model.fit(X_train_norm, y_train)

option = st.selectbox("Select type of user", ("All users", "Active in the last 90 days"))
if option== "Active in the last 90 days":
    st.write(uncut_clean[uncut_clean['is_churned'] == False])
    st.write("Number of unique Uncut User ID:",uncut_clean[uncut_clean['is_churned'] == False]["Uncut User ID"].nunique())
else:
    st.write(uncut_clean) 
    st.write("Number of unique Uncut User ID:", uncut_clean['Uncut User ID'].nunique())
st.write("---")
st.markdown("""This app performs simple visualizations of Uncut data until 15 May 2024.""")

st.write("---")

option = st.selectbox("Predict informing", ("Record ID", "All Info"))
if option== "Record ID":
    record_id = int(st.text_input("Enter Record ID: "))
    st.write(uncut_clean[uncut_clean['Record ID'] == record_id]["is_churned"])
    columns = ["Uncut Collectors #", "Followers #", "ArtX Balance", "ArtX Total Earned", "Created At", "Last Logged In At"]
    result = uncut_clean[uncut_clean["Record ID"] == record_id][columns]
    st.write(result)


else:
  
    collectors = st.text_input("How many Collectores the user has: ")
    followers = st.text_input("How many Followers the user has: ")
    balance = st.text_input("How much Balance the user has: ")
    artx_earned = st.text_input("How many ArtX does user has: ")
    Created_At = st.text_input("When was the user's account created? (YYYY-MM-DD): ")
    last_logged_in_at = st.text_input("When was the user's last login? (YYYY-MM-DD): ")
    Created_At_date = pd.to_datetime(Created_At)
    last_logged_in_at_date = pd.to_datetime(last_logged_in_at)
    signup_lastlogin = (last_logged_in_at_date - Created_At_date).days
    data = {"Uncut Collectors #":[float(collectors)],
            "Followers #":[float(followers)],
            "ArtX Balance":[float(balance)],
            "ArtX Total Earned":[float(artx_earned)],
            "signup_lastlogin":[float(signup_lastlogin)]}
    df = pd.DataFrame(data)
    df = normalizer.transform(df)
    



    if st.button("Predict"):
        prediction = model.predict(df)
        if prediction [0]==False:
            st.write("Your client is staying")
        else:
            st.write("Your client is likely leaving, do something about it!")
