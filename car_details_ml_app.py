import pandas as pd
import numpy as np
import streamlit as st
from sklearn import *
import pickle
df = pickle.load(open('data.pkl','rb'))
pipe_lr = pickle.load(open('lrmodel.pkl','rb'))
st.title('Car Dekho by ROMY BABA')
st.header('Fill the details for the price')

Car_Model = st.selectbox('Car Model',df['name'].unique())
Year = st.selectbox('Year',df['year'].unique())
Km_driven = st.selectbox('Total Driven',df['km_driven'].unique())
fuel = st.selectbox('Fuel Type',df['fuel'].unique())
seller_type = st.selectbox('Seller Type',df['seller_type'].unique())
transmission = st.selectbox('Transmission',df['transmission'].unique())
owner = st.selectbox('Owner Type',df['owner'].unique())

if st.button("Car Price"):
    test_data = np.array([Car_Model,Year,Km_driven,fuel,seller_type,transmission,owner])
    test_data = test_data.reshape([1,7])
    st.success(pipe_lr.predict(test_data)[0])

