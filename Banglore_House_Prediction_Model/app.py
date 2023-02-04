import streamlit as st
import pandas as pd
import numpy as np
import pickle

pickle_in = open("Banglore_House_Prediction_Model/bengaluru_home_prices_model.pickle","rb")
classifier = pickle.load(pickle_in)

def Welcome():
    return "Welcome All!"

def predict_price(location,total_sqft,bath,bhk):
    x = np.zeros(243)
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    return np.round(classifier.predict([x])[0],3)

def main():
    home = pd.read_csv("Banglore_House_Prediction_Model/bengaluru_house_prices.csv")
    loc = home["location"].unique()
    st.title("Bengaluru House Price Prediction")
    html_temp =  """
    <h2 style = "color:black;text-align:left;"> House Price Prediction </h2>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.subheader("Please enter the required details:")
    location = st.selectbox("Location",loc)
    total_sqft = st.text_input("Square-foot Area","")
    bath = st.text_input("Number of Bathroom", "")
    bhk = st.text_input("Number of BHK", "")

    result = ""
    if st.button("House Price in INR - Lakhs"):
        result = predict_price(location,total_sqft,bath,bhk)
    st.success(" The Final Price is INR : {}/- Lakhs ".format(result))


if __name__ == "__main__":
    main()
