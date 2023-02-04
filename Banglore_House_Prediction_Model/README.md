## Bengaluru House Price Forecasting

## 1.Source
-  We will first build a model using sklearn and linear regression using banglore home prices dataset from kaggle.com.
-  Second step would be to write a streamlit app that uses the saved model to serve https requests and allows user to enter house location square ft area, bedrooms, bhk etc and it will use streamlit to retrieve the predicted price.
-  During model building we will cover almost all data science concepts such as data load and cleaning, outlier detection and removal, feature engineering, dimensionality reduction, gridsearchcv for hyperparameter tunning, k fold cross validation etc.
-  Technology and tools wise this project covers,
-    1.Python
-    2.Numpy and Pandas for data cleaning
-    3.Matplotlib for data visualization
-    4.Sklearn for model building
-    5.Jupyter notebook, visual studio code and pycharm as IDE
-    6.Python Streamlit an open source app framework for our UI and https requests
-    7.Git and Github for our source code version control

## 2.Features
- Simple responsive UI
- Select dataset for prediction [House Price in INR-Lakhs]
- Locations [Select from 300+Locations in Begaluru]
- Square-foot Area [Select the Square_foot in numbers]
- Number of Bathrooms [Select the Bathrooms in numbers]
- Number of BHK [Select the BHK in numbers]
- After providing all these preferences we will get the best predicted price from ML-Model in INR-Lakhs

## 3.Usage
- Clone my repository here ===> (https://github.com/balusena/balugithub/tree/master/Banglore_House_Prediction_Model)
- Open CMD in working directory.
- Run following command.
- pip install -r requirements.txt
- 'app.py' is the main Python file of Streamlit Web-Application. 
- To run app, write following command in CMD. or use any IDE.
- streamlit run app.py
- run the application app.py by logging into your streamlit account it works as intended when you follow the instruction stated in this document. 

## 4.Screenshots
## Bengaluru_House_Price_Forecast_App
<img src="https://github.com/balusena/balugithub/tree/master/Banglore_House_Prediction_Model/app_images/Bengaluru_House_Price_Forecast_App.JPG">

## 5.Heroku deployment
- The app is deployed on Heroku at ===> https://bengaluru-house-price-forecast.herokuapp.com/

## Follow these steps to deploy your streamlit app successfully into Heroku Cloud Platform.
-- Deploy using Heroku Git
-  Use git in the command line or a GUI tool to deploy this app.

-  Install the Heroku CLI
-  Download and install the Heroku CLI.

-  If you haven't already, log in to your Heroku account and follow the prompts to create a new SSH public key.

-- $ heroku login
-  Clone the repository
-  Use Git to clone bengaluru-house-price-forecast's source code to your local machine.

-- $ heroku git:clone -a bengaluru-house-price-forecast 
-- $ cd bengaluru-house-price-forecast

-- Deploy your changes
-  Make some changes to the code you just cloned and deploy them to Heroku using Git.

-  Add the changes
-- $ git add .

-  Commit the changes
-- $ git commit -am "bengaluru-house-price-forecast_commit"

-  Push the chnages
-- $ git push heroku master


## 6.Further work to be done
-1.The current implementation would work well for all the locations in the dataset selection, further iam planning to add the new locations data into the dataset.
-2.Need to implement this project on different important places and locations that are important in real_estate_trend_forecasting and compare results.
-3.Aiming to work with larger and real-world-real_estate-dataset and build an end to end house_price_trend_forecasting application.
-4.Find a way to forecast the house prices of different locations in Bengaluru Silicon Valley with implementation (I would be grateful if someone could help me on this project enhancement.)


