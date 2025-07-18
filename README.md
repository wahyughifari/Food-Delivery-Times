# Food Delivery Time Prediction

Food delivery time is critical for customer satisfaction in food delivery services. This project develops a machine learning model to **predict delivery time (in minutes)** based on factors like distance, preparation time, courier experience, traffic, and weather conditions.

## Problem Statement

The goal of this project is to help food delivery platforms **accurately estimate delivery times** for better logistics planning, improved customer expectations, and optimized courier assignments.

## Objectives

- Perform **Exploratory Data Analysis (EDA)** to understand key factors influencing delivery time.
- Train and evaluate a **regression model** to predict delivery duration.
- Deploy the best model in a user-friendly web application using **Streamlit**.

## Dataset Overview

The dataset contains historical food delivery records with the following columns:

| Column Name              | Description                                     |
|--------------------------|-------------------------------------------------|
| `Distance_km`            | Distance from store to customer (km)            |
| `Preparation_Time_min`  | Order preparation time (minutes)                |
| `Courier_Experience_yrs`| Courier's experience in years                   |
| `Weather`               | Weather condition (Clear, Rainy, Foggy, etc.)   |
| `Traffic_Level`         | Traffic situation (Low, Medium, High)           |
| `Time_of_Day`           | Time when delivery started (Morning to Night)   |
| `Vehicle_Type`          | Transport used (Bike, Car, Scooter)             |
| `Delivery_Time_min`     | Target: Total delivery time in minutes          |

## **Model Development**
- Base model: **Linear Regression**
- Alternative models: **Ridge Regression**, **Lasso Regression**, **Random Forest Regression**
- Model evaluation using:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- [Deployment](https://food-delivery-times-predict.streamlit.app/)
