import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from scipy.stats import f_oneway

# === Load Trained Components ===
model_path = os.path.join(os.path.dirname(__file__), 'linear_model.pkl')
model = joblib.load(model_path)
base_dir = os.path.dirname(__file__)
scaler = joblib.load(os.path.join(base_dir, 'scaler.pkl'))
columns = joblib.load(os.path.join(base_dir, 'columns.pkl'))
df = pd.read_csv(os.path.join(base_dir, 'dataset.csv'))

# === Sidebar Navigation ===
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Prediction", "EDA"])

# ===============================
# PAGE 1: PREDICTION
# ===============================
if page == "Prediction":
    st.title("Food Delivery Time Prediction")

    # Input Form
    distance = st.number_input("Distance (km)", min_value=0.0, format="%.1f")
    weather = st.selectbox("Weather", ["Clear", "Windy", "Rainy", "Snowy", "Foggy"])
    traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
    time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
    vehicle = st.selectbox("Vehicle Type", ["Bike", "Car", "Scooter"])
    prep_time = st.number_input("Preparation Time (min)", min_value=0)
    experience = st.number_input("Courier Experience (years)", min_value=1, max_value=50, value=1)

    if st.button("Predict Delivery Time"):
        input_df = pd.DataFrame({
            'Distance_km': [distance],
            'Preparation_Time_min': [prep_time],
            'Courier_Experience_yrs': [experience],
            'Weather': [weather],
            'Traffic_Level': [traffic],
            'Time_of_Day': [time_of_day],
            'Vehicle_Type': [vehicle]
        })

        # One-Hot Encoding
        input_encoded = pd.get_dummies(input_df)

        # Pastikan semua kolom sesuai model training
        input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

        # Hanya scaling kolom numerik
        numerical_cols = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']
        input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])

        # Prediksi
        pred_time = model.predict(input_encoded)[0]

        st.success(f"Estimated Delivery Time: **{pred_time:.2f} minutes**")

# ===============================
# PAGE 2: EDA
# ===============================
elif page == "EDA":
    st.title("Exploratory Data Analysis")

    # Overview
    st.markdown("""
    ### Overview
    This page provides insights into the delivery dataset, analyzing key variables such as distance, time of day, traffic, weather, and courier experience.
    Both descriptive and inferential statistics are used to identify patterns and relationships affecting delivery time.
    """)

    # Descriptive Statistics
    st.markdown("### Descriptive Statistics")
    st.dataframe(df.describe())

    # 1. Distance vs Delivery Time
    st.markdown("### 1. Distance vs Delivery Time")
    dis = df['Distance_km'].corr(df['Delivery_Time_min'])
    fig1 = sns.lmplot(x='Distance_km', y='Delivery_Time_min', data=df, height=3, aspect=2.2, line_kws={'color': 'red'})
    plt.title(f"Pearson correlation = {dis:.2f}")
    st.pyplot(fig1.fig)

    # 2. Average Delivery Time by Time of Day
    st.markdown("### 2. Average Delivery Time by Time of Day")
    avg_time = df.groupby("Time_of_Day")["Delivery_Time_min"].mean().sort_values(ascending=False).reset_index()
    fig2, ax2 = plt.subplots(figsize=(7, 2.8))
    sns.barplot(x='Time_of_Day', y='Delivery_Time_min', data=avg_time, palette='Blues', ax=ax2)
    for p in ax2.patches:
        ax2.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width()/2, p.get_height()), ha='center', fontsize=8)
    st.pyplot(fig2)

    # 3. Average Delivery Time by Weather
    st.markdown("### 3. Average Delivery Time by Weather")
    mean_weather = df.groupby("Weather")["Delivery_Time_min"].mean().sort_values(ascending=False).reset_index()
    fig3, ax3 = plt.subplots(figsize=(7, 2.8))
    sns.barplot(x="Weather", y="Delivery_Time_min", data=mean_weather, palette="pastel", ax=ax3)
    for i, row in mean_weather.iterrows():
        ax3.text(i, row["Delivery_Time_min"] + 0.01, f"{row['Delivery_Time_min']:.1f}", ha='center', fontsize=8)
    st.pyplot(fig3)

    # 4. Delivery Time Bin vs Traffic Level
    st.markdown("### 4. Delivery Time Categories by Traffic Level")
    df['Delivery_Time_Bin'] = pd.cut(df['Delivery_Time_min'], bins=[0, 30, 60, 90, 120], labels=['<30', '30-60', '60-90', '>90'])
    fig4, ax4 = plt.subplots(figsize=(7, 2.8))
    sns.countplot(x='Delivery_Time_Bin', hue='Traffic_Level', data=df, palette='Set2', ax=ax4)
    ax4.legend(title="Traffic Level", fontsize=8, title_fontsize=9)
    st.pyplot(fig4)

    # 5. Average Delivery Time by Vehicle Type
    st.markdown("### 5. Average Delivery Time by Vehicle Type")
    avg_vehicle = df.groupby("Vehicle_Type")["Delivery_Time_min"].mean().sort_values(ascending=False).reset_index()
    fig5, ax5 = plt.subplots(figsize=(7, 2.8))
    sns.barplot(x='Vehicle_Type', y='Delivery_Time_min', data=avg_vehicle, palette='Set2', ax=ax5)
    for p in ax5.patches:
        ax5.annotate(f'{p.get_height():.1f}', (p.get_x()+p.get_width()/2, p.get_height()), ha='center', fontsize=8)
    st.pyplot(fig5)

    # 6. Preparation Time vs Delivery Time
    st.markdown("### 6. Preparation Time vs Delivery Time")
    corr = df['Preparation_Time_min'].corr(df['Delivery_Time_min'])
    fig6 = sns.lmplot(x='Preparation_Time_min', y='Delivery_Time_min', data=df, height=3, aspect=2.2, line_kws={'color': 'orange'})
    plt.title(f"Pearson correlation = {corr:.2f}")
    st.pyplot(fig6.fig)

    # 7. Courier Experience vs Delivery Time
    st.markdown("### 7. Courier Experience vs Delivery Time")
    mean_exp = df.groupby('Courier_Experience_yrs')['Delivery_Time_min'].mean().reset_index().sort_values('Courier_Experience_yrs')
    fig7, ax7 = plt.subplots(figsize=(7, 2.8))
    sns.lineplot(x='Courier_Experience_yrs', y='Delivery_Time_min', data=mean_exp, marker='o', color='green', ax=ax7)
    for i, row in mean_exp.iterrows():
        ax7.text(row['Courier_Experience_yrs'], row['Delivery_Time_min'] + 0.04, f"{row['Delivery_Time_min']:.1f}", ha='center', fontsize=8)
    st.pyplot(fig7)

    # Inferential Statistics - ANOVA
    st.markdown("### Inferential Statistics (ANOVA)")
    groups = [group["Delivery_Time_min"].values for name, group in df.groupby("Weather")]
    anova_result = f_oneway(*groups)
    st.markdown(f"**p-value = {anova_result.pvalue:.4f}**")
    if anova_result.pvalue < 0.05:
        st.success("Significant difference in delivery time across weather groups.")
    else:
        st.info("No significant difference across weather groups.")