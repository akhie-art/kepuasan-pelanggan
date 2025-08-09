import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🔎 Analisis dan Prediksi Kepuasan Pelanggan dengan KNN")

# 1. Upload data
uploaded_file = st.file_uploader("Upload file CSV dataset customer_feedback", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset berhasil diupload!")
    
    # 2. EDA
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.subheader("Preview data")
    st.dataframe(df.head())
    
    st.subheader("Statistik deskriptif")
    st.write(df.describe())
    
    st.subheader("Distribusi Age")
    fig, ax = plt.subplots()
    sns.histplot(df['Age'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Distribusi SatisfactionScore")
    fig, ax = plt.subplots()
    sns.histplot(df['SatisfactionScore'], bins=20, kde=True, ax=ax, color='green')
    st.pyplot(fig)
    
    st.subheader("Korelasi antar fitur numerik")
    numerical_cols = df.select_dtypes(include='number').columns
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    # 3. Preprocessing
    st.header("⚙️ Preprocessing Data")
    categorical_cols = ['Gender', 'Country', 'FeedbackScore', 'LoyaltyLevel']
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    st.write("Data setelah encoding:")
    st.dataframe(df.head())
    
    # Definisikan fitur & target
    X = df.drop(['CustomerID', 'SatisfactionScore'], axis=1)
    y = (df['SatisfactionScore'] >= 75).astype(int)
    
    # 4. Split data
    st.header("🔀 Split Data Train-Test")
    test_size = st.slider("Proporsi data test (%)", 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    st.write(f"Jumlah data training: {len(X_train)}")
    st.write(f"Jumlah data testing: {len(X_test)}")
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Modeling
    st.header("🤖 Training Model KNN")
    n_neighbors = st.slider("Pilih jumlah tetangga (K)", 1, 20, 5)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaled, y_train)
    st.success("Model berhasil dilatih!")
    
    # 6. Evaluasi
    st.header("📈 Evaluasi Model")
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Akurasi: {acc:.4f}")
    
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    # Prediksi manual input
    st.header("🧮 Prediksi Manual")
    age = st.number_input("Usia", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Jenis Kelamin", encoders['Gender'].classes_)
    country = st.selectbox("Negara", encoders['Country'].classes_)
    income = st.number_input("Pendapatan", min_value=0, value=50000)
    product_quality = st.slider("Kualitas Produk", 0, 10, 5)
    service_quality = st.slider("Kualitas Layanan", 0, 10, 5)
    purchase_frequency = st.slider("Frekuensi Pembelian", 0, 10, 5)
    feedback_score = st.selectbox("Feedback Score", encoders['FeedbackScore'].classes_)
    loyalty_level = st.selectbox("Loyalty Level", encoders['LoyaltyLevel'].classes_)
    
    if st.button("Prediksi Kepuasan"):
        input_arr = np.array([[
            age,
            encoders['Gender'].transform([gender])[0],
            encoders['Country'].transform([country])[0],
            income,
            product_quality,
            service_quality,
            purchase_frequency,
            encoders['FeedbackScore'].transform([feedback_score])[0],
            encoders['LoyaltyLevel'].transform([loyalty_level])[0]
        ]])
        input_scaled = scaler.transform(input_arr)
        pred = knn.predict(input_scaled)[0]
        if pred == 1:
            st.success("✅ Pelanggan PUAS")
        else:
            st.error("⚠️ Pelanggan TIDAK PUAS")

else:
    st.info("Silakan upload file CSV dataset terlebih dahulu.")
