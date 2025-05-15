# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

st.title("🧠 Prédiction d'AVC - Application Machine Learning")

# 📂 Chargement du dataset
@st.cache_data
def load_data():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    df.drop(columns=['id'], inplace=True)
    df.dropna(inplace=True)
    return df

df = load_data()

# 🎯 Prétraitement
df_encoded = df.copy()
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df_encoded.drop("stroke", axis=1)
y = df_encoded["stroke"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 🧠 Entraînement modèle
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
model_cols = X.columns

# 🎛️ Choix de l'action
option = st.radio("Choisissez une action :", ["🔮 Prédiction personnalisée", "📊 Visualisation des performances"])

# 🔮 PRÉDICTION
if option == "🔮 Prédiction personnalisée":
    st.subheader("Saisie des données du patient")

    sample = []
    for col in model_cols:
        if col in label_encoders:
            options = label_encoders[col].classes_
            val = st.selectbox(col, options)
            encoded = label_encoders[col].transform([val])[0]
            sample.append(encoded)
        else:
            val = st.number_input(col, value=float(df[col].mean()))
            sample.append(val)

    if st.button("Prédire"):
        sample_scaled = scaler.transform(np.array(sample).reshape(1, -1))
        pred = model.predict(sample_scaled)[0]
        proba = model.predict_proba(sample_scaled)[0][pred]
        st.success(f"Résultat : {'AVC' if pred==1 else 'Pas d’AVC'} (Confiance : {proba:.1%})")

# 📊 VISUALISATION
elif option == "📊 Visualisation des performances":
    st.subheader("📈 Rapport de classification")
    y_pred = model.predict(X_test)
    st.text(classification_report(y_test, y_pred))

    st.subheader("📌 Matrice de confusion")
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    st.pyplot(fig)