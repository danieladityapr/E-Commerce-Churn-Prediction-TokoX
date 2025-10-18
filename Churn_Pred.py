import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import streamlit as st
import pickle
import shap
from streamlit_shap import st_shap
import matplotlib.pyplot as plt

# ==============================================================
# Judul Utama
st.markdown(
    "<h1 style='text-align: center;'>Churn Prediction</h1>",
    unsafe_allow_html=True
)

st.write("")

# ==============================================================
# Fungsi input user tanpa sidebar
def user_input_feature():
    st.subheader("Please input Customer's Feature")
    col1, col2 = st.columns(2)

    with col1:
        Tenure = st.number_input("Tenure", min_value=0, max_value=90, value=10)
        WarehouseToHome = st.number_input("WarehouseToHome", min_value=0, max_value=100, value=5)
        HourSpendOnApp = st.number_input("HourSpendOnApp", min_value=0, max_value=10, value=3)
        NumberOfDeviceRegistered = st.number_input("NumberOfDeviceRegistered", min_value=1, max_value=7, value=1)
        NumberOfAddress = st.number_input("NumberOfAddress", min_value=1, max_value=30, value=1)
        OrderAmountHikeFromlastYear = st.number_input("OrderAmountHikeFromlastYear", min_value=1, max_value=100, value=1)
        CouponUsed = st.number_input("CouponUsed", min_value=0, max_value=30, value=1)
        DaySinceLastOrder = st.number_input("DaySinceLastOrder", min_value=0, max_value=60, value=1)
        CashbackAmount = st.number_input("CashbackAmount", min_value=0, max_value=1000, value=163)

    with col2:
        OrderFrequency = st.number_input("OrderFrequency", min_value=0.0, max_value=30.0, value=1.0, step=0.01)
        ComplainRate = st.number_input("ComplainRate", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        PreferredLoginDevice = st.selectbox("PreferredLoginDevice", ['Mobile Phone', 'Computer'])
        CityTier = st.selectbox("CityTier", ['Small City', 'Medium City', 'Big City'])
        PreferredPaymentMode = st.selectbox("PreferredPaymentMode", ['Debit Card', 'Credit Card', 'E wallet', 'Cash On Delivery', 'UPI'])
        Gender = st.selectbox("Gender", ['Female', 'Male'])
        PreferredOrderCategory = st.selectbox("PreferredOrderCategory", ['Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Grocery', 'Others'])
        SatisfactionScore = st.selectbox("SatisfactionScore", ['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        MaritalStatus = st.selectbox("MaritalStatus", ['Single', 'Married', 'Divorced'])

    df = pd.DataFrame({
        'Tenure': [Tenure],
        'WarehouseToHome': [WarehouseToHome],
        'HourSpendOnApp': [HourSpendOnApp],
        'NumberOfDeviceRegistered': [NumberOfDeviceRegistered],
        'NumberOfAddress': [NumberOfAddress],
        'OrderAmountHikeFromlastYear': [OrderAmountHikeFromlastYear],
        'CouponUsed': [CouponUsed],
        'DaySinceLastOrder': [DaySinceLastOrder],
        'CashbackAmount': [CashbackAmount],
        'OrderFrequency': [OrderFrequency],
        'ComplainRate': [ComplainRate],
        'PreferredLoginDevice': [PreferredLoginDevice],
        'CityTier': [CityTier],
        'PreferredPaymentMode': [PreferredPaymentMode],
        'Gender': [Gender],
        'PreferredOrderCategory': [PreferredOrderCategory],
        'SatisfactionScore': [SatisfactionScore],
        'MaritalStatus': [MaritalStatus]
    })

    return df


# ==============================================================
# Ambil input user
df_customer = user_input_feature()

# Tombol prediksi di bawah input
st.write("")
run_predict = st.button("🔍 Run Prediction", use_container_width=True)

# ==============================================================
# Proses Prediksi
if run_predict:
    with open('model_catboost.sav', 'rb') as file:
        model = pickle.load(file)
    kelas = model.predict(df_customer)[0]
    proba = model.predict_proba(df_customer)[0][1]

    # Tampilkan hasil prediksi
    st.markdown("---")
    st.subheader("Prediction Result")

    if kelas == 1:
        st.markdown("### 🟥 Customer will **CHURN**")
    else:
        st.markdown("### 🟩 Customer will **STAY**")

    st.write(f"**Churn Probability:** {proba:.2%}")

    # Simpan hasil untuk shap
    st.session_state['kelas'] = kelas
    st.session_state['proba'] = proba
    st.session_state['run'] = True

# ==============================================================
# SHAP Explanation
if 'run' in st.session_state and st.session_state['run']:
    st.markdown("---")
    st.subheader("SHAP Force Plot Explanation")

    model_final = model.named_steps['cat']
    transformer = model.named_steps['transformer']
    X_transformed = transformer.transform(df_customer)

    explainer = shap.TreeExplainer(model_final)
    shap_values = explainer.shap_values(X_transformed)

    feat_ordinal = transformer.transformers_[0][2]
    feat_onehot = list(transformer.transformers_[1][1].get_feature_names_out())
    feat_robust = transformer.transformers_[2][2]
    feat_remainder = list(df_customer.columns[transformer.transformers_[3][2]])
    feat = feat_ordinal + feat_onehot + feat_robust + feat_remainder

    st_shap(
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            feature_names=feat,
            matplotlib=False
        ),
        height=300
    )
