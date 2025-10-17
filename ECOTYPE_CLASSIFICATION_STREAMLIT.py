import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =========== CONFIGURABLE =============
BACKGROUND_IMAGE = 'C:/Users/MUTHU SELVAM/OneDrive/Desktop/VS CODE/forest_bg.jpg'    #Replace with your forest image file name
CREATOR = 'MUTHU SELVAM'
APP_TITLE = 'EcoType Forest Cover Type Prediction'

# ================== FEATURES (edit to match your training!) ===================
final_features = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
    'Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3', 'Wilderness_Area_4',
    'Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3', 'Soil_Type_4', 'Soil_Type_5', 'Soil_Type_6',
    'Soil_Type_9', 'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12', 'Soil_Type_13', 'Soil_Type_16',
    'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19', 'Soil_Type_20', 'Soil_Type_21', 'Soil_Type_22',
    'Soil_Type_23', 'Soil_Type_24', 'Soil_Type_26', 'Soil_Type_27', 'Soil_Type_28', 'Soil_Type_29',
    'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32', 'Soil_Type_33', 'Soil_Type_34', 'Soil_Type_35',
    'Soil_Type_38', 'Soil_Type_39', 'Soil_Type_40'
]

continuous_cols = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'
]
wilderness_areas = ['Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3', 'Wilderness_Area_4']
soil_types = [f'Soil_Type_{i}' for i in [
    1,2,3,4,5,6,9,10,11,12,13,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,
    32,33,34,35,38,39,40
]]
cover_type_names = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

# ================== LOAD MODEL/SCALER ===================
model = joblib.load('best_forest_cover_model.pkl')
scaler = joblib.load('scaler.pkl')

# ========== STYLING (background and fonts) =============
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: 'C:/Users/MUTHU SELVAM/OneDrive/Desktop/VS CODE/forest_bg.jpg';
        background-size: cover;
        background-position: center;
        font-family: Arial, Helvetica, sans-serif;
    }}
    div.block-container{{padding-top:2rem}}
    .title-head {{text-align:center; color:#155724; font-size:2.4rem; font-weight:bold}}
    .created {{"text-align:center; color:grey; font-size:1rem; margin-top:40px"}}
    </style>
    """, unsafe_allow_html=True
)

# ======== MAIN UI ==========
st.markdown(f"<div class='title-head'>{APP_TITLE}</div>", unsafe_allow_html=True)
st.write("")
st.info("This app predicts the most likely forest cover type given area features, soil, and wilderness type. Input values below and press **Predict**!")

with st.form(key="forest_form"):
    col1, col2 = st.columns(2)  # To make the form look neat side by side

    user_input = {}
    # Continuous/numeric features side-by-side
    for idx, col in enumerate(continuous_cols):
        with [col1, col2][idx % 2]:
            user_input[col] = st.number_input(
                f"{col}",
                min_value=0.0, max_value=7000.0 if col == "Elevation" else 5000.0,
                value=100.0, step=1.0
            )

    st.markdown("---")
    st.write("#### Wilderness Area (choose one)")
    wa_index = st.radio("Wilderness Area", wilderness_areas, horizontal=True)
    for wa in wilderness_areas:
        user_input[wa] = int(wa == wa_index)

    st.write("#### Soil Type (choose one)")
    soil_index = st.selectbox("Soil Type", soil_types)
    for soil in soil_types:
        user_input[soil] = int(soil == soil_index)

    submitted = st.form_submit_button("Predict Forest Cover Type")

# ========== Prediction and Output ==========
def preprocess_input(user_input):
    for col in continuous_cols:
        user_input[col] = np.log1p(user_input[col])
    df = pd.DataFrame([user_input])[final_features]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)
    df_scaled = scaler.transform(df)
    return df_scaled

if submitted:
    try:
        X = preprocess_input(user_input)
        pred = model.predict(X)[0]
        st.success(f"🌳 **Predicted Forest Cover Type:** {cover_type_names.get(pred, 'Unknown')} (Class {pred})")
        st.markdown(
            f"""
            <div style="background-color:#e9ffe7;padding:1em;border-radius:12px; margin-top: 1em">
                <b>Prediction Insight:</b>
                <ul>
                  <li>Spruce/Fir and Lodgepole Pine are the most common forest types in mountainous terrain.</li>
                  <li>Different soil and wilderness areas strongly influence cover type.</li>
                  <li>Elevation and hydrology distances usually play key roles in classifying forest types.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error occurred: {e}")

# ========== Footer / Credit ==========
st.markdown(
    f"<hr><div class='created'>Created by <b>{CREATOR}</b> | 2025</div>",
    unsafe_allow_html=True
)
