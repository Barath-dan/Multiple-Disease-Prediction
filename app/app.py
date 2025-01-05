import streamlit as st
import pickle
import numpy as np
import xgboost
import sklearn
import requests
import io

ckd_model_url = "https://raw.githubusercontent.com/Barath-dan/Multiple-Disease-Prediction/main/pickle_file/ckd_model.pkl"
lvr_model_url = "https://raw.githubusercontent.com/Barath-dan/Multiple-Disease-Prediction/main/pickle_file/lvr_model.pkl"
pkn_model_url = "https://raw.githubusercontent.com/Barath-dan/Multiple-Disease-Prediction/main/pickle_file/pkn_model.pkl"

def load_pickle_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    pickle_data = pickle.loads(response.content)
    return pickle_data

try:
    kidney_model = load_pickle_from_url(ckd_model_url)
    liver_model = load_pickle_from_url(lvr_model_url)
    parkinson_model = load_pickle_from_url(pkn_model_url)
    print("Models loaded successfully!")
except Exception as e:
    print(f"An error occurred: {e}")

def parkinsons(input_data):
    if parkinson_model:
        return parkinson_model.predict(input_data)[0]
    else:
        return "Choose the correct model"
    
def liver(input_data):
    if liver_model:
        return liver_model.predict(input_data)[0]
    else:
        return "Choose the correct model"

def kidney(input_data):
    if kidney_model:
        return kidney_model.predict(input_data)[0]
    else:
        return "Choose the correct model"

def parkinsons_feature_explanation():
    st.write("""
    ### Feature Explanation:
    1. **MDVP:Fo(Hz)**: This refers to the fundamental frequency of speech. A lower or abnormal value may indicate issues with vocal cords, which are often affected in Parkinson’s disease.
    2. **MDVP:Fhi(Hz)**: The maximum frequency of speech. A significant deviation from normal ranges can be a sign of motor control issues, which are common in Parkinson's.
    3. **MDVP:Flo(Hz)**: The minimum frequency of speech. Abnormal values can indicate difficulties with motor control in the vocal apparatus.
    4. **MDVP:Jitter(%)**: Jitter refers to the variability in frequency. Higher jitter values indicate irregularities in voice pitch, which can be a sign of Parkinson’s disease.
    5. **MDVP:Jitter(Abs)**: This is the absolute value of jitter, another indicator of vocal irregularities. High values can point toward motor control problems in the speech apparatus.
    6. **MDVP:RAP**: The relative average perturbation of the fundamental frequency. A higher RAP value may indicate an irregular speech pattern, often associated with Parkinson’s.
    7. **MDVP:PPQ**: The period perturbation quotient, which quantifies the periodicity of the voice. Increased values suggest voice abnormalities related to Parkinson's disease.
    8. **Jitter:DDP**: The difference in jitter between adjacent cycles of the voice. High values indicate irregularities that can be related to Parkinson’s disease.
    9. **MDVP:Shimmer**: This measures the variation in amplitude during speech. Increased shimmer values indicate instability in voice intensity, often found in Parkinson’s.
    10. **MDVP:Shimmer(dB)**: This is the decibel level of shimmer, another way of measuring voice intensity irregularities, common in those with Parkinson’s disease.
    11. **Shimmer:APQ3**: The amplitude perturbation quotient, which measures instability in voice amplitude. Elevated values can signal motor control issues in Parkinson’s.
    12. **Shimmer:APQ5**: Another measure of amplitude instability. Parkinson’s disease often causes irregularities in voice intensity, leading to higher values in this feature.
    13. **MDVP:APQ**: This is a general measure of amplitude perturbation. Higher values typically indicate more irregular speech patterns, which can be linked to Parkinson’s disease.
    14. **Shimmer:DDA**: The difference in the amplitude of successive speech cycles. High values suggest a lack of smoothness in speech, which can be a symptom of Parkinson’s.
    15. **NHR**: The noise-to-harmonics ratio. A higher value indicates more noise in the voice, which is common in people with Parkinson's due to motor control issues.
    16. **HNR**: The harmonic-to-noise ratio. A lower value may indicate irregularities in speech, which are often associated with Parkinson’s.
    17. **RPDE**: The Recurrence Period Density Entropy. Higher values of RPDE indicate greater complexity in speech patterns, which can be indicative of Parkinson’s disease.
    18. **DFA**: The Detrended Fluctuation Analysis. This measures the fractal-like patterns in speech and can show alterations in motor control, which is seen in Parkinson’s.
    19. **Spread1**: This feature is related to the spread of frequency in speech and can indicate motor dysfunction associated with Parkinson’s disease.
    20. **Spread2**: Similar to Spread1, this measures the spread of frequency but in a different manner, which can also be altered in Parkinson's patients.
    21. **D2**: A measure of complexity in speech patterns. Higher values might be associated with more erratic or irregular patterns in speech that are characteristic of Parkinson’s.
    22. **PPE**: The Pitch Period Entropy. It quantifies the irregularity in speech cycles, and higher values are often associated with motor dysfunction in Parkinson’s disease.
    """)
def parkinsons_diet_paln():
    st.write("[Click here for a diet plan for Parkinson's Disease](https://parkinsonfoundation.org/blog/a-complete-parkinsons-diet-guide)")
def parkinson_exercise_plan():
    st.write("[Click here for exercise plan for Parkinson's Disease](https://www.healthline.com/health/parkinsons/yoga-for-parkinsons)")
def kidney_feature_explanation():
    st.write("""
    ### Feature Explanation:
    1. **Age**: The age of the person, which plays a crucial role in assessing the risk for the disease.
    2. **Blood Pressure**: High blood pressure can contribute to the onset of chronic kidney disease, liver disease, and Parkinson’s.
    3. **Specific Gravity**: The concentration of urine can give clues about kidney function.
    4. **Albumin**: High levels can indicate kidney damage or liver dysfunction.
    5. **Sugar**: Elevated blood sugar levels are linked to diabetes and other complications like kidney disease.
    6. **Red Blood Cells**: Abnormal presence may indicate kidney or liver issues.
    7. **Pus Cell**: Presence of pus in the urine may suggest a urinary tract infection or kidney problem.
    8. **Pus Cell Clumps**: Presence of clumps indicates a more severe infection.
    9. **Bacteria**: Can indicate a urinary tract infection (UTI), related to kidney disease.
    10. **Hypertension, Diabetes, and other medical conditions**: These are essential factors in assessing the risk for chronic diseases.
    11. **Diabetes**: Diabetes is a significant risk factor for kidney disease. High blood sugar levels can damage blood vessels in the kidneys, impairing their ability to function.
    12. **Coronary Artery Disease (CAD)**: CAD refers to narrowed or blocked arteries that supply the heart. This condition is often linked to kidney disease due to shared risk factors like high blood pressure and diabetes.
    13. **Appetite**: A decrease in appetite may signal an underlying health issue such as kidney disease. Poor appetite is common in individuals with chronic kidney problems.
    14. **Pedal Edema (Edema)**: Swelling in the feet or ankles can be a sign of kidney disease, as impaired kidneys may lead to fluid retention in the body.
    15. **Anemia**: Anemia refers to a lack of healthy red blood cells. It is often seen in individuals with chronic kidney disease, as the kidneys are responsible for producing erythropoietin, a hormone that helps produce red blood cells.
    """)
def kidney_diet_plan():
    st.write("[Click here for a diet plan for Kidney Disease](https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/eating-nutrition)")
def kidney_exercise_plan():
    st.write("[Click here for exercise plan for Kidney Disease](https://redcliffelabs.com/myhealth/yoga/11-yoga-asanas-for-liver-and-kidney-health/)")

def liver_feature_explanation():
    st.write("""
    ### Feature Explanation:
    1. **Age**: The age of the individual is an important factor in liver disease diagnosis, as older individuals are at higher risk for liver conditions.
    2. **Gender**: Gender plays a role in liver disease, as certain liver conditions are more prevalent in males or females.
    3. **Total Bilirubin**: Bilirubin is a waste product produced by the breakdown of red blood cells. High levels may indicate liver dysfunction or damage.
    4. **Direct Bilirubin**: This is the bilirubin that has been processed by the liver. Elevated levels could indicate liver or bile duct problems.
    5. **Alkaline Phosphatase**: An enzyme found in the liver and bones. High levels may suggest liver damage or bile duct obstruction.
    6. **Alamine Aminotransferase (ALT)**: An enzyme primarily found in the liver. High levels of ALT can indicate liver damage or inflammation.
    7. **Aspartate Aminotransferase (AST)**: Another enzyme found in the liver. Elevated AST levels may indicate liver disease or muscle damage.
    8. **Total Proteins**: Total protein levels in the blood, which can be a sign of liver function. Low protein levels can indicate liver dysfunction.
    9. **Albumin**: A protein produced by the liver. Low albumin levels can indicate poor liver function or liver disease.
    10. **Albumin and Globulin Ratio**: The ratio of albumin to globulins in the blood. A low ratio can indicate liver disease or kidney issues.
    """)
def liver_diet_plan():
    st.write("[Click here for a diet plan for Liver Disease](https://fundahigadoamerica.org/en/news/2021/07/7-foods-for-a-healthy-liver/?campaignid=1600383838&adgroupid=127683227945&keyword=&device=c&utm_source=Google_Ads&gad_source=1&gclid=CjwKCAiA1eO7BhATEiwAm0Ee-KZf_G3W5tEPcMh5rq9GDcTY0NVEaAPCQMzP1kWBtlMuoDImDxSPvhoCqcAQAvD_BwE)")
def liver_exercise_plan():
    st.write("[Click here for exercise plan for Liver Disease](https://www.google.com/search?q=yoga+for+liver+health&oq=yoga+for+liver+health&gs_lcrp=EgZjaHJvbWUyCQgAEEUYORiABDIICAEQABgWGB4yCAgCEAAYFhgeMggIAxAAGBYYHjIICAQQABgWGB4yCAgFEAAYFhgeMggIBhAAGBYYHjIGCAcQRRg80gEINjgwNGowajSoAgCwAgA&sourceid=chrome&ie=UTF-8)")

#Streamlit UI     

st.set_page_config(
    page_title="Multiple Disease Prediction App",
    layout="wide",
)

# CSS to set a background image
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url({image_url});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    ) 
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:", ["Home", "Parkinson Prediction", "Kidney Disease Prediction", "Liver Disease Prediction"])

if options == "Home":
    background_image_url = "https://raw.githubusercontent.com/Barath-dan/Multiple-Disease-Prediction/main/backgrounds/shield_generated.jpg"
    set_background(background_image_url)
    st.title("Multiple Disease Prediction Application")
    st.write("""
        Welcome to the **Multiple Disease Prediction Application**!\n  
        This app helps you predict the likelihood of the following diseases:
        - **Parkinson's Disease**
        - **Liver Disease**
        - **Chronic Kidney Disease**
    """)

    st.subheader("How to Use the Application:")
    st.write("""
        1. Navigate to the specific page for the disease you want to predict using the **sidebar**.
        2. Enter the diagnostic values in the fields provided.
        3. Click the **Predict** button to view your results.
    """)

    st.info("Choose a prediction page from the sidebar to get started!")

    st.warning("""
        **Important Note:**  
        This application is intended for informational purposes only.  
        The predictions provided are based on machine learning models and should not be considered a substitute for professional medical advice.  
        Always consult with a licensed medical professional for an accurate diagnosis and appropriate treatment.
    """)

elif options == "Parkinson Prediction":
    st.title("Parkinson's Disease Prediction")
    background_image_url = "https://raw.githubusercontent.com/Barath-dan/Multiple-Disease-Prediction/main/backgrounds/wangsina_333_03_2022_8.jpg"
    set_background(background_image_url)
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        if st.button("Explanation"):
            parkinsons_feature_explanation()
    with b_col2:
        if st.button("Get Diet Plan"):
            parkinsons_diet_paln()
    with b_col3:
        if st.button("Yoga for Parkinson's"):
            parkinson_exercise_plan()
    st.write("Enter the required details for prediction:")
    col1, col2, col3, col4= st.columns(4)
    with col1:
        f1=st.number_input("MDVP:Fo(Hz)",value=0.0)
        f2=st.number_input("MDVP:Fhi(Hz)",value=0.0)
        f3=st.number_input("MDVP:Flo(Hz)",value=0.0)
        f4=st.number_input("MDVP:Jitter(%)",value=0.0)
        f5 = st.number_input("MDVP:Jitter(Abs)", value=0.0)
        f21 = st.number_input("D2", value=0.0)
    with col2:
        f6 = st.number_input("MDVP:RAP", value=0.0)
        f7 = st.number_input("MDVP:PPQ", value=0.0)
        f8 = st.number_input("Jitter:DDP", value=0.0)
        f9 = st.number_input("MDVP:Shimmer", value=0.0)
        f10 = st.number_input("MDVP:Shimmer(dB)", value=0.0)
        f22 = st.number_input("PPE", value=0.0)
    with col3:  
        f11 = st.number_input("Shimmer:APQ3", value=0.0)
        f12 = st.number_input("Shimmer:APQ5", value=0.0)
        f13 = st.number_input("MDVP:APQ", value=0.0)
        f14 = st.number_input("Shimmer:DDA", value=0.0)
        f15 = st.number_input("NHR", value=0.0)
    with col4:
        f16 = st.number_input("HNR", value=0.0)
        f17 = st.number_input("RPDE", value=0.0)
        f18 = st.number_input("DFA", value=0.0)
        f19 = st.number_input("spread1", value=0.0)
        f20 = st.number_input("spread2", value=0.0)

    input_data = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22])
    input_data = input_data.reshape(1, -1)

    if st.button("Predict"):
        if np.all(input_data == 0):
            st.warning("Please fill in all fields before predicting!")
        else:
            result=parkinsons(input_data)
            st.write(f"Diagnosis Result: {'Yes' if result == 1 else 'No'}")
    else:
        st.write("Click to Predict !!!")

elif options == "Kidney Disease Prediction":
    background_image_url = "https://raw.githubusercontent.com/Barath-dan/Multiple-Disease-Prediction/main/backgrounds/wwang_040522_7.jpg"
    set_background(background_image_url)
    st.title("Chronic Kidney Disease (CKD) Prediction")
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        if st.button("Explanation"):
            kidney_feature_explanation()
    with b_col2:
        if st.button("Get Diet Plan"):
            kidney_diet_plan()
    with b_col3:
        if st.button("Yoga for CKD"):
            kidney_exercise_plan()
    st.write("Enter the required details for prediction:")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", value=0)
        blood_pressure = st.number_input("Blood Pressure", value=0)
        specific_gravity = st.number_input("Specific Gravity", value=0.0)
        albumin = st.number_input("Albumin", value=0)
        white_blood_cell_count = st.number_input("White Blood Cell Count", value=0)
    with col2:
        sugar = st.number_input("Sugar", value=0)
        blood_glucose_random = st.number_input("Blood Glucose Random", value=0)
        blood_urea = st.number_input("Blood Urea", value=0.0)
        serum_creatinine = st.number_input("Serum Creatinine", value=0.0)
        red_blood_cell_count = st.number_input("Red Blood Cell Count", value=0.0)
    with col3:
        sodium = st.number_input("Sodium", value=0)
        potassium = st.number_input("Potassium", value=0.0)
        hemoglobin = st.number_input("Hemoglobin", value=0.0)
        packed_cell_volume = st.number_input("Packed Cell Volume", value=0)
    st.markdown("---")
    st.write("Select an Option")
    cat_col1, cat_col2, cat_col3 = st.columns(3)
    with cat_col1:
        red_blood_cells = st.radio("Red Blood Cells", options=["Normal", "Abnormal"])
        rbc = 1 if red_blood_cells == "Normal" else 0
        pus_cell = st.radio("Pus Cell", options=["Normal", "Abnormal"])
        pus = 1 if pus_cell == "Normal" else 0
        pus_cell_clumps = st.radio("Pus Cell Clumps", options=["Not Present", "Present"])
        clumps = 1 if pus_cell_clumps == "Present" else 0
        anemia = st.radio("Anemia", options=["No", "Yes"])
        ane = 1 if anemia == "Yes" else 0
    with cat_col2:
        bacteria = st.radio("Bacteria", options=["Not Present", "Present"])
        bac = 1 if bacteria == "Present" else 0
        hypertension = st.radio("Hypertension", options=["No", "Yes"])
        htn = 1 if hypertension == "Yes" else 0
        diabetes_mellitus = st.radio("Diabetes Mellitus", options=["No", "Yes"])
        dm = 1 if diabetes_mellitus == "Yes" else 0
    with cat_col3:
        coronary_artery_disease = st.radio("Coronary Artery Disease", options=["No", "Yes"])
        cad = 1 if coronary_artery_disease == "Yes" else 0
        appetite = st.radio("Appetite", options=["Good", "Poor"])
        app = 1 if appetite == "Poor" else 0
        pedal_edema = st.radio("Pedal Edema", options=["No", "Yes"])
        edema = 1 if pedal_edema == "Yes" else 0
        
        
    # Collecting all inputs into a NumPy array
    input_data = np.array([age, blood_pressure, specific_gravity, albumin, sugar, 
                           rbc, pus, clumps, bac, blood_glucose_random, 
                           blood_urea, serum_creatinine, sodium, potassium, 
                           hemoglobin, packed_cell_volume, white_blood_cell_count, 
                           red_blood_cell_count, htn, dm, cad, app, edema, ane])
   
    input_data = input_data.reshape(1, -1)

    if st.button("Predict"):
        if np.all(input_data) == 0:
            st.warning("Please fill in all fields before predicting!")
        else:
            result=kidney(input_data)
            st.write(f"Diagnosis Result: {'CKD Not Detected' if result == 1 else 'CKD Detected'}")
    else:
        st.write("Click to Predict !!!")

elif options == "Liver Disease Prediction":
    background_image_url = "https://raw.githubusercontent.com/Barath-dan/Multiple-Disease-Prediction/main/backgrounds/hand_care_healt_8.jpg"
    set_background(background_image_url)
    st.title("Liver Disease Prediction")
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        if st.button("Explanation"):
            liver_feature_explanation()
    with b_col2:
        if st.button("Get Diet Plan"):
            liver_diet_plan()
    with b_col3:
        if st.button("Yoga for Liver Health"):
            liver_exercise_plan()
    st.write("Enter the required details for prediction:")
    gender = st.radio("Gender", options=["Female", "Male"])
    gender_value = 1 if gender == "Male" else 0
    col1, col2, col3 = st.columns(3)
    with col1:
        f1 = st.number_input("Age", value=0)
        f3 = st.number_input("Total Bilirubin", value=0.0)
        f4 = st.number_input("Direct Bilirubin", value=0.0)
    with col2:
        f5 = st.number_input("Alkaline Phosphotase", value=0)
        f6 = st.number_input("Alamine Aminotransferase", value=0)
        f7 = st.number_input("Aspartate Aminotransferase", value=0)
    with col3:
        f8 = st.number_input("Total Proteins", value=0.0)
        f9 = st.number_input("Albumin", value=0.0)
        f10 = st.number_input("Albumin and Globulin Ratio", value=0.0)

    input_data = np.array([f1, gender_value, f3, f4, f5, f6, f7, f8, f9, f10])
    input_data = input_data.reshape(1, -1)
    if st.button("Predict"):
        if np.all(input_data == 0):
            st.warning("Please fill in all fields before predicting!")
        else:
            result=liver(input_data)
            st.write(f"Diagnosis Result: {'Liver Disease Detected' if result == 1 else 'Liver Disease Not Detected'}")
    else:
        st.write("Click to Predict !!!")
    

