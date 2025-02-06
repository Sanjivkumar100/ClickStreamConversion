import streamlit as st
import pickle as pkl
import pandas as pd
from sklearn.preprocessing import StandardScaler
def predict(data,model_path,scale_path,features):
    with open(model_path, 'rb') as file:
        model = pkl.load(file)

    scaled=pkl.load(open(scale_path,'rb'))
    data=scaled.transform(data[features])
    predictions = model.predict(data)
    
    return predictions
def preprocess(data):
    
    if 'page2_clothing_model' in data.columns:
        data['page2_clothing_model'] = data['page2_clothing_model'].str.extract('(\d+)', expand=False).astype(int)
    
    return data

def main():
    st.title("🛒 Clickstream Prediction App")
    st.sidebar.header("Upload Data or Enter Manually")
    
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])
    model_type = st.sidebar.radio("Select Model Type", ("Price Prediction", "Purchase"))
    
    if model_type == "Price Prediction":
        expected_features = ['page1_main_category', 'page2_clothing_model', 'colour']
        model_path = r"C:\Users\sanji\Desktop\ClickStream\XGBoost_clickstream.pkl"
        scaler_path = r"C:\Users\sanji\Desktop\ClickStream\scaler_reg.pkl"
        expected_target = "price"
    else:
        expected_features = ['page1_main_category',	'page2_clothing_model',	'colour',	'location',	'model_photography', 'page', 'price']
        model_path = r"C:\Users\sanji\Desktop\ClickStream\Decision Tree_clickstream.pkl"
        scaler_path = r"C:\Users\sanji\Desktop\ClickStream\scaler_cls.pkl"
        expected_target = "price_2"
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = preprocess(df)
        st.subheader("Uploaded Data Preview")
        st.write(df)
        
        missing_features = [feature for feature in expected_features if feature not in df.columns]
        if missing_features:
            st.error(f"The following required features are missing: {missing_features}")
            return
        
        features = st.sidebar.multiselect("Select Features", expected_features, default=expected_features)
        target = st.sidebar.selectbox("Select Target", expected_target)
        
        if features and target:
            df = df[features + [target]]
            row_index = st.sidebar.selectbox("Select a Row for Prediction", df.index)
            selected_row = df.loc[[row_index], features]
        else:
            st.warning("Please select features and target to proceed.")
    
    else:
        st.sidebar.subheader("Enter Data Manually")

        if model_type == "Price Prediction":

            page1_main_category = st.sidebar.number_input("Page1 Main Category", min_value=1, max_value=5, value=3)
            page2_clothing_model = st.sidebar.text_input("Page2 Clothing Model (e.g. C20)", "C20")
            colour = st.sidebar.number_input("Colour", min_value=1, max_value=20, value=5)
            
            
            

            df = pd.DataFrame({
                'page1_main_category': [page1_main_category],
                'page2_clothing_model': [page2_clothing_model],
                'colour': [colour]
                
            })

        else:  
            page1_main_category = st.sidebar.number_input("Page1 Main Category", min_value=1, max_value=10, value=3)
            page2_clothing_model = st.sidebar.text_input("Page2 Clothing Model (e.g. C20)", "C20")
            colour = st.sidebar.number_input("Colour", min_value=1, max_value=20, value=17)
            location = st.sidebar.number_input("Location", min_value=1, max_value=10, value=3)
            model_photography = st.sidebar.number_input("Model Photography", min_value=1, max_value=5, value=2)
            page=st.sidebar.number_input("Page",min_value=1,max_value=5)
            price = st.sidebar.number_input("Price", min_value=1, max_value=1000, value=50)           
            

           
            df = pd.DataFrame({
                'page1_main_category': [page1_main_category],
                'page2_clothing_model': [page2_clothing_model],
                'colour': [colour],
                'location': [location],
                'model_photography': [model_photography],
                'page':[page],
                'price': [price],
            })

        selected_row = preprocess(df)

    if st.button("Predict"):        
        predictions = predict(selected_row, model_path, scaler_path, expected_features)
        
        if model_type == "Price Prediction":
            st.success(f"Predicted Price: ${round(predictions[0])}")
        else:
            st.success("Going To Buy" if predictions[0]==1 else  "Not Going To Buy")


if __name__=="__main__":
    main()
