import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd
import numpy as np
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from models import RandomForest


# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(
     page_title="Microsegmentation for Business",
     page_icon=":technologist:",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
 )


# LAYING OUT THE TOP SECTION OF THE APP
row_1, row_2 = st.columns((1, 2))


with row_1:
    st.title("Microsegmentation for Business")

with row_2:
    st.subheader(
        """
    Generate microsegments for business use cases and get insights through this application.   
    Do you wonder what are the  characteristics of customers...
    1) who has a high conversion to buy a product over the rest of the group?
    2) who has a higher chance to churn compared to others? 
    .. or any other similar use cases?  
    
    """
    )

    st.write("Upload data per the instructions and get answers to those questions.")
############################################################################################################################################
# FIRST SEPERATOR
st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

#STEP 1: UPLOAD DATA
st.subheader("Step 1: Upload data")
st.markdown(":pushpin: Instructions :   \n\
  1) Please label the target variable (purchase/churn etc) as 'target' in the data file.    \n\
  2) Please upload the data file in .csv format.   \n\
  3) Remove any columns with IDs by choosing the ID filter")
uploaded_file = st.file_uploader("", type=['csv'])

# loading data 
# TO DO - na values in data file to be filled with some strings
@st.cache(persist=True, allow_output_mutation=True)
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file, low_memory= False)
    return df

@st.cache()
def remove_columns(data,drop_cols):
    return data.drop(drop_cols, axis = 1, inplace = True)


if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.markdown(":white_check_mark: Data loaded successfully!")
    
    if st.checkbox('Show Uploaded data'):
        gb = GridOptionsBuilder.from_dataframe(data)
        gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
        gridOptions = gb.build()
        AgGrid(data,gridOptions=gridOptions,enable_enterprise_modules=True)

    if st.checkbox('Drop ID columns'):
        drop_cols = st.multiselect('Please select columns to drop', data.columns)
        remove_columns(data,drop_cols)
        st.write("Columns dropped successfully")
        st.dataframe(data.head())

    cont_cols, cat_cols = cont_cat_split(data)
    missing = data.isnull().sum().sort_values(ascending = False)
    missing = missing[missing > 0]
    
    if st.checkbox('Data Summary'):
        st.markdown(f'Total Rows : {data.shape[0]:,} | Total Columns : {data.shape[1]:,}')
        missing = data.isnull().sum().sort_values(ascending = False)
        missing = missing[missing > 0]
        st.markdown(f'Columns with missing values : {len(missing)} --> {dict(missing)}')
        st.write(f'Continuous columns : {len(cont_cols)} --> {cont_cols}')
        st.write(f'Categorical columns : {len(cat_cols)} --> {cat_cols}')
else:
    st.write("Please upload a CSV file")


############################################################################################################################################
# SECOND SEPERATOR
st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


#STEP 2: PREPROCESS DATA
st.subheader("Step 2: Preprocess data")

if uploaded_file is not None:
    if st.checkbox('Replace missing values'):
        if len(list(set(cont_cols).intersection(set(missing.index)))) > 0:
            data[cont_cols] = data[cont_cols].fillna(data[cont_cols].mean())
            st.markdown(":arrow_forward: Missing values for below continuos variables replaced with mean successfully")
            st.markdown(list(set(cont_cols).intersection(set(missing.index))))
            st.write(data.head())
        else :
            st.markdown(":arrow_forward: No missing values found in continuos variables")
        if len(list(set(cat_cols).intersection(set(missing.index)))) > 0:
            data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])
            st.markdown(":arrow_forward: Missing values for below categorical variables replaced with mode successfully")
            st.markdown(list(set(cat_cols).intersection(set(missing.index))))
            st.write(data.head())
        else :
            st.markdown(":arrow_forward: No missing values found in categorical variables")
        missing = data.isnull().sum().sort_values(ascending = False)
        missing = missing[missing > 0]
        if len(missing) == 0:
            st.markdown(":white_check_mark: No missing values found post pre processing!")
        else:
            st.markdown(":warning: Missing values found post pre processing!")
            st.markdown(f'Columns with missing values :{len(missing)} --> {dict(missing)}')
        preprocessed_data = pd.get_dummies(data, columns = cat_cols)
else:
    st.write("Please upload a CSV file")



############################################################################################################################################
# THIRD SEPERATOR
st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


#STEP 3: GENERATE MICRO-SEGMENTS
st.subheader("Step 3: Generate Microsegments")
row_3, row_4, row_5 = st.columns(3)
with row_3:
    st.number_input('Number of Microsegments', min_value=1, max_value=100, value=1)

with row_4:
    st.number_input('Expected uplift over the base rate', min_value=1, max_value=100, value=1)

with row_5:
    st.number_input('Expected lead volume', min_value=1, max_value=100000, value=1)

st.button('Generate Microsegments')

def rf_param_selector():
    criterion = st.selectbox("criterion", ["gini", "entropy"])
    n_estimators = st.number_input("n_estimators", 50, 300, 100, 10)
    max_depth = st.number_input("max_depth", 1, 50, 5, 1)
    min_samples_split = st.number_input("min_samples_split", 1, 20, 2, 1)
    max_features = st.selectbox("max_features", [None, "auto", "sqrt", "log2"])

    params = {
        "criterion": criterion,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
        "n_jobs": -1,
    }

    model = RandomForestClassifier(**params)
    return model

if uploaded_file is not None:
    pass
else:
    st.write("Please upload a CSV file")
