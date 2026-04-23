import os, sys, warnings

import numpy as np

import pandas as pd

import streamlit as st

import matplotlib.pyplot as plt

import posixpath

 

import joblib

import tarfile

import tempfile

 

import boto3

import sagemaker

from sagemaker.predictor import Predictor

from sagemaker.serializers import CSVSerializer

from sagemaker.serializers import JSONSerializer

from sagemaker.deserializers import JSONDeserializer

from sagemaker.serializers import NumpySerializer

from sagemaker.deserializers import NumpyDeserializer

 

 

from sklearn.pipeline import Pipeline

import shap

 

from joblib import dump

from joblib import load

 

 

 

# Setup & Path Configuration

warnings.simplefilter("ignore")

 

# Fix path for Streamlit Cloud (ensure 'src' is findable)

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:

    sys.path.append(project_root)

 

#from src.feature_utils import extract_features

from src.Custom_Classes import DropHighMissingCols, TransactionFeatureEngineer, DropHighCorrelation

 

# Access the secrets

aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]

aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]

aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]

aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]

aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

 

# AWS Session Management

@st.cache_resource # Use this to avoid downloading the file every time the page refreshes

def get_session(aws_id, aws_secret, aws_token):

    return boto3.Session(

        aws_access_key_id=aws_id,

        aws_secret_access_key=aws_secret,

        aws_session_token=aws_token,

        region_name='us-east-1'

    )

 

session = get_session(aws_id, aws_secret, aws_token)

sm_session = sagemaker.Session(boto_session=session)

 

# Data & Model Configuration

#df_features = extract_features()

 


 

#MODEL_INFO = {
#
#    "endpoint"  : aws_endpoint,
#
#    "explainer" : "explainer_fraud.shap",
#
#    "pipeline"  : "fine_tuned_pipeline.tar.gz",
#
#    "keys"      : ['TransactionAmt','addr1','addr2'],
#
#    "inputs"    : [{"name": k, "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01} for k in ['TransactionAmt','addr1','addr2']]
#
#}

MODEL_INFO = {
    "endpoint"  : aws_endpoint,
    "explainer" : "explainer_fraud.shap",
    "pipeline"  : "fine_tuned_pipeline.tar.gz",
    "keys"      : ['TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V138', 'V139', 'V140', 'V141', 'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149', 'V150', 'V151', 'V152', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V168', 'V169', 'V170', 'V171', 'V172', 'V173', 'V174', 'V175', 'V176', 'V177', 'V178', 'V179', 'V180', 'V181', 'V182', 'V183', 'V184', 'V185', 'V186', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197', 'V198', 'V199', 'V200', 'V201', 'V202', 'V203', 'V204', 'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V211', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V218', 'V219', 'V220', 'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236', 'V237', 'V238', 'V239', 'V240', 'V241', 'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V248', 'V249', 'V250', 'V251', 'V252', 'V253', 'V254', 'V255', 'V256', 'V257', 'V258', 'V259', 'V260', 'V261', 'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V269', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278', 'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V286', 'V287', 'V288', 'V289', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298', 'V299', 'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321', 'V322', 'V323', 'V324', 'V325', 'V326', 'V327', 'V328', 'V329', 'V330', 'V331', 'V332', 'V333', 'V334', 'V335', 'V336', 'V337', 'V338', 'V339', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo'],
    "inputs"    : [
        {"name": "TransactionAmt",  "label": "Transaction Amount ($)",           "min": 0.0,  "max": 10000.0,    "default": 100.0,  "step": 0.01},
        {"name": "card1",           "label": "Card Number",                      "min": 0.0,  "max": 20000.0,    "default": 0.0,    "step": 1.0},
        {"name": "addr1",           "label": "Billing Zip Code",                 "min": 0.0,  "max": 600.0,      "default": 0.0,    "step": 1.0},
        {"name": "dist1",           "label": "Distance from Billing Address",    "min": 0.0,  "max": 10000.0,    "default": 0.0,    "step": 1.0},
        {"name": "TransactionDT",   "label": "Transaction Timestamp",            "min": 0.0,  "max": 15811131.0, "default": 86400.0,"step": 1.0},
        {"name": "C1",              "label": "Number of Cards on Billing Address","min": 0.0, "max": 2000.0,     "default": 0.0,    "step": 1.0},
        {"name": "C4",              "label": "Number of Transactions on Card",   "min": 0.0,  "max": 2000.0,     "default": 0.0,    "step": 1.0},
        {"name": "C8",              "label": "Number of Addresses on Card",      "min": 0.0,  "max": 2000.0,     "default": 0.0,    "step": 1.0},
    ]
}
 

 

def load_pipeline(_session, bucket, key):

    s3_client = _session.client('s3')

    filename=MODEL_INFO["pipeline"]

 

    s3_client.download_file(

        Filename=filename,

        Bucket=bucket,

        Key= f"{key}/{os.path.basename(filename)}")

        # Extract the .joblib file from the .tar.gz

    with tarfile.open(filename, "r:gz") as tar:

        tar.extractall(path=".")

        #joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

        joblib_file = [f for f in tar.getnames() if f.endswith('.pkl')][0]

   

 

    # Load the full pipeline

    return joblib.load(f"{joblib_file}")

 

def load_shap_explainer(_session, bucket, key, local_path):

    s3_client = _session.client('s3')

    local_path = local_path

 

    # Only download if it doesn't exist locally to save time

    if not os.path.exists(local_path):

        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)

       

    with open(local_path, "rb") as f:

        return load(f)

        #return shap.Explainer.load(f)

 

# Prediction Logic

#def call_model_api(input_df):
#    print(f"Type received: {type(input_df)}")
#    print(f"Value: {input_df}")
#    predictor = Predictor(
#        endpoint_name=MODEL_INFO["endpoint"],
#        sagemaker_session=sm_session,
#        serializer=JSONSerializer(),
#        deserializer=NumpyDeserializer()
#    )
#    try:
        #if you do option 1 you want to uncomment the ones you want to use and comment the ones you dont use
        # For regression
        # raw_pred = predictor.predict(input_df)
        # pred_val = pd.DataFrame(raw_pred).values[-1][0]
        # return round(float(pred_val), 4), 200
        # For classification
#        if isinstance(input_df, pd.DataFrame): #this whole if else statement is from claude can be taken out
#            input_data = input_df.to_dict(orient='records')
#        else:
#            input_data = input_df
#        raw_pred = predictor.predict(input_df)
#        pred_val = pd.DataFrame(raw_pred).values[-1][0]
#        #mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}
#        mapping = {0: "Legitimate", 1: "Fraud"}
#        return mapping.get(pred_val), 200
#    except Exception as e:
#        return f"Error: {str(e)}", 500

def call_model_api(input_data):
    predictor = Predictor(
        endpoint_name     = MODEL_INFO["endpoint"],
        sagemaker_session = sm_session,
        serializer        = JSONSerializer(),
        deserializer      = JSONDeserializer()
    )
    try:
        raw_pred = predictor.predict(input_data)
        print(f"Raw prediction: {raw_pred}")
        pred_val = int(raw_pred['prediction'][0])
        mapping  = {0: "Legitimate", 1: "Fraud"}
        return mapping.get(pred_val), 200
    except Exception as e:
        return f"Error: {str(e)}", 500



# Local Explainability

#def display_explanation(input_df, session, aws_bucket):
#
#    explainer_name = MODEL_INFO["explainer"]
#    explainer = load_shap_explainer(session, aws_bucket, posixpath.join('explainer', explainer_name),os.path.join(tempfile.gettempdir(), explainer_name))
#    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')
#    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-2])
#    input_df_transformed = preprocessing_pipeline.transform(input_df)
#    feature_names = best_pipeline[:-2].get_feature_names_out()
#   input_df_transformed = pd.DataFrame(input_df_transformed, columns=feature_names)
#    shap_values = explainer(input_df_transformed)
#    st.subheader("🔍 Decision Transparency (SHAP)")
#    fig, ax = plt.subplots(figsize=(10, 4))
#    #shap.plots.waterfall(shap_values[0], max_display=10)
#    #shap.plots.waterfall(shap_values[0, :, 0]) #classification
#    shap.plots.waterfall(shap_values[0, :, 1])  # class 1 = fraud
#    st.pyplot(fig)
#    # top feature
#    #regression
#    # top_feature = pd.Series(shap_values[0].values, index=shap_values[0].feature_names).abs().idxmax()
#    #classification
#    top_feature = pd.Series(shap_values[0, :, 0].values, index=shap_values[0, :, 0].feature_names).abs().idxmax()
#    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")

def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(session, aws_bucket, posixpath.join('explainer', explainer_name), os.path.join(tempfile.gettempdir(), explainer_name))
    
    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')
    
    # Remove last 3 steps (undersampler, sampler, model)
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-3])
    input_df_transformed   = preprocessing_pipeline.transform(input_df)

    # Get real feature names by applying each mask in order
    variance       = best_pipeline.named_steps['variance']
    kbest          = best_pipeline.named_steps['kbest']
    drop_collinear = best_pipeline.named_steps['drop_collinear']

    # Start with column names after feature engineering
    sample        = best_pipeline.named_steps['drop_missing'].transform(input_df)
    sample        = best_pipeline.named_steps['feature_engineering'].transform(sample)
    feature_names = np.array(sample.columns.tolist())

    # Apply each mask in order
    feature_names = feature_names[variance.get_support()]
    feature_names = feature_names[kbest.get_support()]
    feature_names = np.array([
        col for col in feature_names
        if col not in drop_collinear.cols_to_drop_
    ])

    input_df_transformed = pd.DataFrame(input_df_transformed, columns=feature_names)
    shap_values = explainer(input_df_transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 1])
    st.pyplot(fig)

    top_feature = pd.Series(
        shap_values[0, :, 1].values,
        index=shap_values[0, :, 1].feature_names
    ).abs().idxmax()
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")

 

 

# Streamlit UI

st.set_page_config(page_title="ML Deployment", layout="wide")

st.title("👨‍💻 ML Deployment")

 

with st.form("pred_form"):
    st.subheader(f"Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp.get('label', inp['name']),
                min_value=inp['min'], max_value=inp['max'], value=inp['default'], step=inp['step']
        )


 
#    for i, inp in enumerate(MODEL_INFO["inputs"]):
#        with cols[i % 2]:
#            user_inputs[inp['name']] = st.number_input(
#                inp['name'].replace('_', ' ').upper(),
#                min_value=inp['min'], max_value=inp['max'], value=inp['default'], step=inp['step']
#
#            )

   

    submitted = st.form_submit_button("Run Prediction")

 

#if submitted:
##    res, status = call_model_api([user_inputs])
#    if status == 200:
#        st.metric("Prediction Result", res)
#        display_explanation([user_inputs],session, aws_bucket)
#    else:
#        st.error(res)

if submitted:
    # Start with all keys set to 0
    full_row = {col: 0 for col in MODEL_INFO["keys"]}
    
    # Set categorical columns to default string values
    cat_defaults = {
        'ProductCD'     : 'W',
        'card4'         : 'visa',
        'card6'         : 'debit',
        'P_emaildomain' : 'gmail.com',
        'R_emaildomain' : 'gmail.com',
        'M1'            : 'T',
        'M2'            : 'T',
        'M3'            : 'T',
        'M4'            : 'M0',
        'M5'            : 'T',
        'M6'            : 'F',
        'M7'            : 'T',
        'M8'            : 'T',
        'M9'            : 'T',
        'id_12'         : 'Found',
        'id_15'         : 'New',
        'id_16'         : 'Found',
        'id_23'         : 'TRANSPARENT',
        'id_27'         : 'Found',
        'id_28'         : 'New',
        'id_29'         : 'Found',
        'id_35'         : 'T',
        'id_36'         : 'F',
        'id_37'         : 'T',
        'id_38'         : 'T',
        'DeviceType'    : 'desktop',
        'DeviceInfo'    : 'Windows',
    }
    full_row.update(cat_defaults)

    # Override with user inputs
    full_row.update(user_inputs)

    # Convert numeric values to float but leave strings alone
    full_row = {
        k: (float(v) if not isinstance(v, str) else v)
        for k, v in full_row.items()
    }

    input_data = [full_row]

    res, status = call_model_api(input_data)
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(pd.DataFrame(input_data), session, aws_bucket)
    else:
        st.error(res)
