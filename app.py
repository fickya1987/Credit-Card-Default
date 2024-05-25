import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb

def main():
    st.title('Prediksi Resiko Kredit Pelanggan/Nasabah')

    # Load the saved model
    # model = joblib.load('saved_model.pkl')
    model = xgb.XGBClassifier()
    model.load_model('xgb_model.json')

    # Create sidebar with option to upload csv or manually input the fields
    st.sidebar.subheader('Data Input')
    input_option = st.sidebar.radio('Pilih Opsi', ('Upload CSV', 'Input Parameter'))

    if input_option == 'Upload CSV':
        # File upload option 
        uploaded_file = st.sidebar.file_uploader('Upload CSV', type=['CSV'])

        if uploaded_file is not None:
            # Read the CSV file
            data = pd.read_csv(uploaded_file)
            st.subheader('Uploaded Data')
            data.drop(['Unnamed: 0'], axis=1, inplace=True)
            st.write(data)

            # Preprocess the data
            preprocessed_data = preprocess_data_csv(data)

            # Make predictions 
            prediction = model.predict(preprocessed_data)
            prediction_proba = model.predict_proba(preprocessed_data)[:, 1]

            # Create a DataFrame with predictions
            predictions_df = data
            predictions_df['Predicted SeriousDlqin2yrs'] = prediction
            predictions_df['Predicted Probability of default'] = prediction_proba

            # Drop 'SeriousDlqin2yrs' column
            predictions_df.drop('SeriousDlqin2yrs', axis=1, inplace=True)

            # Sort DataFrame by 'Predicted Probability of default' in descending order
            predictions_df.sort_values('Predicted Probability of default', ascending=False, inplace=True)

            # Rearrange the columns
            column_order = ['Predicted SeriousDlqin2yrs', 'Predicted Probability of default'] + predictions_df.columns[:-2].tolist()
            predictions_df = predictions_df[column_order]

            # Display the predictions DataFrame
            st.subheader('Predictions')
            st.write(predictions_df)
    else:
        # Manual entry option
        st.subheader('Masukkan Informasi Pelanggan/Nasabah')
        revolving_utilization = st.number_input('Penggunaan Bergulir dari Jalur Kredit Tanpa Jaminan', min_value=0, value=0)
        Usia = st.number_input('Usia', min_value=0, value=0)
        Rasio_Hutang = st.number_input('Rasio Hutang', min_value=0.0, value=0.0)
        Pendapatan_Bulanan = st.number_input('Pendapatan Bulanan', min_value=0.0, value=0.0)
        Pinjaman_Kredit = st.number_input('Jumlah Jalur Kredit dan Pinjaman yang Terbuka', min_value=0, value=0)
        Pinjaman_Properti = st.number_input('Pinjaman Properti', min_value=0, value=0)
        Pihak_Tertanggung = st.number_input('Pihak Tertanggung', min_value=0, value=0)
        Tunggakan_Selama_90_Hari = st.number_input('Tunggakan Selama 90 Hari', min_value=0, value=0)

        if st.button('Predict'):
            # Create a DataFrame with the entered values
            data = pd.DataFrame({'RevolvingUtilizationOfUnsecuredLines': [revolving_utilization],
                                 'age': [Usia],
                                 'DebtRatio': [Rasio_Hutang],
                                 'MonthlyIncome': [Pendapatan_Bulanan],
                                 'NumberOfOpenCreditLinesAndLoans': [Pinjaman_kredit],
                                 'NumberRealEstateLoansOrLines': [Pinjaman_Properti],
                                 'NumberOfDependents': [Pihak_Tertanggung],
                                 'TotalDelinquencies90DaysLate': [Tunggakan_Selama_90_Hari]})

            # Preprocess the data
            preprocessed_data = preprocess_data(data)

            # Make predictions
            prediction = model.predict(preprocessed_data)
            prediction_proba = model.predict_proba(preprocessed_data)[:, 1]

            # Create a DataFrame with predictions
            predictions_df = pd.DataFrame({
                'Predicted SeriousDlqin2yrs': prediction,
                'Predicted Probability of default': prediction_proba
            })

            # Display the predictions DataFrame
            st.subheader('Predictions')
            st.write(predictions_df)

def preprocess_data_csv(test_df):
    test_df = test_df.drop(['SeriousDlqin2yrs'], axis=1)

    # Create composite feature
    test_df['TotalDelinquencies90DaysLate'] = test_df['NumberOfTimes90DaysLate'] + test_df['NumberOfTime60-89DaysPastDueNotWorse'] + test_df['NumberOfTime30-59DaysPastDueNotWorse']

    # Drop the correlated features
    test_df.drop(['NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTime30-59DaysPastDueNotWorse'], axis=1, inplace=True)

    numeric_features = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'DebtRatio',
                        'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
                        'NumberRealEstateLoansOrLines', 'NumberOfDependents', 'TotalDelinquencies90DaysLate']

    # Impute missing values in the numeric features
    imputer = SimpleImputer(strategy='median')
    test_df[numeric_features] = imputer.fit_transform(test_df[numeric_features])

    # Standard scale the numerical variables
    scaler = StandardScaler()
    test_df[numeric_features] = scaler.fit_transform(test_df[numeric_features])

    return test_df

def preprocess_data(test_df):
    numeric_features = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'DebtRatio',
                        'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
                        'NumberRealEstateLoansOrLines', 'NumberOfDependents', 'TotalDelinquencies90DaysLate']

    # Impute missing values in the numeric features
    imputer = SimpleImputer(strategy='median')
    test_df[numeric_features] = imputer.fit_transform(test_df[numeric_features])

    # Standard scale the numerical variables
    scaler = StandardScaler()
    test_df[numeric_features] = scaler.fit_transform(test_df[numeric_features])

    return test_df

if __name__ == '__main__':
    main()
