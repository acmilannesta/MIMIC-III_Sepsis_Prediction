import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def race_recode(df):
    cond_white = df['ethnicity'].str.contains('WHITE')
    cond_black = df['ethnicity'].str.contains('BLACK')
    cond_asian = df['ethnicity'].str.contains('ASIAN')
    cond_hispa = df['ethnicity'].str.contains('HISPANIC')

    df.loc[cond_white, 'ethnicity'] = 'WHITE'
    df.loc[cond_black, 'ethnicity'] = 'BLACK'
    df.loc[cond_asian, 'ethnicity'] = 'ASIAN'
    df.loc[cond_hispa, 'ethnicity'] = 'HISPANIC'
    df.loc[~(cond_white | cond_black | cond_asian | cond_hispa), 'ethnicity'] = 'OTHER'
    
    df['ethnicity'] = df['ethnicity'].apply(lambda x: x[0] + x[1:].lower())
    return df

def model_data(model_type, path='Data', test_size=.1):

    case_labs = pd.read_csv(f"{path}/case_48h_labs_ex3h.csv")
    case_vitals = pd.read_csv(f"{path}/case_48h_vitals_ex3h.csv")
    case_static = pd.read_csv(f"{path}/static_variables_cases_ex3h.csv")

    control_labs = pd.read_csv(f"{path}/control_48h_labs_ex3h.csv")
    control_vitals = pd.read_csv(f"{path}/control_48h_vitals_ex3h.csv")
    control_static = pd.read_csv(f"{path}/static_variables_controls_ex3h.csv")

    # if path == 'data_v2':
    #     case_labs.drop(columns=['Unnamed: 0'], inplace=True)
    #     case_vitals.drop(columns=['Unnamed: 0'], inplace=True)
    #     case_static.drop(columns=['Unnamed: 0'], inplace=True)
    #     control_labs.drop(columns=['Unnamed: 0'], inplace=True)
    #     control_vitals.drop(columns=['Unnamed: 0'], inplace=True)
    #     control_static.drop(columns=['Unnamed: 0'], inplace=True)

    if model_type in ("LGBM", "SVM", "LR"):
        # case_labs = case_labs.drop(columns=['chart_time', 'subject_id', 'sepsis_onset', 'hr_feature'])
        # apply_dict = {col: ['mean', 'median', 'std',  'min', 'max'] for col in case_labs.columns if col != 'icustay_id'}
        # case_labs = case_labs.groupby('icustay_id').agg(apply_dict)
        # case_labs.columns = ['_'.join(col) for col in case_labs.columns]

        case_static = case_static[['gender', 'ethnicity', 'admission_age', 'icustay_id']]
        case_static['label'] = 1
        case_static = race_recode(case_static)

        # case_static['ethnicity'] = LabelEncoder().fit_transform(case_static['ethnicity'])
        case_static = pd.concat((case_static.drop(columns=['ethnicity']), pd.get_dummies(case_static['ethnicity'])), axis = 1)
        case_static['gender'] = LabelEncoder().fit_transform(case_static['gender'])

        # case_vitals = case_vitals.drop(columns=['chart_time', 'subject_id', 'sepsis_onset', 'hr_feature'])
        # apply_dict = {col: ['mean', 'median', 'std', 'min', 'max'] for col in case_vitals.columns if col != 'icustay_id'}
        # case_vitals = case_vitals.groupby('icustay_id').agg(apply_dict)
        # case_vitals.columns = ['_'.join(col) for col in case_vitals.columns]

        case_all = case_static#.merge(case_labs, on='icustay_id').merge(case_vitals, on='icustay_id')


        # control_labs = control_labs.drop(columns=['chart_time', 'subject_id', 'control_onset_time', 'hr_feature'])
        # apply_dict = {col: ['mean', 'median', 'std', 'min', 'max'] for col in control_labs.columns if col != 'icustay_id'}
        # control_labs = control_labs.groupby('icustay_id').agg(apply_dict)
        # control_labs.columns = ['_'.join(col) for col in control_labs.columns]

        control_static = control_static[['gender', 'ethnicity', 'admission_age', 'icustay_id']]
        control_static['label'] = 0
        control_static = race_recode(control_static)
        # control_static['ethnicity'] = LabelEncoder().fit_transform(control_static['ethnicity'])

        control_static = pd.concat((control_static.drop(columns=['ethnicity']), pd.get_dummies(control_static['ethnicity'])), axis = 1)
        control_static['gender'] = LabelEncoder().fit_transform(control_static['gender'])

        # control_vitals = control_vitals.drop(columns=['chart_time', 'subject_id', 'control_onset_time', 'hr_feature'])
        # apply_dict = {col: ['mean', 'median', 'std', 'min', 'max'] for col in control_vitals.columns if col != 'icustay_id'}
        # control_vitals = control_vitals.groupby('icustay_id').agg(apply_dict)
        # control_vitals.columns = ['_'.join(col) for col in control_vitals.columns]

        control_all = control_static#.merge(control_labs, on='icustay_id').merge(control_vitals, on='icustay_id')

        df_all = pd.concat([case_all, control_all], ignore_index=True).sort_values('icustay_id')

        df_train, df_test = train_test_split(df_all, test_size=test_size, random_state=42, stratify=df_all["label"])
        print(f"Train/Test data size: {df_train.shape[0]}/{df_test.shape[0]}")
        return df_train, df_test

    elif model_type == "RNN":
        case_static = case_static.drop(columns=['subject_id'])
        case_static['label'] = 1
        case_labs = case_labs.drop(columns=['subject_id', 'chart_time', 'sepsis_onset'])
        case_labs['hr_feature'] = 45-np.ceil(case_labs['hr_feature'])
        case_labs = case_labs.groupby(['icustay_id', 'hr_feature'], as_index=False).mean()
        case_vitals = case_vitals.drop(columns=['subject_id'])
        case_vitals['hr_feature'] = 45-np.ceil(case_vitals['hr_feature'])
        case_vitals = case_vitals.groupby(['icustay_id', 'hr_feature'], as_index=False).mean()
        case_labs_vital = case_labs.merge(case_vitals, on=['icustay_id', 'hr_feature'], how='outer')

        control_static = control_static.drop(columns=['subject_id']) \
            .rename(columns={'control_onset_time': 'sepsis_onset', 'control_onset_hour': 'sepsis_onset_hour'})
        control_static['label'] = 0
        control_labs = control_labs.drop(columns=['subject_id', 'chart_time', 'control_onset_time'])
        control_labs['hr_feature'] = 45-np.ceil(control_labs['hr_feature'])
        control_labs = control_labs.groupby(['icustay_id', 'hr_feature'], as_index=False).mean()
        control_vitals = control_vitals.drop(columns=['subject_id'])
        control_vitals['hr_feature'] = 45-np.ceil(control_vitals['hr_feature'])
        control_vitals = control_vitals.groupby(['icustay_id', 'hr_feature'], as_index=False).mean()
        control_labs_vital = control_labs.merge(control_vitals, on=['icustay_id', 'hr_feature'], how='outer')

        sequence_all = pd.concat([case_labs_vital, control_labs_vital]).set_index(['icustay_id', 'hr_feature']).sort_index()
        sequence_all = sequence_all.groupby(level=0).ffill().fillna(0)
        sequence_all = sequence_all.reset_index(level=1)
        
        case_static = case_static.merge(case_labs[['icustay_id']].drop_duplicates('icustay_id'), on='icustay_id').merge(case_vitals[['icustay_id']].drop_duplicates('icustay_id'), on='icustay_id')
        control_static = control_static.merge(control_labs[['icustay_id']].drop_duplicates('icustay_id'), on='icustay_id').merge(control_vitals[['icustay_id']].drop_duplicates('icustay_id'), on='icustay_id')
        static_all = pd.concat([case_static, control_static]).set_index('icustay_id')
        static_all = static_all.join(sequence_all.groupby(sequence_all.index).first()[[]], how="inner")
        static_all = static_all.sort_index()
        static_train, static_test = train_test_split(static_all, test_size=test_size, random_state=42, stratify=static_all["label"])
        sequence_train = sequence_all.join(static_train[[]], how="inner")
        sequence_test = sequence_all.join(static_test[[]], how="inner")

        print(f"Train/Test data size: {static_train.shape[0]}/{static_test.shape[0]}")
        return static_train, static_test, sequence_train, sequence_test

    else:
        raise ValueError(f"Model {model_type} not supported")