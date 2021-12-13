'''
Summary: Script to match controls to cases based on icustay_ids and case/control ratio.
NOTE (11/1/21): Original input/output steps below have been modified to use BigQuery instead of local CSV files

Input: it takes a
    - cases csv, that links case icustay_id to sepsis_onset_hour (relative time after icu-intime in hours)
    - control csv listing control icustay_ids and corresponding icu-intime
Output:
    - matched_controls.csv that list controls the following way:
        icustay_id, control_onset_time, control_onset_hour, matched_case_icustay_id  
Detailed Description:
    1. Load Input files
    2. Determine Control vs Case ratio p (e.g. 10/1)
    3. Loop:
        For each case:
            randomly select (without repe) p controls as matched_controls
            For each selected control:
                append to result df: icustay_id, control_onset_time, control_onset_hour, matched_case_icustay_id
                (icustay_id of this control, the cases sepsis_onset_hour as control_onset_hour and the absolute time as control_onset_time, and the matched_case_icustay_id)
    4. return result df as output
'''
import pandas as pd
import numpy as np
import os
import argparse
from google.cloud import bigquery

def get_matched_controls():
    result = pd.DataFrame()
    #--------------------
    # 1. Load Input data
    # This step has been modified from https://github.com/BorgwardtLab/mgp-tcn
    #--------------------
    cases = bqclient.query(f"select * from {TABLE_lOC}.cases").result().to_dataframe()
    controls = bqclient.query(f"select * from {TABLE_lOC}.controls").result().to_dataframe()

    controls = controls.drop_duplicates() # drop duplicate rows (NOTE: it seems like there are no duplicates) 
    controls = controls.reset_index(drop=True) # resetting row index for aesthetic reasons

    case_ids = cases['icustay_id'].unique() # get unique ids
    control_ids = controls['icustay_id'].unique()

    #--------------------------------
    # 2. Determine Control/Case Ratio
    #--------------------------------
    ratio = len(control_ids)/float(len(case_ids))

    #---------------------------------------------
    # 3. For each case match 'ratio-many' controls 
    #---------------------------------------------

    # random matching without conditions
    controls_s = controls.iloc[np.random.permutation(len(controls))] # Shuffle controls dataframe rows, for random control selection

    for i, case_id in enumerate(case_ids):
        matched_controls = controls_s[int(i*ratio):int(ratio*(i+1))].copy() # select the next batch of controls to match to current case
        onset_hour = float(cases[cases['icustay_id']==case_id]['sepsis_onset_hour']) # get float of current case onset hour
        matched_controls['control_onset_hour'] = onset_hour # use sepsis_onset_hour of current case as control_onset_hour
        matched_controls['control_onset_time'] = matched_controls['intime'] + pd.Timedelta(hours=onset_hour) # compute control_onset time w.r.t. control icu-intime
        matched_controls['matched_case_icustay_id'] = case_id # so that each matched control can be mapped back to its matched case
        result = result.append(matched_controls, ignore_index=True)
        
    # Sanity Check:
    if abs(len(result) - int(ratio*(i+1)))>1:
        raise ValueError('Resulting matched_controls dataframe not as long as ratio * #cases!')    

    # drop controls with onset later than discharge
    result = result[result.control_onset_hour < result.length_of_stay]

    print('Number of Cases: {}'.format(len(case_ids)))
    print('Number of Controls: {}'.format(len(control_ids)))
    print('Matching Ratio: {}'.format(ratio))
    print('Matched controls: {}'.format(result.shape[0]))
    #---------------------------------------------------------------------
    # 4. Return matched controls for next step (load BigQuery table)
    #---------------------------------------------------------------------
    return result

def load_matched_controls(df):
    table_id = f"{TABLE_lOC}.matched_controls_hourly"
    print("Delete table: {} if exists...".format(table_id))
    bqclient.delete_table(table_id, not_found_ok=True)

    # Reference: 
    # https://googleapis.dev/python/bigquery/latest/usage/pandas.html#load-a-pandas-dataframe-to-a-bigquery-table
    # https://cloud.google.com/bigquery/docs/tables 
    # https://github.com/googleapis/python-bigquery/issues/56
    print("Loading table: {}...".format(table_id))

    schema = [
        bigquery.SchemaField("icustay_id", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("hadm_id", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("intime", "DATETIME", mode="NULLABLE"),
        bigquery.SchemaField("outtime", "DATETIME", mode="NULLABLE"),
        bigquery.SchemaField("length_of_stay", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("SUBJECT_ID", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("control_onset_time", "DATETIME", mode="NULLABLE"),
        bigquery.SchemaField("control_onset_hour", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("matched_case_icustay_id", "INTEGER", mode="NULLABLE"),
    ]

    table = bigquery.Table(table_id, schema=schema)
    bqclient.create_table(table)  # Make an API request.
    job = bqclient.load_table_from_dataframe(df, table)
    
    job.result()  # Wait for the job to complete.

    table = bqclient.get_table(table_id)  
    print(
        "Loaded {} rows and {} columns to {}".format(
            table.num_rows, len(table.schema), table_id
        )
    )

def main():
    cc_matches = get_matched_controls()
    load_matched_controls(cc_matches)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate control_onset_time")
    parser.add_argument("-c", "--credential", required=True,
                        help="Google credential json file name under current working directory")
    parser.add_argument("-t", "--tableloc", required=True, 
                        help="Project and dataset name, e.g. cdc.project")
    args = parser.parse_args()
    # GCP credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getcwd(), args.credential)
    bqclient = bigquery.Client()
    
    np.random.seed(1)

    TABLE_lOC = args.tableloc
    main()
