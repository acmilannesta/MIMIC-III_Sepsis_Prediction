WITH sepsis_diag AS (
    SELECT DISTINCT 
    SUBJECT_ID
    FROM `physionet-data.mimiciii_clinical.diagnoses_icd`
    WHERE ICD9_CODE IN ('99591', '99592', '78552')
),

sepsis_pat AS(
    SELECT DISTINCT 
    a.SUBJECT_ID,
    DOB,
    DOD,
    gender
    FROM `physionet-data.mimiciii_clinical.patients` a
    INNER JOIN sepsis_diag b
    ON a.SUBJECT_ID = b.SUBJECT_ID
),

los AS (
    SELECT DISTINCT 
    SUBJECT_ID,
    SUM(DATE_DIFF(DISCHTIME, ADMITTIME, DAY)) OVER(PARTITION BY SUBJECT_ID) AS LOS
    FROM `physionet-data.mimiciii_clinical.admissions`
),

first_admission AS (
    SELECT DISTINCT 
    SUBJECT_ID,
    ETHNICITY,
    MARITAL_STATUS,
    INSURANCE,
    MIN(ADMITTIME) OVER(PARTITION BY SUBJECT_ID) AS admit_time
    FROM `physionet-data.mimiciii_clinical.admissions`
),

los_demo AS (
    SELECT 
    a.*,        
    LOS
    FROM first_admission  a 
    INNER JOIN los b 
    ON a.SUBJECT_ID = b.SUBJECT_ID
),

sepsis_pat_demo AS (
    SELECT DISTINCT
    a.subject_id,
    gender,
    dob,
    dod,
    ETHNICITY,
    MARITAL_STATUS,
    INSURANCE,
    los,
    admit_time
    FROM sepsis_pat a 
    INNER JOIN los_demo b
    ON a.SUBJECT_ID = b.SUBJECT_ID
)

SELECT *,
DATE_DIFF(admit_time , dob, YEAR) AS age_admit,
DATE_DIFF(dod, dob, YEAR) AS age_death
FROM sepsis_pat_demo 

