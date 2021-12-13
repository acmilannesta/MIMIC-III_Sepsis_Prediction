SELECT ie.subject_id, ie.hadm_id, ie.icustay_id

-- patient level factors
, pat.gender

-- hospital level factors
, adm.admittime
, ROUND( (CAST(DATETIME_DIFF(adm.admittime, pat.dob, SECOND)/(60*60*24*365.242) AS numeric)), 4) AS admission_age
, adm.ethnicity, adm.admission_type
, adm.admission_location
, ie.first_careunit

, DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) AS hospstay_seq
, CASE
    WHEN DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) = 1 THEN 1
    ELSE 0 END AS first_hosp_stay

-- icu level factors
, ie.intime
, ie.outtime
, DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) AS icustay_seq

-- first ICU stay *for the current hospitalization*
, CASE
    WHEN DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) = 1 THEN 1
    ELSE 0 END AS first_icu_stay

FROM `physionet-data.mimiciii_clinical.icustays` ie
INNER JOIN `physionet-data.mimiciii_clinical.admissions` adm
    ON ie.hadm_id = adm.hadm_id

INNER JOIN `physionet-data.mimiciii_clinical.patients` pat
    ON ie.subject_id = pat.subject_id    
WHERE adm.has_chartevents_data = 1
ORDER BY ie.subject_id, adm.admittime, ie.intime;