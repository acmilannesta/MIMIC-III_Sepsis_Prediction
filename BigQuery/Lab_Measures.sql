    
WITH 
    --WBC Count
    WBC AS (
    select distinct
    SUBJECT_ID,
    AVG(VALUENUM) AS WBC_avg,
    MAX(VALUENUM) AS WBC_max,
    MIN(VALUENUM) AS WBC_min,
    CASE
        WHEN STDDEV(VALUENUM) IS NOT NULL THEN STDDEV(VALUENUM)
        ELSE 0
    END AS WBC_stddev
    from `physionet-data.mimiciii_clinical.labevents`
    where ITEMID IN (
        select ITEMID
        from `physionet-data.mimiciii_clinical.d_labitems`
        where LOWER(LABEL) LIKE '%wbc%') AND VALUENUM IS NOT NULL
    GROUP BY SUBJECT_ID),

    --Creatinine
    creatinine AS (
    select distinct
    SUBJECT_ID,
    AVG(VALUENUM) AS creatinine_avg,
    MAX(VALUENUM) AS creatinine_max,
    MIN(VALUENUM) AS creatinine_min,
    CASE
        WHEN STDDEV(VALUENUM) IS NOT NULL THEN STDDEV(VALUENUM)
        ELSE 0
    END AS creatinine_stddev
    from `physionet-data.mimiciii_clinical.labevents`
    where ITEMID IN (
        select ITEMID
        from `physionet-data.mimiciii_clinical.d_labitems`
        where LOWER(LABEL) LIKE '%creatinine%') AND VALUENUM IS NOT NULL
    GROUP BY SUBJECT_ID
    ),

    --Anion gap
    anion_gap AS (
    select distinct
    SUBJECT_ID,
    AVG(VALUENUM) AS anion_gap_avg,
    MAX(VALUENUM) AS anion_gap_max,
    MIN(VALUENUM) AS anion_gap_min,
    CASE
        WHEN STDDEV(VALUENUM) IS NOT NULL THEN STDDEV(VALUENUM)
        ELSE 0
    END AS anion_gap_stddev
    from `physionet-data.mimiciii_clinical.labevents`
    where ITEMID IN (
        (select ITEMID
        from `physionet-data.mimiciii_clinical.d_labitems`
        where LABEL IN ('Anion Gap'))) AND VALUE IS NOT NULL
    GROUP BY SUBJECT_ID
    ),

    --sodium
    sodium AS (
    select distinct
    SUBJECT_ID,
    AVG(VALUENUM) AS sodium_avg,
    MAX(VALUENUM) AS sodium_max,
    MIN(VALUENUM) AS sodium_min,
    CASE
        WHEN STDDEV(VALUENUM) IS NOT NULL THEN STDDEV(VALUENUM)
        ELSE 0
    END AS sodium_stddev
    from `physionet-data.mimiciii_clinical.labevents`
    where ITEMID IN (
        select ITEMID
        from `physionet-data.mimiciii_clinical.d_labitems`
        where LOWER(LABEL) LIKE '%sodium%') AND VALUENUM IS NOT NULL
    GROUP BY SUBJECT_ID
    ),

    --hemoglobin
    hemoglobin AS (
    select distinct
    SUBJECT_ID,
    AVG(VALUENUM) AS hemoglobin_avg,
    MAX(VALUENUM) AS hemoglobin_max,
    MIN(VALUENUM) AS hemoglobin_min,
    CASE
        WHEN STDDEV(VALUENUM) IS NOT NULL THEN STDDEV(VALUENUM)
        ELSE 0
    END AS hemoglobin_stddev
    from `physionet-data.mimiciii_clinical.labevents`
    where ITEMID IN (
        select ITEMID
        from `physionet-data.mimiciii_clinical.d_labitems`
        where LOWER(LABEL) LIKE '%hemoglobin%') AND VALUENUM IS NOT NULL
    GROUP BY SUBJECT_ID
    ),

    --lactate
    lactate AS (
    select distinct
    SUBJECT_ID,
    AVG(VALUENUM) AS lactate_avg,
    MAX(VALUENUM) AS lactate_max,
    MIN(VALUENUM) AS lactate_min,
    CASE
        WHEN STDDEV(VALUENUM) IS NOT NULL THEN STDDEV(VALUENUM)
        ELSE 0
    END AS lactate_stddev
    from `physionet-data.mimiciii_clinical.labevents`
    where ITEMID IN (
        select ITEMID
        from `physionet-data.mimiciii_clinical.d_labitems`
        where LOWER(LABEL) LIKE '%lactate%') AND VALUENUM IS NOT NULL
    GROUP BY SUBJECT_ID
    ),

    --potassium
    potassium AS (
    select distinct
    SUBJECT_ID,
    AVG(VALUENUM) AS potassium_avg,
    MAX(VALUENUM) AS potassium_max,
    MIN(VALUENUM) AS potassium_min,
    CASE
        WHEN STDDEV(VALUENUM) IS NOT NULL THEN STDDEV(VALUENUM)
        ELSE 0
    END AS potassium_stddev
    from `physionet-data.mimiciii_clinical.labevents`
    where ITEMID IN (
        select ITEMID
        from `physionet-data.mimiciii_clinical.d_labitems`
        where LOWER(LABEL) LIKE '%potassium%') AND VALUENUM IS NOT NULL
    GROUP BY SUBJECT_ID
    ),

    --Urea Nitrogen
    urea_nitrogen AS (
    select distinct
    SUBJECT_ID,
    AVG(VALUENUM) AS urea_nitrogen_avg,
    MAX(VALUENUM) AS urea_nitrogen_max,
    MIN(VALUENUM) AS urea_nitrogen_min,
    CASE
        WHEN STDDEV(VALUENUM) IS NOT NULL THEN STDDEV(VALUENUM)
        ELSE 0
    END AS urea_nitrogen_stddev
    from `physionet-data.mimiciii_clinical.labevents`
    where ITEMID IN (
        select ITEMID
        from `physionet-data.mimiciii_clinical.d_labitems`
        where LOWER(LABEL) LIKE '%urea nitrogen%') AND VALUENUM IS NOT NULL
    GROUP BY SUBJECT_ID
    ),

    --Glucose
    glucose AS(
    select distinct
    SUBJECT_ID,
    AVG(VALUENUM) AS glucose_avg,
    MAX(VALUENUM) AS glucose_max,
    MIN(VALUENUM) AS glucose_min,
    CASE
        WHEN STDDEV(VALUENUM) IS NOT NULL THEN STDDEV(VALUENUM)
        ELSE 0
    END AS glucose_stddev
    from `physionet-data.mimiciii_clinical.labevents`
    where ITEMID IN (
        select ITEMID
        from `physionet-data.mimiciii_clinical.d_labitems`
        where LOWER(LABEL) LIKE '%glucose%') AND VALUENUM IS NOT NULL
    GROUP BY SUBJECT_ID
    )

SELECT *
FROM WBC a
FULL OUTER JOIN creatinine b
USING(SUBJECT_ID)
FULL OUTER JOIN anion_gap c
USING(SUBJECT_ID)
FULL OUTER JOIN sodium d
USING(SUBJECT_ID)
FULL OUTER JOIN hemoglobin e
USING(SUBJECT_ID)
FULL OUTER JOIN lactate f
USING(SUBJECT_ID)
FULL OUTER JOIN potassium g
USING(SUBJECT_ID)
FULL OUTER JOIN urea_nitrogen h
USING(SUBJECT_ID)
FULL OUTER JOIN glucose i
USING(SUBJECT_ID)



