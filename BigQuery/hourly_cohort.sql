
-- define sepsis onset event if
--  1. has suspected_infection_time from derived.suspinfect_poe
--  2. sofa score >=2 for the first time during a time window around the suspected_infection_time

-- create data with icustay with sofa >= 2
create or replace table `cdcproject.BDFH.sofa_delta` as
-- get sofa scores for the 72 time window around suspected_infection_time
with step1 as 
(
    select 
    ps.icustay_id,
    ps.hr,
    ps.starttime,
    ps.endtime,
    ps.SOFA_24hours,
    sp.suspected_infection_time
    from `physionet-data.mimiciii_derived.pivoted_sofa` ps
    inner join `physionet-data.mimiciii_derived.suspinfect_poe` sp
    using (icustay_id)
    where sp.suspected_infection_time is not null and datetime_diff(ps.starttime, sp.suspected_infection_time, hour) between -48 and 24
),
-- define sepsis onset time as the endtime when delta_score>=2
step3 as
(
    select 
    icustay_id,
    suspected_infection_time,
    endtime as sepsis_onset,
    sofa_24hours
    from step1
    where sofa_24hours >= 2
),
-- keep the first record with delta_score>=2
step4 as
(
select
*,
 ROW_NUMBER() OVER(PARTITION BY icustay_id 
                                 ORDER BY sepsis_onset ASC) AS row_id
                                 from step3
)

select * from step4 where row_id=1;

-- create data for sepsis cases
create or replace table `cdcproject.BDFH.cases` as
select sd.icustay_id
, s3c.intime
    , s3c.outtime
    , datetime_diff(s3c.outtime, s3c.intime, second)
          / 60.0 / 60.0 as length_of_stay
    , sd.sepsis_onset
    , datetime_diff(sd.sepsis_onset, s3c.intime, second)
          / 60.0 / 60.0 / 24.0 as sepsis_onset_day
    , datetime_diff(sd.sepsis_onset, s3c.intime, second)
          / 60.0 / 60.0 as sepsis_onset_hour
    , icu.SUBJECT_ID
from `cdcproject.BDFH.sofa_delta` sd
inner join `cdcproject.BDFH.Cohort` s3c
on sd.icustay_id = s3c.icustay_id
inner join `physionet-data.mimiciii_clinical.admissions` adm
on s3c.hadm_id = adm.hadm_id 
inner join `physionet-data.mimiciii_clinical.icustays` icu
on s3c.icustay_id = icu.ICUSTAY_ID
where not (
                s3c.age <= 16
                    or adm.HAS_CHARTEVENTS_DATA = 0
                    or s3c.intime is null
                    or s3c.outtime is null
                    or s3c.dbsource != 'metavision'
                or s3c.suspected_of_infection_poe = 0
                 or sd.sepsis_onset > s3c.outtime
                 or sd.sepsis_onset < s3c.intime
                );

-- create data for control
create or replace table `cdcproject.BDFH.controls` as
select s3c.icustay_id
    , s3c.hadm_id
    , s3c.intime
    , s3c.outtime
    , datetime_diff(s3c.outtime, s3c.intime, second)
          / 60.0 / 60.0 as length_of_stay
    , sd.sepsis_onset
    , icu.SUBJECT_ID
    from `cdcproject.BDFH.Cohort` s3c
    left join `cdcproject.BDFH.sofa_delta` sd 
    on s3c.icustay_id = sd.icustay_id
    inner join `physionet-data.mimiciii_clinical.admissions` adm
    on s3c.hadm_id = adm.hadm_id 
    inner join `physionet-data.mimiciii_clinical.icustays` icu
    on s3c.icustay_id = icu.ICUSTAY_ID
          -- NEW: to remove icd9 sepsis from controls! 

    where 
        s3c.hadm_id not in (
        select distinct(dg.hadm_id) 
            from `physionet-data.mimiciii_clinical.diagnoses_icd` dg 
            where dg.icd9_code in ('78552','99591','99592'))
        and not (
            s3c.age <= 16
            or adm.HAS_CHARTEVENTS_DATA = 0
            or s3c.intime is null
            or s3c.outtime is null
            or s3c.dbsource != 'metavision'
            or sd.sepsis_onset is not null
        );
