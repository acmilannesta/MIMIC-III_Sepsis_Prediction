# CSE-6250-Group-Project

## Data

Run sql and python code in following orders to recreate the csv files used in modeling. **Notice running the 48h series sql with control lab and vital will cost 1GB and 10GB usage of bigquery quota.**

```
Cohort.sql
hourly_cohort.sql
match-control.py
static-query.sql
extract-48h-of-hourly-case-lab-series.sql
extract-48h-of-hourly-case-vital-series.sql
extract-48h-of-hourly-control-lab-series.sql
extract-48h-of-hourly-control-vital-series.sql
data_prep_step1.py
```

For all sql codes, change project and data names to your own location accordingly. The two python files can be run as follows assuming you are in the top level folder
```
python Python/match-control.py -c <bigquery credential json filename> -t <bigquery table reference>
python Python/data_prep_step1.py -c <bigquery credential json filename> -t <bigquery table reference> -w <prediction window hours(default is 3)>
```
and an example would be
```
python Python/match-control.py -c bdfh.json -t cdcproject.BDFH
python Python/data_prep_step1.py -c bdfh.json -t cdcproject.BDFH -w 3
```
The data will be saved in the Data folder. 

## Models

All models are in development. For the latest versions, check the individual branches if not found under the main branch:

* [Logistic regression, SVM](https://github.gatech.edu/amalhotra60/CSE-6250-Group-Project/blob/main/Lightgbm__LR_SVM_Model.ipynb)
* [LightGBM](https://github.gatech.edu/amalhotra60/CSE-6250-Group-Project/blob/main/Lightgbm_Model.ipynb)
* [RNN](https://github.gatech.edu/amalhotra60/CSE-6250-Group-Project/blob/main/RNN_Model.ipynb)

## Resource

* Data pipeline: Detailed data filtering and label definition: Early Recognition of Sepsis with Gaussian Process Temporal
Convolutional Networks and Dynamic Time Warping, [paper](http://proceedings.mlr.press/v106/moor19a/moor19a.pdf), [code](https://github.com/BorgwardtLab/mgp-tcn/tree/master/src/query)
* Reference library: https://www.zotero.org/groups/4456592/bdfh_project/items/7JQNQF39/library