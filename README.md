# Neural Network Analysis for Charity Donations

## Overview

Using a deep neural net and supervised learning, attempt to predict which investments will be successful. The goal is to create a model that achieves at least 75% accuracy in its predictions.

### Software

The analysis made use of the following software:
- Python
  - math
  - Pandas
  - scikit-learn
    - model_selection
    - preprocessing
  - tensorflow
    - keras.callbacks
  - os

### Data Source

[The data used in this analysis](./Resources/charity_data.csv) was provided by Alphabet Soup’s business team and contains more than 34,000 organizations that have received funding from Alphabet Soup.

The data contained the following columns (here with their meanings included):
- `EIN`, `NAME`—Identification columns
- `APPLICATION_TYPE`—Alphabet Soup application type
- `AFFILIATION`—Affiliated industry sector
- `CLASSIFICATION`—Government organization classification
- `USE_CASE`—Use case for funding
- `ORGANIZATION`—Organization type
- `STATUS`—Active status
- `INCOME_AMT`—Income classification
- `SPECIAL_CONSIDERATIONS`—Special consideration for application
- `ASK_AMT`—Funding amount requested
- `IS_SUCCESSFUL`—Whether the money was used effectively; this is the **target column**[^target]

[^target]: _i.e._, the column whose values the machine learning models try to predict

## Analysis

Initial exploratory data analysis and preprocessing was done on the data, which was then fed into a deep neural net (hereafter referred to as `attempt 0`—further details below). The data was then processed further, and all subsequent attempts were made on the more-heavily processed data.

### Initial Preprocessing Steps

The steps taken in the initial preprocessing phase were:
- Remove the `EIN` and `NAME` columns. These are identification labels, only, and do not have predictive value.
- Collect rarely occurring values in the `APPLICATION_TYPE` column into a single `Other` category.
- Collect rarely occurring values in the `CLASSIFICATION` column into a single `Other` category.
- Perform [one-hot encoding](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f) on all columns with categorical values.
- Split the data into training and testing sets.
- Scale each column's values to have a mean of 0 and a standard deviation of 1 (using `sklearn.preprocessing`'s `StandardScaler()` function).

### Secondary Preprocessing Steps

All machine learning attempts after `attempt 0` were also subject to the following:
- Drop rows with null values.
  - There were none, so nothing was actually dropped, but the consideration _was_ made.
- Collect rarely occurring values in the `AFFILIATION` column into a single `Other` category.
- Convert `INCOME_AMT` to a numeric column.
  - Values were stored as `string`s representing numeric intervals. Each was converted to an `integer` whose values was the (approximate) midpoint of the interval (with the exception of the `50m+` category, which was given a value of `50,000,000`).
- Drop the `SPECIAL_CONSIDERATIONS` column.
  - It is a binary column in which one of the values appeared less than 0.1% of the time. It was removed primary to see if the removal showed a marked improvement in the predictions' accuracy scores.
- Drop the `STATUS` column.
  - Dropped for reasons similar to those for the `SPECIAL_CONSIDERATIONS` column (except the rare `STATUS` values are _even more rare_ than those in `SPECIAL_CONSIDERATIONS`).
- Re-scale the `ASK_AMT` according to the value's order of magnitude.
  - There was an enormous disparity between the lowest ($5,000) and highest (more than $8.5 billion) amounts requested. These values were compressed using a (base-10) logarithm, with any fractional parts dropped.

Any columns that remain (other than the target column[^target]) after preprocessing are **feature columns** for the machine learning models.

### Building the Models

Each attempt was made using a `Sequential` model (from `tensorflow.keras.models`) with the following settings:

| Attempt | Nodes per Layer[^nodes] | Layer Activation Function[^activation] |
| :-:     | :-:                     | :-:                                    |
|    0    | 80 ; 30                 | ReLu ; ReLu                            |
|    1    | 80 ; 30                 | ReLu ; ReLu                            |
|    2    | 90 ; 40                 | ReLu ; ReLu                            |
|    3    | 80 ; 30 ; 10            | ReLu ; ReLu ; ReLu                     |
|    4    | 80 ; 30                 | Tanh ; Tanh                            |
|    5    | 60 ; 30 ; 5             | ReLU ; Tanh ; Tanh                     |
|    6    | 1,000 ; 300 ; 50        | ReLU ; Tanh ; Tanh                     |

[^nodes]: Hidden layers, only. All output layers consisted of a single node with a `sigmoid` activation function.
[^activation]: `ReLU`: rectified linear unit; `Tanh`: hyperbolic tangent.

Each model was then compiled with a _binary crossentropy_ loss function, _Adam_ optimizer, and monitored _accuracy_ as its metric.

Model weight checkpoints were saved after every five passes ("epochs"). A completed model was also saved after attempts 0 and 6.

## Results

The loss and accuracy values for each attempt are shown in the following table:

| Attempt | Loss[^loss] | Accuracy[^accuracy] |
| :-:     | :-:         | :-:                 |
|    0    | 0.579       | 72.44%              |
|    1    | 0.562       | **72.64%**          |
|    2    | 0.561       | 72.48%              |
|    3    | 0.563       | 72.21%              |
|    4    | 0.560       | 72.41%              |
|    5    | **0.558**   | 72.34%              |
|    6    | 0.559       | 72.56%              |

[^loss]: Rounded to nearest 0.001; lower is better.
[^accuracy]: Rounded to nearest 0.01%; higher is better.

## Summary

Neither additional preprocessing steps, nor increasing the nodes per layer, nor increasing the number of layers, nor changing the layers' activation functions showed any marked improvement (or deterioration) in the model's accuracy as compared to the initial machine learning attempt (`attempt 0`). The additional preprocessing had the largest apparent effect, but even showed an accuracy only 0.2% more than the original.

After all attempts were completed, we noted that `INCOME_AMT` and `ASK_AMT` are on similar scales and should probably have _both_ been scaled according to their orders of magnitude (instead of just `ASK_AMT`). However, given that each numeric column was already subject to the `StandardScaler()` function at the and of the preprocessing steps, and given how little effect any other adjustment seemed to have on any of the models, we would be surprised if such scaling made any significant difference in the models' accuracies.

Given the unsuccessful results of the methods attempted, it may be worthwhile to re-attempt the experiment with one or more of the following changes:
1. Given that two of the columns were dropped for having highly imbalanced value counts, re-sample the testing set using oversampling on those features. (This may be difficult considering that there were two columns—`SPECIAL_CONSIDERATIONS` and `STATUS`—that had this problem.)
2. It is possible that a `Sequential` model is simply not the proper choice for this data. Re-attempt using other classification models (support-vector machine, random forest, _etc._).
