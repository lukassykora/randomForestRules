# Random Forest Rules
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Get the representation of all rules found by sklearn RandomForestClassifier. It works in following way:

- On each feature, it applies one-hot encoding that makes each column binary.
- Random Forest runs on the features and a target attribute.
- All trees are extracted from the Random Forest Regressor.
- Decision Trees are split to classification rules.


## GIT repository

https://github.com/lukassykora/randomForestRules

## Installation

pip install randomForestRules-lukassykora

## Jupyter Notebook

- [Audiology](https://github.com/lukassykora/randomForestRules/blob/master/notebooks/AudiologyRandomForest.ipynb) 

## Example
```python
from randomForestRules import RandomForestRules
import pandas as pd

df = pd.read_csv("data/audiology.csv")
df.columns = df.columns.str.replace("_", "-") # underscore not allowed
# All feature columns
cols=[]
for col in df.columns:
    if col != 'binaryClass':
        cols.append(col)
# Initialize
randomForest = RandomForestRules()
# Load data
randomForest.load_pandas(df)
# Fit
randomForest.fit(antecedent = cols, consequent = 'binaryClass', supp=0.005, conf=50)
# Get result
frame = randomForest.get_frame()
```
