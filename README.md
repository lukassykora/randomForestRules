# Random Forest Rules
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Get the representation of all rules found by sklearn RandomForestClassifier. It works in following way:

- On each feature, it applies one-hot encoding that makes each column binary.
- Random Forest runs on the features and a target attribute.
- All trees are extracted from the Random Forest Regressor.
- Decision Trees are split to classification rules.


## GIT repository

https://github.com/lukassykora/randomForestRules

## Example
```python
from randomForest import RandomForest
import pandas as pd

df = pd.read_csv("data/audiology.csv")
df.columns = df.columns.str.replace("_", "-") # underscore not allowed
df['target'] = df['binaryClass'].apply(lambda x: 1 if x == "P" else 0) # target musts be numerical
# All feature columns
cols=[]
for col in df.columns:
    if col != 'binaryClass' and col != 'target':
        cols.append(col)
# Initialize
randomForest = RandomForest()
# Load data
randomForest.load_pandas(df)
# Fit
randomForest.fit(antecedents = cols, consequent = 'target', supp=0.005, conf=50)
# Get result
frame = randomForest.get_frame()
```
