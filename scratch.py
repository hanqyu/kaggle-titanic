import turicreate as tc
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
train.head()

train.columns
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')

test = pd.read_csv('test.csv')

train_sf = tc.SFrame().read_csv('train.csv')

train_sf.show()
test_sf = tc.SFrame().read_csv('test.csv')
model = tc.classifier.create(train_sf, target='Survived', features=['Pclass', 'Sex', 'SibSp',
       'Parch', 'Fare', 'Embarked'])
'''
PROGRESS: Model selection based on validation accuracy:
PROGRESS: ---------------------------------------------
PROGRESS: BoostedTreesClassifier          : 0.804347813129425
PROGRESS: RandomForestClassifier          : 0.782608687877655
PROGRESS: DecisionTreeClassifier          : 0.8260869383811951
PROGRESS: SVMClassifier                   : 0.804348
PROGRESS: LogisticClassifier              : 0.804348
PROGRESS: ---------------------------------------------
PROGRESS: Selecting DecisionTreeClassifier based on validation set performance.
'''

model = tc.decision_tree_classifier.create(train_sf, target='Survived', features=['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked'])
train_sf[train_sf['Age'] is None]