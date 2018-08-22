import pandas as pd

df = pd.read_csv('train.csv')

df.isnull().sum()

df.columns
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

# Embarked 예측 결과 S 필요.
df['Embarked'] = df['Embarked'].fillna('S')
df[df['Embarked'].isnull()]

