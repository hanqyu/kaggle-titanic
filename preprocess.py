import pandas as pd

df = pd.read_csv('train.csv')

df.isnull().sum()

df.columns
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

# Embarked 예측 결과 : S로 채우기로함
df['Embarked'] = df['Embarked'].fillna('S')
df[df['Embarked'].isnull()]

df.to_pickle("./data.pkl")


#