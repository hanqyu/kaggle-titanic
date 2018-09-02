import pandas as pd

df = pd.read_csv('train.csv')

df.isnull().sum()

df.columns
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'AgeCategorized'],
      dtype='object')

# Embarked 예측 결과 : S로 채우기로함
df['Embarked'] = df['Embarked'].fillna('S')
df[df['Embarked'].isnull()]

df.to_pickle("./data.pkl")


# Age 간단히 10의 자리로 -> AgeCategorized
df['AgeCategorized'] = df['Age'].dropna().apply(lambda x: int(round(x/10))*10).astype(dtype='category')

df.to_pickle('./age_categorized.pkl')
df = pd.read_pickle('./age_categorized.pkl')

# Age float to int
