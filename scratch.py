import turicreate as tc

'''
train.columns
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')
'''

train_sf = tc.SFrame().read_csv('train.csv')
train, test = tc.SFrame().read_csv('train.csv').random_split(0.8)


# train['Pclass_str'] = train.apply(lambda x: str(x['Pclass'])) # 별로 효과 없더라

# 일단 classifier를 돌려봄.
model = tc.classifier.create(train, target='Survived', features=['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked'])

# BoostedTreesClassifier로 작업해보기로
model = tc.random_forest_classifier.create(train, target='Survived', features=['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked'], max_iterations=11)

#성능을 향상시키려면 feature 컬럼을 더 추가하고 max-depth 등을 만져야 할 듯


# 부모자식의 최빈값 3순위 0,1,2의 나이 평균 보기
for i in [0,1,2]: print(i, non_empty_ages[non_empty_ages['Parch'] == i]['Age'].mean())

'''
########### age 빈 컬럼을 예측값으로 채워보자 ###########
non_empty_ages = train_sf[train_sf['Age'] != None]
empty_ages = train_sf[train_sf['Age'] == None]

non_empty_ages['AgeInt'] = non_empty_ages.apply(lambda x: int(x['Age']))
train, test = non_empty_ages.random_split(0.8)

age_predict_model = tc.classifier.create(train, target='AgeInt', features=['Pclass', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'])
age_predict_model = tc.boosted_trees_regression.create(train, max_depth=20, target='AgeInt', features=['Pclass', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'])

empty_ages['AgeInt'] = age_predict_model.predict(empty_ages).apply(lambda x: round(x))
empty_ages[['PassengerId', 'AgeInt']]
empty_ages.show()
age_predict_model.predict(test) == test['AgeInt'].apply(lambda x: round(x)
'''

# age problem
1. age가 정규분포를 따른다고 가정하고 나머지 값을 채워본다
2. age를 예측한다. 그냥 예측하면 힘드니까 범주형변수로 바꾼뒤 채운다


#data insight
pd.crosstab(df['AgeCategorized'],df['Survived']).apply(lambda x: round(x/x.sum(),3))


########### 결측값 Age & Cabin ###########
import pandas as pd
df = pd.read_pickle('./age_categorized.pkl')
df.head()
df.columns

df['Cabin'].isnull().sum()


AgeCategorized

df.to_csv('temp.csv')
sf = tc.SFrame('temp.csv')
sf['AgeCategorized'] = sf['AgeCategorized'].apply(lambda x: int(x))
# sf = sf.remove_column('Age')

train, test = sf.dropna_split(columns='AgeCategorized')

# columns = sf.column_names()
# columns.pop(columns.index('AgeCategorized'))
age_predict_model = tc.classifier.create(train, target='AgeCategorized', features=['Parch','SibSp'])

age_predict_model
0.53 이하.. 쓸모가 없는건가

########### Name 활용하기 ###########
# FamilyName
sf['FamilyName'] = sf['Name'].apply(lambda x: x.split(',')[0])

sf['FamilyName'].show()
sf[sf['FamilyName'].apply(lambda x: x in ['Andersson','Sage','Skoog','Panula','Johnson','Goodwin','Carter'])].sort(['FamilyName', 'Age'], ascending=False).explore()

# 결혼 여부
sf['Name'].apply(lambda x: x.split(', ')[1].split('.')[0] in ['Mr', 'Mrs']).sum() # 642
sf['Name'].apply(lambda x: x.split(', ')[1].split('.')[0] == 'Ms').sum() # 1
sf['Name'].apply(lambda x: x.split(', ')[1].split('.')[0] in ['Master', 'Miss']).sum() # 222
sf[sf['Name'].apply(
    lambda x: x.split(', ')[1].split('.')[0] not in
              ['Mr', 'Mrs', 'Ms', 'Master', 'Miss'])].explore()
'''
sf['Family'] = sf.apply(lambda x: \
    if x['Parch'] == 0 & x['SibSp'] == 0:
        'No Family'
    elif )
'''

sf['NameTitle'] = sf['Name'].apply(lambda x: x.split(', ')[1].split('.')[0])

target = sf[sf['AgeCategorized'] is None]
train, valid = sf.dropna('AgeCategorized')[sf.dropna('AgeCategorized').apply(lambda x: x['NameTitle'] in ['Mr', 'Mrs', 'Ms', 'Master', 'Miss'])].random_split(0.8)
tc.classifier.create(train, target='AgeCategorized', features=['NameTitle', 'Parch', 'SibSp'])
model = tc.random_forest_classifier.create(train, target='AgeCategorized', features=['NameTitle','Parch','SibSp'])
(valid['AgeCategorized'] == model.classify(valid)['class']).sum()


### feature와 survived간의 상관관계 찾기
import matplotlib
import matplotlib.pyplot as plt

survived = df[df['Survived']==1]
dead = df[df['Survived']==0]

survived.plot.scatter(x='Pclass',y='Survived')
plt.show()




### classifer loop 돌려보기
sf = tc.SFrame.read_csv('train.csv')
train, val = sf.random_split


