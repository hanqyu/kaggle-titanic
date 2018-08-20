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

# age 빈 컬럼을 예측값으로 채워보자
non_empty_ages = train_sf[train_sf['Age'] != None]
empty_ages = train_sf[train_sf['Age'] == None]

non_empty_ages['AgeInt'] = non_empty_ages.apply(lambda x: int(x['Age']))
train, test = non_empty_ages.random_split(0.8)

age_predict_model = tc.boosted_trees_regression.create(non_empty_ages, max_depth=20, target='AgeInt', max_iterations=1000)#, features=['Pclass', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'])
age_predict_model.save('age_prediction')

# results = age_predict_model.evaluate(test)
empty_ages['AgeInt'] = age_predict_model.predict(empty_ages).apply(lambda x: round(x))
empty_ages[['PassengerId', 'AgeInt']]

age_predict_model.predict(test)
empty_ages.show()