from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics # accuracy measure


train_sk, test_sk = train_test_split(df, test_size=0.3,random_state=0)
target_col = ['Pclass', 'Sex', 'Embarked']

train_sk_X=train_sk[target_col]
train_sk_Y=train_sk['Survived']
test_sk_X=test_sk[target_col]
test_sk_Y=test_sk['Survived']

features_one = train_sk_X.values
target = train_sk_Y.values

tree_model = DecisionTreeClassifier()
tree_model.fit(features_one, target)
dt_prediction = tree_model.predict(test_sk_X)

print('The accuracy of the Decision Tree is',metrics.accuracy_score(dt_prediction, test_Y))



/// 별로인듯


import turicreate as tc

df.to_csv('temp.csv')
sf = tc.SFrame('temp.csv')

train, test = sf.random_split(0.8)
predict_model = tc.decision_tree_classifier.create(train, target='Survived', features=['Pclass','Sex','Embarked'])
(predict_model.predict(test) == test['Survived']).sum()

result = tc.SFrame('test.csv')
result['Survived'] = predict_model.predict(result)
result[['PassengerId','Survived']].save('final.csv','csv')