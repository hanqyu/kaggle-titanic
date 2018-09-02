
########### Embarked 예측 ###########
embarked_predict_model = tc.classifier.create(train_sf[train_sf['Embarked'] != ''].dropna(), target='Embarked')
embarked_predict_model.predict(train_sf[train_sf['Embarked'] == ''])
['S', 'S']