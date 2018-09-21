#!/usr/bin/python
import random
import numpy as np
import matplotlib.pyplot
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list =  ['poi','salary','to_messages', 'deferral_payments','total_payments','exercised_stock_options','restricted_stock', 'bonus', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options','restricted_stock_deferred', 'loan_advances','other','director_fees','shared_receipt_with_poi','from_messages','long_term_incentive','from_poi_to_this_person'] # You will need to use more features
#features_list =  ['poi','to_messages', 'loan_advances','other','director_fees','shared_receipt_with_poi','from_poi_to_this_person'] # You will need to use more features
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#scale feature
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
#select k best features
#from sklearn.feature_selection import SelectKBest, f_classif
#kbest = SelectKBest(f_classif,k=5)
#kbest=kbest.fit(features,labels )
#print list(kbest.get_support())
#print list(kbest.scores_)
print len(data_dict)
print(data_dict.keys())
#print(data_dict['TOTAL'].keys())
#print(data_dict['TOTAL'])
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

print(data_dict['THE TRAVEL AGENCY IN THE PARK'])
### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
data_dict.pop('LOCKHART EUGENE E')
print len(data_dict)
### Task 3: Create new feature(s)

data = featureFormat(data_dict, features)
### your code below
for employee in data_dict:
    if (data_dict[employee]['to_messages'] not in ['NaN', 0]) and (data_dict[employee]['from_this_person_to_poi'] not in ['NaN', 0]):
        data_dict[employee]['from_poi'] = float(data_dict[employee]['from_this_person_to_poi'])/float(data_dict[employee]['to_messages'])
    else:
        data_dict[employee]['from_poi'] = 0
for employee in data_dict:
    if (data_dict[employee]['from_messages'] not in ['NaN', 0]) and (data_dict[employee]['from_poi_to_this_person'] not in ['NaN', 0]):
        data_dict[employee]['to_poi'] = float(data_dict[employee]['from_poi_to_this_person'])/float(data_dict[employee]['from_messages'])
    else:
        data_dict[employee]['to_poi'] = 0
### Store to my_dataset for easy export below.
my_dataset = data_dict
#print(data_dict['METTS MARK'])
### Extract features and labels from dataset for local testing

#print features_list
#new_features_list =  ['poi','to_messages', 'loan_advances','other','director_fees','shared_receipt_with_poi','from_poi_to_this_person'] # You will need to use more features
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
# Scale features



from sklearn.metrics import accuracy_score
from sklearn import cross_validation
features_train, features_test,labels_train, labels_test=cross_validation.train_test_split(features,labels, test_size=0.4,random_state=0)






### Task 4: Try a varity of classifiers

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import GaussianNB
clf_naive_bayes = GaussianNB()
clf_naive_bayes = clf_naive_bayes.fit(features_train, labels_train)   
ypred = clf_naive_bayes.predict(features_test)
acc_naive_bayes = accuracy_score(ypred, labels_test)
print "accuary score of naive_bayes:",acc_naive_bayes
print "precision score of naive_bayes:",precision_score(labels_test, ypred, average='macro')
print "recall score of naive_bayes:",recall_score(labels_test, ypred, average='macro')

from sklearn.tree import DecisionTreeClassifier
clf_Tree = DecisionTreeClassifier()
clf_Tree = clf_Tree.fit(features_train, labels_train)   
ypred = clf_Tree.predict(features_test)
acc_Tree = accuracy_score(ypred, labels_test)
print "accuary score of DecisionTreeClassifier:",acc_naive_bayes
print "precision score of DecisionTreeClassifier:",precision_score(labels_test, ypred, average='macro')
print "recall score of DecisionTreeClassifier:",recall_score(labels_test, ypred, average='macro')


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit


from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
#new_feature_list =   ['poi','salary','to_messages', 'deferral_payments','total_payments','exercised_stock_options','restricted_stock', 'bonus', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options','restricted_stock_deferred', 'loan_advances','other','director_fees','shared_receipt_with_poi','from_messages','long_term_incentive','from_poi_to_this_person','from_poi','to_poi'] # You will need to use more features
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


dtree =  DecisionTreeClassifier()
scaler = preprocessing.MinMaxScaler()

gs =  Pipeline(steps=[('scaling',scaler), ('selection', SelectKBest(k=4)),("dt", dtree)])
param_grid = {"selection__k": range(4,10),
               "dt__criterion": ["gini", "entropy"],
              "dt__min_samples_split": [2, 10, 20],
              "dt__max_depth": [None, 2, 5, 10],
              "dt__min_samples_leaf": [1, 5, 10],
              "dt__max_leaf_nodes": [None, 5, 10, 20],
              }
sss = StratifiedShuffleSplit(n_splits=100, test_size=0.3, random_state = 42)
dtcclf = GridSearchCV(gs, param_grid, scoring='f1', cv=sss)

dtcclf.fit(features, labels)

clf = dtcclf.best_estimator_    


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)




def combine(self, n, k):      
        # write your code here  
        self.res = []
        tmp = []
        self.dfs(n, k, 1, 0, tmp)       
        return self.res

def dfs(self, n, k, m, p, tmp):
        if k == p:
            self.res.append(tmp[:])
            return
        for i in range(m, n+1):            
            tmp.append(i)            
            self.dfs(n, k, i+1, p+1, tmp)            
            tmp.pop()    
