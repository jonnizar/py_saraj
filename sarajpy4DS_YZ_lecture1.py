# Titel: Homework 1 for Saraj's intro to data science:  https://www.youtube.com/watch?v=T5pRlIbr6gg&index=1&list=PL2-dafEMk2A6QKz1mrk1uIGfHkC1zZ6UU
# Author: Yan Zaripov <yan dot zaripov at gmail dot com>
# License: BSD 3 clause
# Copyright <2017> <Yan Zaripov>

from sklearn import tree

#let's try naive bayes to train and classify the data gven below
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#decision trees classifier
clf = tree.DecisionTreeClassifier()


# CHALLENGE - create 3 more classifiers...
# 1: naive Bayes Gaussian classifier
gnb = GaussianNB()
# 2: C-Support Vector Classification
svc = SVC()
# 3: RandomForestClassifier
rndf = RandomForestClassifier()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#useful variables

numInputObs = len(Y)

# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
gnb = gnb.fit(X, Y)
svc = svc.fit(X, Y)
rndf = rndf.fit(X, Y)

#Predict the values using the input X for [height, weight, shoe_size]
clfPrediction = clf.predict(X)
gnbPrediction = gnb.predict(X)
svcPrediction = svc.predict(X)
rndfPrediction = rndf.predict(X)

#initiate match counters

clfMatchCounter = 0;
gnbMatchCounter = 0;
svcMatchCounter = 0;
rndfMatchCounter = 0;

# CHALLENGE compare their reusults and print the best one!

for i in range(numInputObs):
	#compare prediction and training inputs
	if clfPrediction[i] == Y [i]:
		clfMatchCounter += 1

	if gnbPrediction[i] == Y [i]:
		gnbMatchCounter += 1

	if svcPrediction[i] == Y [i]:
		svcMatchCounter += 1

	if svcPrediction[i] == Y [i]:
		rndfMatchCounter += 1

#output the results AND find the winner

matchTracker = {"Decision trees classifier": clfMatchCounter, 
		"Naive Bayes Gaussian classifier": gnbMatchCounter,
		"C-Support Vector Classification": svcMatchCounter,
		"Random Forest ": rndfMatchCounter}


print("Nubmer of classifier matches:")

print()

#inititate bias for comparing matching avlues

curMaxVal = 0
bestMatch = {}

# iterate through results

for key, val in matchTracker.items():
	print (key, val)
	#determine max Matching Value
	if val >= curMaxVal:
		bestMatch [key] = val 
		curMaxVal = val

print()

print ("Best classifications:")

print()

for key, val in bestMatch.items():
	print (key, val)




