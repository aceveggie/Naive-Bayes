import NaiveBayes
from numpy import *

#using scikits learn library 0.11

from sklearn.naive_bayes import GaussianNB

data = [[5.92, 190, 11] ,[5.58, 170, 12] ,[5.92, 165, 10],[5, 100, 6] ,[5.5, 150, 8] ,[5.42, 130, 7] ,[5.75, 150, 9], [6, 180,12], [7,220, 11]]
labs = ['male','male','male','female','female','female','female', 'male','male']

pred_data = [[6,130, 8],[7,199,12],[5.42,170,8],[5.8,220,11]]
node, prior_prob = NaiveBayes.train(data,labs)

output = NaiveBayes.predict(node, prior_prob, pred_data)

for predicted_value in output:
    print predicted_value



### NOW TESTING ITS OUTPUT COMPARED TO sklearn.naive_bayes implementation.

X = array(data)
y = array(labs)

gnb = GaussianNB()

classifier = gnb.fit(X, y)
print classifier.predict(array(pred_data))
print classifier.predict_proba(array(pred_data))
