#!/bin/bash

#make

echo \
echo '********** Stump 4 decision tree experiment **********' > ../result/DecisionTree4_result.txt
echo '********** Stump 4 decision tree experiment **********'
echo \ >> ../result/DecisionTree4_result.txt

# Using the features generated above, train a decision tree classifier
# to predict the data. This is just an example code and in the
# homework, you should perform five fold cross-validation. 

echo '********** Cross Validation 1 **********' >> ../result/DecisionTree4_result.txt
echo '********** Cross Validation 1 **********' 
echo \ >> ../result/DecisionTree4_result.txt

java -cp lib/weka.jar:bin cs446.homework2.decisionStump4 ./../feature/attribute/badges.attribute.train1.arff ./../feature/attribute/badges.attribute.test1.arff >> ../result/DecisionTree4_result.txt

echo '********** Cross Validation 2 **********' >> ../result/DecisionTree4_result.txt
echo '********** Cross Validation 2 **********' 
echo \ >> ../result/DecisionTree4_result.txt

java -cp lib/weka.jar:bin cs446.homework2.decisionStump4 ./../feature/attribute/badges.attribute.train2.arff ./../feature/attribute/badges.attribute.test2.arff >> ../result/DecisionTree4_result.txt

echo '********** Cross Validation 3 **********' >> ../result/DecisionTree4_result.txt
echo '********** Cross Validation 3 **********' 
echo \ >> ../result/DecisionTree4_result.txt

java -cp lib/weka.jar:bin cs446.homework2.decisionStump4 ./../feature/attribute/badges.attribute.train3.arff ./../feature/attribute/badges.attribute.test3.arff >> ../result/DecisionTree4_result.txt

echo '********** Cross Validation 4 **********' >> ../result/DecisionTree4_result.txt
echo '********** Cross Validation 4 **********' 
echo \ >> ../result/DecisionTree4_result.txt

java -cp lib/weka.jar:bin cs446.homework2.decisionStump4 ./../feature/attribute/badges.attribute.train4.arff ./../feature/attribute/badges.attribute.test4.arff >> ../result/DecisionTree4_result.txt

echo '********** Cross Validation 5 **********' >> ../result/DecisionTree4_result.txt
echo '********** Cross Validation 5 **********' 
echo \ >> ../result/DecisionTree4_result.txt

java -cp lib/weka.jar:bin cs446.homework2.decisionStump4 ./../feature/attribute/badges.attribute.train5.arff ./../feature/attribute/badges.attribute.test5.arff >> ../result/DecisionTree4_result.txt

