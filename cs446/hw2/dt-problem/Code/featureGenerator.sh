#!/bin/bash

# Generate the example features from the entire dataset. This shows an example of how
# the featurre files may be built. Note that don't necessarily have to
# use Java for this step.

mkdir bin

make

mkdir ./../feature/attribute
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold1 ./../feature/attribute/badges.attribute.fold1.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold2 ./../feature/attribute/badges.attribute.fold2.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold3 ./../feature/attribute/badges.attribute.fold3.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold4 ./../feature/attribute/badges.attribute.fold4.arff
cat ./../feature/attribute/badges.attribute.fold1.arff > ./../feature/attribute/badges.attribute.train1.arff
echo \ >> ./../feature/attribute/badges.attribute.train1.arff
tail -n+266 ./../feature/attribute/badges.attribute.fold2.arff >> ./../feature/attribute/badges.attribute.train1.arff
echo \ >> ./../feature/attribute/badges.attribute.train1.arff
tail -n+266 ./../feature/attribute/badges.attribute.fold3.arff >> ./../feature/attribute/badges.attribute.train1.arff
echo \ >> ./../feature/attribute/badges.attribute.train1.arff
tail -n+266 ./../feature/attribute/badges.attribute.fold4.arff >> ./../feature/attribute/badges.attribute.train1.arff
rm ./../feature/attribute/badges.attribute.fold*

java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold1 ./../feature/attribute/badges.attribute.fold1.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold2 ./../feature/attribute/badges.attribute.fold2.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold3 ./../feature/attribute/badges.attribute.fold3.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold5 ./../feature/attribute/badges.attribute.fold4.arff
cat ./../feature/attribute/badges.attribute.fold1.arff > ./../feature/attribute/badges.attribute.train2.arff
echo \ >> ./../feature/attribute/badges.attribute.train2.arff
tail -n+266 ./../feature/attribute/badges.attribute.fold2.arff >> ./../feature/attribute/badges.attribute.train2.arff
echo \ >> ./../feature/attribute/badges.attribute.train2.arff
tail -n+266 ./../feature/attribute/badges.attribute.fold3.arff >> ./../feature/attribute/badges.attribute.train2.arff
echo \ >> ./../feature/attribute/badges.attribute.train2.arff
tail -n+266 ./../feature/attribute/badges.attribute.fold4.arff >> ./../feature/attribute/badges.attribute.train2.arff
rm ./../feature/attribute/badges.attribute.fold*

java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold1 ./../feature/attribute/badges.attribute.fold1.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold2 ./../feature/attribute/badges.attribute.fold2.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold4 ./../feature/attribute/badges.attribute.fold3.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold5 ./../feature/attribute/badges.attribute.fold4.arff
cat ./../feature/attribute/badges.attribute.fold1.arff > ./../feature/attribute/badges.attribute.train3.arff
echo \ >> ./../feature/attribute/badges.attribute.train3.arff
tail -n+266 ./../feature/attribute/badges.attribute.fold2.arff >> ./../feature/attribute/badges.attribute.train3.arff
echo \ >> ./../feature/attribute/badges.attribute.train3.arff
tail -n+266 ./../feature/attribute/badges.attribute.fold3.arff >> ./../feature/attribute/badges.attribute.train3.arff
echo \ >> ./../feature/attribute/badges.attribute.train3.arff
tail -n+266 ./../feature/attribute/badges.attribute.fold4.arff >> ./../feature/attribute/badges.attribute.train3.arff
rm ./../feature/attribute/badges.attribute.fold*

java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold1 ./../feature/attribute/badges.attribute.fold1.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold3 ./../feature/attribute/badges.attribute.fold2.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold4 ./../feature/attribute/badges.attribute.fold3.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold5 ./../feature/attribute/badges.attribute.fold4.arff
cat ./../feature/attribute/badges.attribute.fold1.arff > ./../feature/attribute/badges.attribute.train4.arff
echo \ >> ./../feature/attribute/badges.attribute.train4.arff
tail -n+266 ./../feature/attribute/badges.attribute.fold2.arff >> ./../feature/attribute/badges.attribute.train4.arff
echo \ >> ./../feature/attribute/badges.attribute.train4.arff
tail -n+266 ./../feature/attribute/badges.attribute.fold3.arff >> ./../feature/attribute/badges.attribute.train4.arff
echo \ >> ./../feature/attribute/badges.attribute.train4.arff
tail -n+266 ./../feature/attribute/badges.attribute.fold4.arff >> ./../feature/attribute/badges.attribute.train4.arff
rm ./../feature/attribute/badges.attribute.fold*

java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold2 ./../feature/attribute/badges.attribute.fold1.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold3 ./../feature/attribute/badges.attribute.fold2.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold4 ./../feature/attribute/badges.attribute.fold3.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold5 ./../feature/attribute/badges.attribute.fold4.arff
cat ./../feature/attribute/badges.attribute.fold1.arff > ./../feature/attribute/badges.attribute.train5.arff
echo \ >> ./../feature/attribute/badges.attribute.train5.arff
tail -n+266 ./../feature/attribute/badges.attribute.fold2.arff >> ./../feature/attribute/badges.attribute.train5.arff
echo \ >> ./../feature/attribute/badges.attribute.train5.arff
tail -n+266 ./../feature/attribute/badges.attribute.fold3.arff >> ./../feature/attribute/badges.attribute.train5.arff
echo \ >> ./../feature/attribute/badges.attribute.train5.arff
tail -n+266 ./../feature/attribute/badges.attribute.fold4.arff >> ./../feature/attribute/badges.attribute.train5.arff
rm ./../feature/attribute/badges.attribute.fold*

java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold5 ./../feature/attribute/badges.attribute.test1.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold4 ./../feature/attribute/badges.attribute.test2.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold3 ./../feature/attribute/badges.attribute.test3.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold2 ./../feature/attribute/badges.attribute.test4.arff
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badgesData/badges.modified.data.fold1 ./../feature/attribute/badges.attribute.test5.arff
