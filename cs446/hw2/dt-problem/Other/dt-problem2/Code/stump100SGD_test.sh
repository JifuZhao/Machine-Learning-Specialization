#!/bin/bash

#make

echo '********** stump 100 SGD experiment **********'

java -cp lib/weka.jar:bin cs446.homework2.SGDgenerator ./../feature/attribute/badges.attribute.train1.arff ./../feature/attribute/badges.attribute.test1.arff ./../feature/attribute/badges.attribute.SGD100train1.arff ./../feature/attribute/badges.attribute.SGD100test1.arff

java -cp lib/weka.jar:bin cs446.homework2.SGDgenerator ./../feature/attribute/badges.attribute.train2.arff ./../feature/attribute/badges.attribute.test2.arff ./../feature/attribute/badges.attribute.SGD100train2.arff ./../feature/attribute/badges.attribute.SGD100test2.arff

java -cp lib/weka.jar:bin cs446.homework2.SGDgenerator ./../feature/attribute/badges.attribute.train3.arff ./../feature/attribute/badges.attribute.test3.arff ./../feature/attribute/badges.attribute.SGD100train3.arff ./../feature/attribute/badges.attribute.SGD100test3.arff

java -cp lib/weka.jar:bin cs446.homework2.SGDgenerator ./../feature/attribute/badges.attribute.train4.arff ./../feature/attribute/badges.attribute.test4.arff ./../feature/attribute/badges.attribute.SGD100train4.arff ./../feature/attribute/badges.attribute.SGD100test4.arff

java -cp lib/weka.jar:bin cs446.homework2.SGDgenerator ./../feature/attribute/badges.attribute.train5.arff ./../feature/attribute/badges.attribute.test5.arff ./../feature/attribute/badges.attribute.SGD100train5.arff ./../feature/attribute/badges.attribute.SGD100test5.arff


chmod +x ./src/stump100SGD.py
./src/stump100SGD.py
