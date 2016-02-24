#!/bin/bash

echo '**********************************************************************'
echo '****************************** Note **********************************'
echo '******** All result will be stored in ./result/result.txt ************'
echo '********** There will no result shown on the screen ******************'
echo '**********************************************************************'
echo '**********************************************************************'
echo \

# Generate all training and testing samples
chmod +x featureGenerator.sh
./featureGenerator.sh

#chmod +x featureGenerator2.sh
#./featureGenerator2.sh

# Decision tree of 4 stump experiment
chmod +x decision4_test.sh
./decision4_test.sh

# Decision tree of 8 stump experiment
chmod +x decision8_test.sh
./decision8_test.sh

# Full Decision tree experiment
chmod +x decisionFull_test.sh
./decisionFull_test.sh

# simple SGD experiment
chmod +x simpleSGD_test.sh
./simpleSGD_test.sh

# SGD 100 experiment
chmod +x stump100SGD_test.sh
./stump100SGD_test.sh
