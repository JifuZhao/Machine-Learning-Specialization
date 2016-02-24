% randomly select the data as train and test data

clear all;
load ./data/ex1.mat;
[train1, test1] = randomSelect(Ex1l10m100n500);
Ex1l10m100n500train = Ex1l10m100n500(train1, :);
Ex1l10m100n500test = Ex1l10m100n500(test1, :);

[train2, test2] = randomSelect(Ex1l10m100n1000);
Ex1l10m100n1000train = Ex1l10m100n1000(train2, :);
Ex1l10m100n1000test = Ex1l10m100n1000(test2, :);

save ./data/ex1random Ex1l10m100n500train Ex1l10m100n500test Ex1l10m100n1000train Ex1l10m100n1000test

%%
clear all
load ./data/ex2.mat;

[train1, test1] = randomSelect(Ex2l10m20n40);
Ex2l10m20n40train = Ex2l10m20n40(train1, :);
Ex2l10m20n40test = Ex2l10m20n40(test1, :);

[train2, test2] = randomSelect(Ex2l10m20n80);
Ex2l10m20n80train = Ex2l10m20n80(train2, :);
Ex2l10m20n80test = Ex2l10m20n80(test2, :);

[train3, test3] = randomSelect(Ex2l10m20n120);
Ex2l10m20n120train = Ex2l10m20n120(train3, :);
Ex2l10m20n120test = Ex2l10m20n120(test3, :);

[train4, test4] = randomSelect(Ex2l10m20n160);
Ex2l10m20n160train = Ex2l10m20n160(train4, :);
Ex2l10m20n160test = Ex2l10m20n160(test4, :);

[train5, test5] = randomSelect(Ex2l10m20n200);
Ex2l10m20n200train = Ex2l10m20n200(train5, :);
Ex2l10m20n200test = Ex2l10m20n200(test5, :);

save ./data/ex2random Ex2l10m20n40train Ex2l10m20n40test Ex2l10m20n80train Ex2l10m20n80test ...
    Ex2l10m20n120train Ex2l10m20n120test Ex2l10m20n160train Ex2l10m20n160test Ex2l10m20n200train ...
    Ex2l10m20n200test

%%
clear all
load ./data/ex3.mat;

[train1, test1] = randomSelect(Ex3l10m100n1000train);
Ex3l10m100n1000randomtrain = Ex3l10m100n1000train(train1, :);
Ex3l10m100n1000randomtest = Ex3l10m100n1000train(test1, :);

[train2, test2] = randomSelect(Ex3l10m500n1000train);
Ex3l10m500n1000randomtrain = Ex3l10m500n1000train(train2, :);
Ex3l10m500n1000randomtest = Ex3l10m500n1000train(test2, :);

[train3, test3] = randomSelect(Ex3l10m1000n1000train);
Ex3l10m1000n1000randomtrain = Ex3l10m1000n1000train(train3, :);
Ex3l10m1000n1000randomtest = Ex3l10m1000n1000train(test3, :);

save ./data/ex3random Ex3l10m100n1000randomtrain Ex3l10m100n1000randomtest Ex3l10m500n1000randomtrain ...
    Ex3l10m500n1000randomtest Ex3l10m1000n1000randomtrain Ex3l10m1000n1000randomtest

%%
clear all
load ./data/ex4.mat;

[train1, test1] = randomSelect(Ex4l10m100n1000train);
Ex4l10m100n1000randomtrain = Ex4l10m100n1000train(train1, :);
Ex4l10m100n1000randomtest = Ex4l10m100n1000train(test1, :);

[train2, test2] = randomSelect(Ex4l10m500n1000train);
Ex4l10m500n1000randomtrain = Ex4l10m500n1000train(train2, :);
Ex4l10m500n1000randomtest = Ex4l10m500n1000train(test2, :);

[train3, test3] = randomSelect(Ex4l10m1000n1000train);
Ex4l10m1000n1000randomtrain = Ex4l10m1000n1000train(train3, :);
Ex4l10m1000n1000randomtest = Ex4l10m1000n1000train(test3, :);

save ./data/ex4random Ex4l10m100n1000randomtrain Ex4l10m100n1000randomtest Ex4l10m500n1000randomtrain ...
    Ex4l10m500n1000randomtest Ex4l10m1000n1000randomtrain Ex4l10m1000n1000randomtest
