%% Experiment 1
[y1, x1] = gen(10, 100, 500, 50000, 0);
Ex1l10m100n500 = [x1, y1];

[y2, x2] = gen(10, 100, 1000, 50000, 0);
Ex1l10m100n1000 = [x2, y2];

save ./data/ex1 Ex1l10m100n500 Ex1l10m100n1000

%% Experiment 2
[y3, x3] = gen(10, 20, 40, 50000, 0);
Ex2l10m20n40 = [x3, y3];

[y4, x4] = gen(10, 20, 80, 50000, 0);
Ex2l10m20n80 = [x4, y4];

[y5, x5] = gen(10, 20, 120, 50000, 0);
Ex2l10m20n120 = [x5, y5];

[y6, x6] = gen(10, 20, 160, 50000, 0);
Ex2l10m20n160 = [x6, y6];

[y7, x7] = gen(10, 20, 200, 50000, 0);
Ex2l10m20n200 = [x7, y7];

save ./data/ex2 Ex2l10m20n40 Ex2l10m20n80 Ex2l10m20n120 Ex2l10m20n160 Ex2l10m20n200


%% Experiment 3
[y8, x8] = gen(10, 100, 1000, 50000, 1);
Ex3l10m100n1000train = [x8, y8];

[y9, x9] = gen(10, 100, 1000, 10000, 0);
Ex3l10m100n1000test = [x9, y9];

[y10, x10] = gen(10, 500, 1000, 50000, 1);
Ex3l10m500n1000train = [x10, y10];

[y11, x11] = gen(10, 500, 1000, 10000, 0);
Ex3l10m500n1000test = [x11, y11];

[y12, x12] = gen(10, 1000, 1000, 50000, 1);
Ex3l10m1000n1000train = [x12, y12];

[y13, x13] = gen(10, 1000, 1000, 10000, 0);
Ex3l10m1000n1000test = [x13, y13];

save ./data/ex3 Ex3l10m100n1000train Ex3l10m100n1000test Ex3l10m500n1000train ...
    Ex3l10m500n1000test Ex3l10m1000n1000train Ex3l10m1000n1000test

%% Experiment 4
[y14, x14] = unba_gen(10, 100, 1000, 50000, 0.1);
Ex4l10m100n1000train = [x14, y14];

[y15, x15] = unba_gen(10, 100, 1000, 10000, 0.1);
Ex4l10m100n1000test = [x15, y15];

[y16, x16] = unba_gen(10, 500, 1000, 50000, 0.1);
Ex4l10m500n1000train = [x16, y16];

[y17, x17] = unba_gen(10, 500, 1000, 10000, 0.1);
Ex4l10m500n1000test = [x17, y17];

[y18, x18] = unba_gen(10, 1000, 1000, 50000, 0.1);
Ex4l10m1000n1000train = [x18, y18];

[y19, x19] = unba_gen(10, 1000, 1000, 10000, 0.1);
Ex4l10m1000n1000test = [x19, y19];

save ./data/ex4 Ex4l10m100n1000train Ex4l10m100n1000test Ex4l10m500n1000train ...
    Ex4l10m500n1000test Ex4l10m1000n1000train Ex4l10m1000n1000test