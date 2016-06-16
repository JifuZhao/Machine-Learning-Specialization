clear all; clc;
% a3(wd_coefficient, n_hid, n_iters, learning_rate, momentum_multiplier, do_early_stopping, mini_batch_size)

%% Question 1, without any training
a3(0, 0, 0, 0, 0, false, 0)

%% Question 2
a3(0, 10, 70, 0.005, 0, false, 4)

%% Question 3
for learning_rate = [0.002, 0.01, 0.05, 0.2, 1.0, 5.0, 20.0]
    momentum = 0.0;
    a3(0, 10, 70, learning_rate, momentum, false, 4);
    pause;
end

for learning_rate = [0.002, 0.01, 0.05, 0.2, 1.0, 5.0, 20.0]
    momentum = 0.9;
    a3(0, 10, 70, learning_rate, momentum, false, 4);
    pause;
end

%% Question 5
a3(0, 200, 1000, 0.35, 0.9, false, 100)

%% Question 6
a3(0, 200, 1000, 0.35, 0.9, true, 100)

%% Question 7
a3(0, 200, 1000, 0.35, 0.9, false, 100)
pause

a3(0.0001, 200, 1000, 0.35, 0.9, false, 100)
pause

a3(0.001, 200, 1000, 0.35, 0.9, false, 100)
pause

a3(0.1, 200, 1000, 0.35, 0.9, false, 100)
pause

a3(1, 200, 1000, 0.35, 0.9, false, 100)
pause

a3(10, 200, 1000, 0.35, 0.9, false, 100)

%% Question 8
a3(0, 10, 1000, 0.35, 0.9, false, 100)
pause

a3(0, 30, 1000, 0.35, 0.9, false, 100)
pause

a3(0, 100, 1000, 0.35, 0.9, false, 100)
pause

a3(0, 130, 1000, 0.35, 0.9, false, 100)
pause

a3(0, 170, 1000, 0.35, 0.9, false, 100)

%% Question 9
a3(0, 18, 1000, 0.35, 0.9, true, 100)
pause

a3(0, 37, 1000, 0.35, 0.9, true, 100)
pause

a3(0, 83, 1000, 0.35, 0.9, true, 100)
pause

a3(0, 113, 1000, 0.35, 0.9, true, 100)
pause

a3(0, 189, 1000, 0.35, 0.9, true, 100)

%% Question 10
a3(0, 37, 1000, 0.35, 0.9, true, 100)


