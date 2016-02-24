% Experiment 3
clear all;

load ./data/ex3.mat;
load ./data/ex3random.mat

%% Setting the parameters for different algorithms
% Perception with margin algorithm
eta_perceptionMargin = [1.5, 0.25, 0.03, 0.005, 0.001];
% Winnow algorithm
alpha_winnow = [1.1, 1.01, 1.005, 1.0005, 1.0001];
% Winnow with margin algorithm
alpha_winnowMargin = [1.1, 1.01, 1.005, 1.0005, 1.0001];
gamma_winnowMargin = [2.0, 0.3, 0.04, 0.006, 0.001];
% AdaGrad algorithm
eta_AdaGrad = [1.5, 0.25, 0.03, 0.005, 0.001];

%% Experiment for l = 10, m = 100, 500, 1000, n = 1000
fileID = fopen('./result/ex3.txt','w');

for i = 1:3
    if i == 1
        trainx = Ex3l10m100n1000randomtrain(:, 1:(end-1));
        trainy = Ex3l10m100n1000randomtrain(:, end);
        testx = Ex3l10m100n1000randomtest(:, 1:(end-1));
        testy = Ex3l10m100n1000randomtest(:, end);
        fprintf(fileID, '\n*******************************************\n');
        fprintf(fileID, 'Experiment with l = 10, m = 100, n = 1000\n');
        fprintf(fileID, '*******************************************\n');
    elseif i == 2
        trainx = Ex3l10m500n1000randomtrain(:, 1:(end-1));
        trainy = Ex3l10m500n1000randomtrain(:, end);
        testx = Ex3l10m500n1000randomtest(:, 1:(end-1));
        testy = Ex3l10m500n1000randomtest(:, end);
        fprintf(fileID, '\n*******************************************\n');
        fprintf(fileID, 'Experiment with l = 10, m = 500, n = 1000\n');
        fprintf(fileID, '*******************************************\n');
    elseif i == 3
        trainx = Ex3l10m1000n1000randomtrain(:, 1:(end-1));
        trainy = Ex3l10m1000n1000randomtrain(:, end);
        testx = Ex3l10m1000n1000randomtest(:, 1:(end-1));
        testy = Ex3l10m1000n1000randomtest(:, end);
        fprintf(fileID, '\n*******************************************\n');
        fprintf(fileID, 'Experiment with l = 10, m = 1000, n = 1000\n');
        fprintf(fileID, '*******************************************\n');
    end

    % Perception
    fprintf(fileID, '\nPerception Experiment\n');
    [w, theta, mistake] = Perceptron(20, trainx, trainy);
    [n, accuracy] = mistakeCalculator(w, testx, theta, testy);
    fprintf(fileID, 'Mistake numbers is %d and accuracy is %f\n', [n, accuracy]);

    % Perception with Margin
    fprintf(fileID, '\nPerception with Margin Experiment\n');
    for i = 1:length(eta_perceptionMargin)
        eta = eta_perceptionMargin(i);
        fprintf(fileID, 'When eta is %f:\n', eta);
        [w, theta, mistake] = PerceptronMargin(20, trainx, trainy, eta);
        [n, accuracy] = mistakeCalculator(w, testx, theta, testy);
        fprintf(fileID, 'Mistake numbers is %d and accuracy is %f\n', [n, accuracy]);
    end

    % Winnow
    fprintf(fileID, '\nWinnow Experiment\n');
    for i = 1:length(alpha_winnow)
        alpha = alpha_winnow(i);
        fprintf(fileID, 'When alpha is %f:\n', alpha);
        [w, theta, mistake] = Winnow(20, trainx, trainy, alpha);
        [n, accuracy] = mistakeCalculator(w, testx, theta, testy);
        fprintf(fileID, 'Mistake numbers is %d and accuracy is %f\n', [n, accuracy]);
    end
        
    % Winnow with margin
    fprintf(fileID, '\nWinnow with Margin Experiment\n');
    for i = 1:length(alpha_winnowMargin)
        for j = 1:length(gamma_winnowMargin)
           alpha = alpha_winnow(i);
           gamma = gamma_winnowMargin(j);
           fprintf(fileID, 'When alpha is %f and gamma is %f \n', [alpha, gamma]);
           [w, theta, mistake] = WinnowMargin(20, trainx, trainy, alpha, gamma);
           [n, accuracy] = mistakeCalculator(w, testx, theta, testy);
           fprintf(fileID, 'Mistake numbers is %d and accuracy is %f\n', [n, accuracy]);
        end
    end

    % AdaGrad
    fprintf(fileID, '\nAdaGrad Experiment\n');
    for i = 1:length(eta_AdaGrad)
        eta = eta_AdaGrad(i);
        fprintf(fileID, 'When eta is %f:\n', eta);
        [w, theta, mistake] = AdaGrad(20, trainx, trainy, eta);
        [n, accuracy] = mistakeCalculator(w, testx, theta, testy);
        fprintf(fileID, 'Mistake numbers is %d and accuracy is %f\n', [n, accuracy]);
    end

end

fclose(fileID);

%%
fileID = fopen('./result/ex3_2.txt','w');

% Perception with margin algorithm
eta_perceptionMargin = [0.005, 0.005, 0.03];
% Winnow algorithm
alpha_winnow = [1.1, 1.1, 1.1];
% Winnow with margin algorithm
alpha_winnowMargin = [1.1, 1.1, 1.1];
gamma_winnowMargin = [0.006, 0.04, 0.001];
% AdaGrad algorithm
eta_AdaGrad = [0.25, 1.5, 1.5];

for i = 1:3
    if i == 1
        trainx = Ex3l10m100n1000train(:, 1:(end-1));
        trainy = Ex3l10m100n1000train(:, end);
        testx = Ex3l10m100n1000test(:, 1:(end-1));
        testy = Ex3l10m100n1000test(:, end);
        fprintf(fileID, '\n*******************************************\n');
        fprintf(fileID, 'Experiment with l = 10, m = 100, n = 1000\n');
        fprintf(fileID, '*******************************************\n');
    elseif i == 2
        trainx = Ex3l10m500n1000train(:, 1:(end-1));
        trainy = Ex3l10m500n1000train(:, end);
        testx = Ex3l10m500n1000test(:, 1:(end-1));
        testy = Ex3l10m500n1000test(:, end);
        fprintf(fileID, '\n*******************************************\n');
        fprintf(fileID, 'Experiment with l = 10, m = 500, n = 1000\n');
        fprintf(fileID, '*******************************************\n');
    elseif i == 3
        trainx = Ex3l10m1000n1000train(:, 1:(end-1));
        trainy = Ex3l10m1000n1000train(:, end);
        testx = Ex3l10m1000n1000test(:, 1:(end-1));
        testy = Ex3l10m1000n1000test(:, end);
        fprintf(fileID, '\n*******************************************\n');
        fprintf(fileID, 'Experiment with l = 10, m = 1000, n = 1000\n');
        fprintf(fileID, '*******************************************\n');
    end

    % Perception
    fprintf(fileID, '\nPerception Experiment\n');
    [w, theta, mistake] = Perceptron(20, trainx, trainy);
    [n, accuracy] = mistakeCalculator(w, testx, theta, testy);
    fprintf(fileID, 'Mistake numbers is %d and accuracy is %f\n', [n, accuracy]);

    % Perception with Margin
    fprintf(fileID, '\nPerception with Margin Experiment\n');
    eta = eta_perceptionMargin(i);
    fprintf(fileID, 'When eta is %f:\n', eta);
    [w, theta, mistake] = PerceptronMargin(20, trainx, trainy, eta);
    [n, accuracy] = mistakeCalculator(w, testx, theta, testy);
    fprintf(fileID, 'Mistake numbers is %d and accuracy is %f\n', [n, accuracy]);

    % Winnow
    fprintf(fileID, '\nWinnow Experiment\n');
    alpha = alpha_winnow(i);
    fprintf(fileID, 'When alpha is %f:\n', alpha);
    [w, theta, mistake] = Winnow(20, trainx, trainy, alpha);
    [n, accuracy] = mistakeCalculator(w, testx, theta, testy);
    fprintf(fileID, 'Mistake numbers is %d and accuracy is %f\n', [n, accuracy]);
        
    % Winnow with margin
    fprintf(fileID, '\nWinnow with Margin Experiment\n');
    alpha = alpha_winnow(i);
    gamma = gamma_winnowMargin(i);
    fprintf(fileID, 'When alpha is %f and gamma is %f \n', [alpha, gamma]);
    [w, theta, mistake] = WinnowMargin(20, trainx, trainy, alpha, gamma);
    [n, accuracy] = mistakeCalculator(w, testx, theta, testy);
    fprintf(fileID, 'Mistake numbers is %d and accuracy is %f\n', [n, accuracy]);

    % AdaGrad
    fprintf(fileID, '\nAdaGrad Experiment\n');
    eta = eta_AdaGrad(i);
    fprintf(fileID, 'When eta is %f:\n', eta);
    [w, theta, mistake] = AdaGrad(20, trainx, trainy, eta);
    [n, accuracy] = mistakeCalculator(w, testx, theta, testy);
    fprintf(fileID, 'Mistake numbers is %d and accuracy is %f\n', [n, accuracy]);

end

fclose(fileID);