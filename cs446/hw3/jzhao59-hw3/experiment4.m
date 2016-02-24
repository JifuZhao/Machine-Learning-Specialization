% Experiment 4
clear all;

load ./data/ex4.mat;
load ./data/ex4random.mat

%% Modified Perception
eta_perceptionMargin = [1.5, 0.25, 0.03, 0.005, 0.001];
GammaP = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001];
GammaN = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001];

fileID = fopen('./result/ex4.txt','w');

for i = 1:3
    if i == 1
        trainx = Ex4l10m100n1000randomtrain(:, 1:(end-1));
        trainy = Ex4l10m100n1000randomtrain(:, end);
        testx = Ex4l10m100n1000randomtest(:, 1:(end-1));
        testy = Ex4l10m100n1000randomtest(:, end);
        fprintf(fileID, '\n*******************************************\n');
        fprintf(fileID, 'Experiment with l = 10, m = 100, n = 1000\n');
        fprintf(fileID, '*******************************************\n');
    elseif i == 2
        trainx = Ex4l10m500n1000randomtrain(:, 1:(end-1));
        trainy = Ex4l10m500n1000randomtrain(:, end);
        testx = Ex4l10m500n1000randomtest(:, 1:(end-1));
        testy = Ex4l10m500n1000randomtest(:, end);
        fprintf(fileID, '\n*******************************************\n');
        fprintf(fileID, 'Experiment with l = 10, m = 500, n = 1000\n');
        fprintf(fileID, '*******************************************\n');
    elseif i == 3
        trainx = Ex4l10m1000n1000randomtrain(:, 1:(end-1));
        trainy = Ex4l10m1000n1000randomtrain(:, end);
        testx = Ex4l10m1000n1000randomtest(:, 1:(end-1));
        testy = Ex4l10m1000n1000randomtest(:, end);
        fprintf(fileID, '\n*******************************************\n');
        fprintf(fileID, 'Experiment with l = 10, m = 1000, n = 1000\n');
        fprintf(fileID, '*******************************************\n');
    end

    % Modified Perception
    fprintf(fileID, '\nModified Perception\n');
    for j = 1:length(GammaP)
        for k = 1:length(GammaN)
            for l = 1:length(eta_perceptionMargin)
                eta = eta_perceptionMargin(l);
                gammaP = GammaP(j);
                gammaN = GammaN(k);
                fprintf(fileID, 'When eta is %f, gammaP is %f and gammaN is %f :\n', [eta, gammaP, gammaN]);
                [w, theta, mistake] = PerceptronModified(20, trainx, trainy, eta, gammaP, gammaN);
                [n, accuracy] = mistakeCalculator(w, testx, theta, testy);
                fprintf(fileID, 'Mistake numbers is %d and accuracy is %f\n', [n, accuracy]);
            end
        end
    end

end

fclose(fileID);

%% Perception
fileID = fopen('./result/ex4_2.txt','w');

etaSet = [0.001, 0.001, 0.001];
GammaPSet = [0.005, 1.000, 0.005];
GammaNSet = [1.000, 1.000, 0.005];

for i = 1:3
    if i == 1
        trainx = Ex4l10m100n1000train(:, 1:(end-1));
        trainy = Ex4l10m100n1000train(:, end);
        testx = Ex4l10m100n1000test(:, 1:(end-1));
        testy = Ex4l10m100n1000test(:, end);
        fprintf(fileID, '\n*******************************************\n');
        fprintf(fileID, 'Experiment with l = 10, m = 100, n = 1000\n');
        fprintf(fileID, '*******************************************\n');
    elseif i == 2
        trainx = Ex4l10m500n1000train(:, 1:(end-1));
        trainy = Ex4l10m500n1000train(:, end);
        testx = Ex4l10m500n1000test(:, 1:(end-1));
        testy = Ex4l10m500n1000test(:, end);
        fprintf(fileID, '\n*******************************************\n');
        fprintf(fileID, 'Experiment with l = 10, m = 500, n = 1000\n');
        fprintf(fileID, '*******************************************\n');
    elseif i == 3
        trainx = Ex4l10m1000n1000train(:, 1:(end-1));
        trainy = Ex4l10m1000n1000train(:, end);
        testx = Ex4l10m1000n1000test(:, 1:(end-1));
        testy = Ex4l10m1000n1000test(:, end);
        fprintf(fileID, '\n*******************************************\n');
        fprintf(fileID, 'Experiment with l = 10, m = 1000, n = 1000\n');
        fprintf(fileID, '*******************************************\n');
    end

    % Perception
    fprintf(fileID, '\nPerception Experiment\n');
    [w, theta, mistake] = Perceptron(1, trainx, trainy);
    [n, accuracy] = mistakeCalculator(w, testx, theta, testy);
    fprintf(fileID, 'Mistake numbers is %d and accuracy is %f\n', [n, accuracy]);
    
    % Modified Perception method    
    fprintf(fileID, '\nPerception Experiment\n');
    eta = etaSet(i);
    gammaP = GammaPSet(i);
    gammaN = GammaNSet(i);
    fprintf(fileID, 'When eta is %f, gammaP is %f and gammaN is %f :\n', [eta, gammaP, gammaN]);
    [w, theta, mistake] = PerceptronModified(1, trainx, trainy, eta, gammaP, gammaN);
    [n, accuracy] = mistakeCalculator(w, testx, theta, testy);
    fprintf(fileID, 'Mistake numbers is %d and accuracy is %f\n', [n, accuracy]);
end

fclose(fileID);

