% Experiment 2
clear all;

load ./data/ex2.mat;
load ./data/ex2random.mat;

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

%% Experiment for l = 10, m = 100, n = 40, 80, 120, 160, 200
fileID = fopen('./result/ex2.txt','w');

for i = 1:5
    if i == 1
        trainx = Ex2l10m20n40train(:, 1:(end-1));
        trainy = Ex2l10m20n40train(:, end);
        testx = Ex2l10m20n40test(:, 1:(end-1));
        testy = Ex2l10m20n40test(:, end);
        fprintf(fileID, '\n*******************************************\n');
        fprintf(fileID, 'Experiment with l = 10, m = 20, n = 40\n');
        fprintf(fileID, '*******************************************\n');
    elseif i == 2
        trainx = Ex2l10m20n80train(:, 1:(end-1));
        trainy = Ex2l10m20n80train(:, end);
        testx = Ex2l10m20n80test(:, 1:(end-1));
        testy = Ex2l10m20n80test(:, end);
        fprintf(fileID, '\n*******************************************\n');
        fprintf(fileID, 'Experiment with l = 10, m = 20, n = 80\n');
        fprintf(fileID, '*******************************************\n');
    elseif i == 3
        trainx = Ex2l10m20n120train(:, 1:(end-1));
        trainy = Ex2l10m20n120train(:, end);
        testx = Ex2l10m20n120test(:, 1:(end-1));
        testy = Ex2l10m20n120test(:, end);
        fprintf(fileID, '\n*******************************************\n');
        fprintf(fileID, 'Experiment with l = 10, m = 20, n = 120\n');
        fprintf(fileID, '*******************************************\n');
    elseif i == 4
        trainx = Ex2l10m20n160train(:, 1:(end-1));
        trainy = Ex2l10m20n160train(:, end);
        testx = Ex2l10m20n160test(:, 1:(end-1));
        testy = Ex2l10m20n160test(:, end);
        fprintf(fileID, '\n*******************************************\n');
        fprintf(fileID, 'Experiment with l = 10, m = 20, n = 160\n');
        fprintf(fileID, '*******************************************\n');
    elseif i == 5
        trainx = Ex2l10m20n200train(:, 1:(end-1));
        trainy = Ex2l10m20n200train(:, end);
        testx = Ex2l10m20n200test(:, 1:(end-1));
        testy = Ex2l10m20n200test(:, end);
        fprintf(fileID, '\n*******************************************\n');
        fprintf(fileID, 'Experiment with l = 10, m = 20, n = 200\n');
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
% Perception with margin algorithm
eta_perceptionMargin = [0.25, 0.25, 0.03, 0.03, 0.03];
% Winnow algorithm
alpha_winnow = [1.1, 1.1, 1.1, 1.1, 1.1];
% Winnow with margin algorithm
alpha_winnowMargin = [1.1, 1.1, 1.1, 1.1, 1.1];
gamma_winnowMargin = [2.0, 2.0, 0.001, 0.001, 2.0];
% AdaGrad algorithm
eta_AdaGrad = [1.5, 1.5, 1.5, 1.5, 1.5];

num_mistake = zeros(5, 5);
R = zeros(5, 5);
threshold = 1000;
iternation = 10;
for i = 1:5
    if i == 1
        datax = Ex2l10m20n40(:, 1:(end-1));
        datay = Ex2l10m20n40(:, end);
        etaPerception = eta_perceptionMargin(1);
        alphaWinnow = alpha_winnow(1);
        alphaWinnowMargin = alpha_winnowMargin(1);
        gammaWinnowMargin = gamma_winnowMargin(1);
        etaAda = eta_AdaGrad(1);
    elseif i == 2
        datax = Ex2l10m20n80(:, 1:(end-1));
        datay = Ex2l10m20n80(:, end);
        etaPerception = eta_perceptionMargin(2);
        alphaWinnow = alpha_winnow(2);
        alphaWinnowMargin = alpha_winnowMargin(2);
        gammaWinnowMargin = gamma_winnowMargin(2);
        etaAda = eta_AdaGrad(2);
    elseif i == 3
        datax = Ex2l10m20n120(:, 1:(end-1));
        datay = Ex2l10m20n120(:, end);
        etaPerception = eta_perceptionMargin(3);
        alphaWinnow = alpha_winnow(3);
        alphaWinnowMargin = alpha_winnowMargin(3);
        gammaWinnowMargin = gamma_winnowMargin(3);
        etaAda = eta_AdaGrad(3);
    elseif i == 4
        datax = Ex2l10m20n160(:, 1:(end-1));
        datay = Ex2l10m20n160(:, end);
        etaPerception = eta_perceptionMargin(4);
        alphaWinnow = alpha_winnow(4);
        alphaWinnowMargin = alpha_winnowMargin(4);
        gammaWinnowMargin = gamma_winnowMargin(4);
        etaAda = eta_AdaGrad(4);
    elseif i == 5
        datax = Ex2l10m20n200(:, 1:(end-1));
        datay = Ex2l10m20n200(:, end);
        etaPerception = eta_perceptionMargin(5);
        alphaWinnow = alpha_winnow(5);
        alphaWinnowMargin = alpha_winnowMargin(5);
        gammaWinnowMargin = gamma_winnowMargin(5);
        etaAda = eta_AdaGrad(5);
    end
    
    [w, theta, num_mistake(1, i), R(1, i)] = Perceptron2(iternation, datax, datay, threshold);
    [w, theta, num_mistake(2, i), R(2, i)] = PerceptronMargin2(iternation, datax, datay, etaPerception, threshold);    
    [w, theta, num_mistake(3, i), R(3, i)] = Winnow2(iternation, datax, datay, alphaWinnow, threshold);   
    [w, theta, num_mistake(4, i), R(4, i)] = WinnowMargin2(iternation, datax, datay, alphaWinnowMargin, gammaWinnowMargin, threshold);
    [w, theta, num_mistake(5, i), R(5, i)] = AdaGrad2(iternation, datax, datay, etaAda, threshold);
end

x = 40:40:200;
plot(x, num_mistake(1, :), '-o', 'LineWidth',2); hold on;
plot(x, num_mistake(2, :), '-*', 'LineWidth',2); hold on;
plot(x, num_mistake(3, :), '-+', 'LineWidth',2); hold on;
plot(x, num_mistake(4, :),'-X', 'LineWidth',2); hold on;
plot(x, num_mistake(5, :), '-^', 'LineWidth',2); hold on;
ax = gca; ax.XTick = [40 80 120 160 200];
legend('Perception', 'Perception with Margin', 'Winnow', 'Winnow with Margin', ...
'AdaGrad', 'Location','northwest');
title('Mistakes numbers when l=10, m=20, n=40, 80, 120, 160, 200');
saveas(gcf, './result/Ex2mistake1.png') 