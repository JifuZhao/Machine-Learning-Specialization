% Experiment 1
clear all;

load ./data/ex1random.mat;
load ./data/ex1.mat

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

x1 = Ex1l10m100n500train(:, 1:(end-1));
y1 = Ex1l10m100n500train(:, end);

x2 = Ex1l10m100n1000train(:, 1:(end-1));
y2 = Ex1l10m100n1000train(:, end);

%% Experiment for l = 10, m = 100, n = 500, N = 50000
fileID = fopen('./result/ex1.txt','w');

fprintf(fileID, '\n*******************************************\n');
fprintf(fileID, 'Experiment with l = 10, m = 100, n = 500\n');
fprintf(fileID, '*******************************************\n');

testx = Ex1l10m100n500test(:, 1:(end-1));
testy = Ex1l10m100n500test(:, end);

% Perception
fprintf(fileID, '\nPerception Experiment\n');
[w, theta, mistake] = Perceptron(20, x1, y1);
[n, accuracy] = mistakeCalculator(w, testx, theta, testy);
fprintf(fileID, 'Mistake numbers is %d and accuracy is %f\n', [n, accuracy]);

% Perception with Margin
fprintf(fileID, '\nPerception with Margin Experiment\n');
for i = 1:length(eta_perceptionMargin)
    eta = eta_perceptionMargin(i);
    fprintf(fileID, 'When eta is %f:\n', eta);
    [w, theta, mistake] = PerceptronMargin(20, x1, y1, eta);
    [n, accuracy] = mistakeCalculator(w, testx, theta, testy);
    fprintf(fileID, 'Mistake numbers is %d and accuracy is %f\n', [n, accuracy]);
end

% Winnow
fprintf(fileID, '\nWinnow Experiment\n');
for i = 1:length(alpha_winnow)
    alpha = alpha_winnow(i);
    fprintf(fileID, 'When alpha is %f:\n', alpha);
    [w, theta, mistake] = Winnow(20, x1, y1, alpha);
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
        [w, theta, mistake] = WinnowMargin(20, x1, y1, alpha, gamma);
        [n, accuracy] = mistakeCalculator(w, testx, theta, testy);
        fprintf(fileID, 'Mistake numbers is %d and accuracy is %f\n', [n, accuracy]);
    end
end

% AdaGrad
fprintf(fileID, '\nAdaGrad Experiment\n');
for i = 1:length(eta_AdaGrad)
    eta = eta_AdaGrad(i);
    fprintf(fileID, 'When eta is %f:\n', eta);
    [w, theta, mistake] = AdaGrad(20, x1, y1, eta);
    [n, accuracy] = mistakeCalculator(w, testx, theta, testy);
    fprintf(fileID, 'Mistake numbers is %d and accuracy is %f\n', [n, accuracy]);
end

%% Experiment for l = 10, m = 100, n = 1000, N = 50000
fprintf(fileID, '\n*******************************************\n');
fprintf(fileID, 'Experiment with l = 10, m = 100, n = 1000\n');
fprintf(fileID, '*******************************************\n');

testx = Ex1l10m100n1000test(:, 1:(end-1));
testy = Ex1l10m100n1000test(:, end);

% Perception
fprintf(fileID, '\nPerception Experiment\n');
[w, theta, mistake] = Perceptron(20, x2, y2);
[n, accuracy] = mistakeCalculator(w, testx, theta, testy);
fprintf(fileID, 'Mistake numbers is %d and accuracy is %f\n', [n, accuracy]);

% Perception with Margin
fprintf(fileID, '\nPerception with Margin Experiment\n');
for i = 1:length(eta_perceptionMargin)
    eta = eta_perceptionMargin(i);
    fprintf(fileID, 'When eta is %f:\n', eta);
    [w, theta, mistake] = PerceptronMargin(20, x2, y2, eta);
    [n, accuracy] = mistakeCalculator(w, testx, theta, testy);
    fprintf(fileID, 'Mistake numbers is %d and accuracy is %f\n', [n, accuracy]);
end

% Winnow
fprintf(fileID, '\nWinnow Experiment\n');
for i = 1:length(alpha_winnow)
    alpha = alpha_winnow(i);
    fprintf(fileID, 'When alpha is %f:\n', alpha);
    [w, theta, mistake] = Winnow(20, x2, y2, alpha);
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
        [w, theta, mistake] = WinnowMargin(20, x2, y2, alpha, gamma);
        [n, accuracy] = mistakeCalculator(w, testx, theta, testy);
        fprintf(fileID, 'Mistake numbers is %d and accuracy is %f\n', [n, accuracy]);
    end
end

% AdaGrad
fprintf(fileID, '\nAdaGrad Experiment\n');
for i = 1:length(eta_AdaGrad)
    eta = eta_AdaGrad(i);
    fprintf(fileID, 'When eta is %f:\n', eta);
    [w, theta, mistake] = AdaGrad(20, x2, y2, eta);
    [n, accuracy] = mistakeCalculator(w, testx, theta, testy);
    fprintf(fileID, 'Mistake numbers is %d and accuracy is %f\n', [n, accuracy]);
end

fclose(fileID);

%% plot 1
iternation = 1;
datax = Ex1l10m100n500(:, 1:(end-1));
datay = Ex1l10m100n500(:, end);
[w, theta, num_mistakes1] = Perceptron(iternation, datax, datay);
[w, theta, num_mistakes2] = PerceptronMargin(iternation, datax, datay, 0.005);
[w, theta, num_mistakes3] = Winnow(iternation, datax, datay, 1.1);
[w, theta, num_mistakes4] = WinnowMargin(iternation, datax, datay, 1.1, 2.0);
[w, theta, num_mistakes5] = AdaGrad(iternation, datax, datay, 0.25);

figure(1)
x = 1:50000.*iternation;
plot(x, num_mistakes1, 'LineWidth',2); hold on;
plot(x, num_mistakes2, '--', 'LineWidth',2); hold on;
plot(x, num_mistakes3, '-', 'LineWidth',2); hold on;
plot(x, num_mistakes4, '-.', 'LineWidth',2); hold on;
plot(x, num_mistakes5, ':', 'LineWidth',2);
legend('Perception', 'Perception with Margin', 'Winnow', 'Winnow with Margin', ...
'AdaGrad', 'Location','northwest');
title('Mistakes numbers when l=10, m=100, n=500 with 50,000 examples');
saveas(gcf, './result/Ex1mistake1.png')

%% plot 2
iternation = 5;
datax = Ex1l10m100n500(:, 1:(end-1));
datay = Ex1l10m100n500(:, end);
[w, theta, num_mistakes1] = Perceptron(iternation, datax, datay);
[w, theta, num_mistakes2] = PerceptronMargin(iternation, datax, datay, 0.005);
[w, theta, num_mistakes3] = Winnow(iternation, datax, datay, 1.1);
[w, theta, num_mistakes4] = WinnowMargin(iternation, datax, datay, 1.1, 2.0);
[w, theta, num_mistakes5] = AdaGrad(iternation, datax, datay, 0.25);

figure(2)
x = 1:50000.*iternation;
plot(x, num_mistakes1, 'LineWidth',2); hold on;
plot(x, num_mistakes2, '--', 'LineWidth',2); hold on;
plot(x, num_mistakes3, '-', 'LineWidth',2); hold on;
plot(x, num_mistakes4, '-.', 'LineWidth',2); hold on;
plot(x, num_mistakes5, ':', 'LineWidth',2);
legend('Perception', 'Perception with Margin', 'Winnow', 'Winnow with Margin', ...
'AdaGrad', 'Location','northwest');
title('Mistakes numbers when l=10, m=100, n=500 with 250,000 examples');
saveas(gcf, './result/Ex1mistake2.png')
 
%% plot 3  
iternation = 1;
datax = Ex1l10m100n1000(:, 1:(end-1));
datay = Ex1l10m100n1000(:, end);
[w, theta, num_mistakes1] = Perceptron(iternation, datax, datay);
[w, theta, num_mistakes2] = PerceptronMargin(iternation, datax, datay, 0.03);
[w, theta, num_mistakes3] = Winnow(iternation, datax, datay, 1.1);
[w, theta, num_mistakes4] = WinnowMargin(iternation, datax, datay, 1.1, 0.04);
[w, theta, num_mistakes5] = AdaGrad(iternation, datax, datay, 0.25);

figure(3)
x = 1:50000.*iternation;
plot(x, num_mistakes1, 'LineWidth',2); hold on;
plot(x, num_mistakes2, '--', 'LineWidth',2); hold on;
plot(x, num_mistakes3, '-', 'LineWidth',2); hold on;
plot(x, num_mistakes4, '-.', 'LineWidth',2); hold on;
plot(x, num_mistakes5, ':', 'LineWidth',2);
legend('Perception', 'Perception with Margin', 'Winnow', 'Winnow with Margin', ...
'AdaGrad', 'Location','northwest');
title('Mistakes numbers when l=10, m=100, n=1000 with 50,000 examples');
saveas(gcf, './result/Ex1mistake3.png')  

%% plot 4   
iternation = 5;
datax = Ex1l10m100n1000(:, 1:(end-1));
datay = Ex1l10m100n1000(:, end);
[w, theta, num_mistakes1] = Perceptron(iternation, datax, datay);
[w, theta, num_mistakes2] = PerceptronMargin(iternation, datax, datay, 0.03);
[w, theta, num_mistakes3] = Winnow(iternation, datax, datay, 1.1);
[w, theta, num_mistakes4] = WinnowMargin(iternation, datax, datay, 1.1, 0.04);
[w, theta, num_mistakes5] = AdaGrad(iternation, datax, datay, 0.25);

figure(4)
x = 1:50000.*iternation;
plot(x, num_mistakes1, 'LineWidth',2); hold on;
plot(x, num_mistakes2, '--', 'LineWidth',2); hold on;
plot(x, num_mistakes3, '-', 'LineWidth',2); hold on;
plot(x, num_mistakes4, '-.', 'LineWidth',2); hold on;
plot(x, num_mistakes5, ':', 'LineWidth',2);
legend('Perception', 'Perception with Margin', 'Winnow', 'Winnow with Margin', ...
'AdaGrad', 'Location','northwest');
title('Mistakes numbers when l=10, m=100, n=1000 with 250,000 examples');
saveas(gcf, './result/Ex1mistake4.png')   
    