% Lecture 7, Slide 51

clear all;

%% read the input 
data = dlmread('swillRollData.txt');
M = length(data);
cmap = jet(M);

%% manifold structure analysis
N = 7;
sigma = 10;

[rows, cols] = size(data);
W = zeros(cols, cols);
for i = 1:cols
    for j = 1:cols
        diff = data(:, i) - data(:, j);
        W(i, j) = exp(-diff'*diff./sigma);
        W(j, j) = 0;
    end
end

for i = 1:cols
    s = sort(W(i, :), 'descend');
    threshold = s(N);
    for j = 1:cols
        if W(i, j) < threshold
            W(i, j) = 0;
        end
    end
end

D = diag(ones(1, cols)* W);
L = W - D;
Ln = pinv(D^0.5)*L*pinv(D^0.5);
[Z, S, V] = svd(Ln);
Y = Z'*D^0.5;

% Plot the figure
figure
subplot(1,2, 1)
scatter3(data(1,:), data(2,:), data(3,:), 20, cmap); axis image
subplot(1, 2, 2)
Point = Y(end-N: end-1, :);
scatter(Point(end, :), Point(end-1, :), 20, cmap); axis image; grid on

saveas(gcf, '7s51.png')
