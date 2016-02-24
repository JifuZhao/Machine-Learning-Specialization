% extra problem

clear all;

%% reading data from local file
input = matfile('one.mat');
input = input.one; 
[l, w] = size(input{1, 1});

L = size(input, 2);

data = zeros(l*w, L);
for i = 1:L
    data(:, i) = reshape(input{i}, [l*w, 1]);
end

[rows, cols] = size(data);

%% manifold structure analysis
N = 20;
sigma = 100;
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
colormap(gray)
deltaX = 0.015;
deltaY = deltaX;
P = Y((end-2):(end-1), :);
No = cols;
plot(P(1, :), P(2, :), '.');
hold on;
for i = 1:No
    image = reshape(data(:, i), [l, w]);
    x = P(1, i);
    y = P(2, i);
    imagesc([x, x+deltaX], [y, y-deltaY], image);
    hold on;
end
axis image

saveas(gcf, '7extraCredit.png')