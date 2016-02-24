% Lecture 6, Slide 45

clear all;

%% generate the input signal X
R = 2;
rows = 100;
cols = 200;
X = zeros(rows, cols);
for i = 5:20
    for j = 1:25
        X(i, j) = 1;
    end
    for j = 51:90
        X(i, j) = 1;
    end
    for j = 131:160
        X(i, j) = 1;
    end
    for j = 186:200
        X(i, j) = 1;
    end
end
for i = 81:90
    for j = 20:40
        X(i, j) = 1;
    end
    for j = 80:120
        X(i, j) = 1;
    end
    for j = 150:180
        X(i, j) = 1;
    end
end

%% calculate the matrix of W and H
[W, H] = nmf(X, 0.0000001, 2);
W = W';

H(1, :) = (H(1, :) - min(H(1, :)))./(max(H(1, :)) - min(H(1, :)));
H(2, :) = (H(2, :) - min(H(2, :)))./(max(H(2, :)) - min(H(2, :)));
W(1, :) = (W(1, :) - min(W(1, :)))./(max(W(1, :)) - min(W(1, :)));
W(2, :) = (W(2, :) - min(W(2, :)))./(max(W(2, :)) - min(W(2, :)));

%% Plot the figure
figure;
subplot(2, 2, 2)
plot(H(1, :)/2)
hold on;
plot(H(2, :)/2+ 1)
ylim([-0.5, 2])

subplot(2, 2, 3)
plot(-W(2, :)/2, 1:rows)
hold on
plot(-W(1, :)/2 + 1, 1:rows)
xlim([-1, 1.5])

subplot(2, 2, 4)
imagesc(X)
colormap(flipud(bone)); set(gca,'Ydir','Normal')

saveas(gcf, '6s45.png')
