% Lecture 6, Slide 22

clear all;

%% generating the original data through Gaussian function
num = 1000;
r1 = rand(1, num);
r2 = rand(1, num);
r1 = 2*r1 - 1;
r2 = 2*r2 - 1;
x = 2.*r1 + r2;
y = r1 + r2;
M = [x; y];

% make sure that the data are zero centered
average1 = sum(M(1,:))/num;
average2 = sum(M(2,:))/num;
M(1, :) = M(1, :) - average1;
M(2, :) = M(2, :) - average2;

%% PCA analysis
[U, S, V] = svd(cov(M'));
W = (S^-0.5)*U';
Z = W*M;

%% ICA analysis
W = ica(M, 0.0000001);
Y = W*M;

%% plot the figure
figure
subplot(1, 3, 1)
plot(M(1,:), M(2,:), '.')
axis image; grid on; title('Input')

subplot(1, 3, 2)
plot(Z(1,:), Z(2,:), '.')
axis image; grid on;  title('PCA')

subplot(1, 3, 3)
plot(Y(1,:), Y(2, :), '.')
axis image; grid on;  title('ICA')

saveas(gcf, '6s22.png')
