% Lecture 5, Slide 25

clear all;

%% generating the original data through Gaussian function
num = 1000;
mu1 = 0;
mu2 = 0;
sigma1 = 5;
sigma2 = 0.5;
x0 = mu1 + sigma1.*randn(num, 1);
y0 = mu2 + sigma2.*randn(num, 1);
x = cos(pi/8).*x0 - sin(pi/8).*y0;
y = sin(pi/8).*x0 + cos(pi/8).*y0;
M = [x'; y'];

%% make sure that the data are zero centered
average1 = sum(M(1,:))/num;
average2 = sum(M(2,:))/num;
M(1, :) = M(1, :) - average1;
M(2, :) = M(2, :) - average2;

%% PCA analysis
[U, S, V] = svd(cov(M'));
W = (S^-0.5)*U';
Z = W*M;

%% plot the figure
figure
subplot(1, 2, 1)
plot(M(1,:), M(2,:), '.')
axis image; hold on;
plot(10*[U(1,1), -U(1,1)], 10*[U(2,1), -U(2,1)], 'r-', 'linewidth', 2)
hold on;
plot([U(1,2), -U(1,2)], [U(2,2), -U(2,2)], 'r-', 'linewidth', 2)
title('Input Data')

subplot(1, 2, 2)
plot(Z(1,:), Z(2,:), '.')
axis image; title('Transformed Data (feature weights)')

saveas(gcf, '5s25.png')