% Lecture 7, Slide 36

clear all;

%% reading data from local file
data = matfile('cities.mat');
city = data.D;
[rows, cols] = size(city);

%% calculate the location
N = rows;
X = city;
I = ones(N, 1);
D = diag(X'*X)*I' + I*diag(X'*X)' - 2.*X'*X;
S = -0.5.*(D - D*I*I'./N - I*I'*D./N + I*I'*D*I*I'./N.^2);
[V D] = eig(S);
Y = D^0.5*V';

Loc = Y(1:3, :)./20000;

%% Plot the figure
figure;
subplot(1, 2, 1)
imagesc(city)
colormap(flipud(bone)); colorbar;
set(gca, 'XTickLabel', data.cities)
axis off; axis image; set(gca,'YDir','normal')
title('City distances (mi)')

subplot(1, 2, 2)
scatter3(Loc(1, :), Loc(2, :), Loc(3, :))
text(Loc(1, :)+0.1, Loc(2, :)+0.1, Loc(3, :)+0.1, data.cities)
view([-1, 2, 1]); grid on; axis image
 
saveas(gcf, '7s36.png')
