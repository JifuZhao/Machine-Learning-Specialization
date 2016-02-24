% Lecture 5, Slide 39

clear all;

%% Reading data from local file
data = matfile('faces.mat');
faceOriginal = data.X;

[rows, cols] = size(faceOriginal);

[U, S, V] = svd(faceOriginal*faceOriginal'/cols);

W = S^-0.5*U';

Z = W*faceOriginal;

X50 = pinv(W(1:50, :)) * Z(1:50, :);
X10 = pinv(W(1:10, :)) * Z(1:10, :);

face = faceOriginal(:, 11);
face50 = X50(:, 11);
face10 = X10(:, 11);

face = reshape(face, [30, 26]);
pic1 = reshape(face50, [30, 26]);
pic2 = reshape(face10, [30, 26]);
weight1 = Z(1:50, 11);
weight2 = Z(1:10, 11);

%% plot the image
figure;
subplot(2, 3, 1)
imagesc(face)
axis image; axis off; title('Input')
subplot(2, 3, 2)
imagesc(weight1)
axis image; axis off; title('Weights')
subplot(2, 3, 3)
imagesc(pic1)
axis image; axis off; title('Approximation')
subplot(2, 3, 4)
imagesc(face)
axis image; axis off; title('Input')
subplot(2, 3, 5)
imagesc(weight2)
axis image; axis off; title('Weights')
subplot(2, 3, 6)
imagesc(pic1)
axis image; axis off; title('Approximation')
colormap(gray)
 
saveas(gcf, '5s39.png')