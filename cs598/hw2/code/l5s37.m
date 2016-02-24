% Lecture 5, Slide 37

clear all;

%% reading data from local file
data = matfile('faces.mat');
faceOriginal = data.X;

[rows, cols] = size(faceOriginal);

%% make sure that the mean of the input is zero
Average = zeros(rows, 1);

for i = 1:rows
    Average(i) = sum(faceOriginal(i, :))/cols;
end

for i = 1:rows
    for j = 1:cols
        face(i, j) = faceOriginal(i, j) - Average(i);
    end
end
        
%% SVD to cov(face')
[U, S, V] = svd(cov(face'));

%% plot the result
figure
for i = 0:5
    for j = 1:6
        subplot(6, 6, i*6 + j);
        % recreate the eigen faces
        img = reshape(U(:,i*6 + j), [30, 26]);
        imagesc(img);
        axis off; axis image;
    end
end
colormap(gray)

saveas(gcf, '5s37.png')
