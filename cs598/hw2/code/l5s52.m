% Lecture 4, Slide 52

clear all;

%% Reading data from local file
data = matfile('faces.mat');
faceOriginal = data.X;

%% select 10*10 pixel patches
r1 = randi([1, 21], 1, 600);
r2 = randi([1, 17], 1, 600);

for i = 1:600
    img = reshape(faceOriginal(:, i), [30, 26]);
    row = r1(i);
    col = r2(i);
    img = img(row:row+9, col:col+9);
    face2(:,i) = img(:);
end

[rows, cols] = size(face2);

%% make sure that the mean of the input is zero
Average = zeros(rows, 1);

for i = 1:rows
    Average(i) = sum(face2(i, :))/cols;
end

for i = 1:rows
    for j = 1:cols
        face(i, j) = face2(i, j) - Average(i);
    end
end
        
[U, S, V] = svd(cov(face'));

%% plot the result
figure
for i = 0:4
    for j = 1:10
        subplot(5, 10, i*10 + j);
        % Recreate the eigen faces
        img = reshape(U(:,i*10 + j), [10, 10]);
        imagesc(img);
        axis off; axis image;
    end
end
colormap(bone)

saveas(gcf, '5s52.png')
