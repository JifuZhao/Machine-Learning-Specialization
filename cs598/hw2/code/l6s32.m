% Lecture 6, Slide 32

clear all;

%% reading data from local file
data = matfile('faces.mat');
faceOriginal = double(data.X);

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
[Up, Sp, Vp] = svd(cov(face'));
% Wp = (Sp^-0.5)*Up';
Wp = Up';
Zp = Wp * face;

PCA_face = Wp';

%% ICA analysis
Wp2 = Wp(1:16, :);
Zp = Wp2*face;
Wi = ica(Zp, 0.0001);
ICA_face = (Wi*Wp2)';

%% plot the result
figure
for i = 0:3
    for j = 1:4
        subplot(4, 8, i*8 + j);
        % recreate the eigen faces
        img = reshape(PCA_face(:,i*4 + j), [30, 26]);
        imagesc(img); axis off; axis image;
    end
end

for i = 0:3
    for j = 1:4
        subplot(4, 8, i*8+j+4);
        % recreate the ICA faces
        img = reshape(ICA_face(:,i*4 + j), [30, 26]);
        imagesc(img); axis off; axis image;
    end
end
colormap(gray)
ha = axes('Position',[0 0 1 1],'Box','off','Visible','off','Units','normalized', 'clipping', 'off');
text(0.5, 1,'\bf Left is Eigenfaces,            Right is ICA faces','HorizontalAlignment', ...
    'center','VerticalAlignment', 'top');

saveas(gcf, '6s32.png')
