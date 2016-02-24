% Lecture 4, Slide 52

clear all;

% Reading image from local file
img = imread('52.png');
img = double(img)/255;
[x, y] = size(img);

%Matrix size
M = 30;
N = 2*M;
Q = 0.01;

M_0 = Gabor(M, 0);
M_90 = Gabor(M, pi/2);
M_45 = Gabor(M, pi/4);
M_135 = Gabor(M, -pi/4);

img1 = conv2(img, Q*M_0, 'same');
img2 = conv2(img, Q*M_90, 'same');
img3 = conv2(img, Q*M_45, 'same');
img4 = conv2(img, Q*M_135, 'same');

% define the font Size = 7
Size = 7;

% Plot the figure
figure;
subplot(2,5,2);
imshow(M_0);
title('0 filter','FontSize', Size)
subplot(2,5,3');
imshow(M_90);
title('90 filter','FontSize', Size)
subplot(2,5,4);
imshow(M_45);
title('45 filter','FontSize', Size)
subplot(2,5,5);
imshow(M_135);
title('-45 filter','FontSize', Size)
subplot(2,5,6);
imshow(img);
title('Input','FontSize', Size)
subplot(2,5,7);
imshow(img1);
title('0 response','FontSize', Size)
subplot(2,5,8);
imshow(img2);
title('90 response','FontSize', Size)
subplot(2,5,9);
imshow(img3);
title('45 response','FontSize', Size)
subplot(2,5,10);
imshow(img4);
title('-45 response','FontSize', Size)

saveas(gcf, '53Self.png')
