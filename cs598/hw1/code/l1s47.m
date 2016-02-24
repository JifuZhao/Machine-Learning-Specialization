% Lecture 1, Slide 47

clear all;
% Reading image from local file Source/self.jpg
img = imread('47.jpg');

% Resize the picture to a small size
img = imresize(img, 0.1);
[x, y, z] = size(img);

% Generate the color, horizontal and vertical matrix
C = diag([1, 0, 1]);
H = eye(y);

V1 = zeros(x);
for i = 1:x
    V1(i, x+1-i) = 1;
end

X1 = [0, 1; 1, 0];
X2 = eye(x/2);
V2 = kron(X1, X2);
V = V1 * V2;

img1 = img(:, :, 1);
img2 = img(:, :, 2);
img3 = img(:, :, 3);
img_1d = double([img1(:); img2(:); img3(:)]);
img_1d2 = kron(kron(C, H), V)*img_1d;
img_3d = uint8(reshape(img_1d2, [x, y, z]));

figure
subplot(1,2,1);
imshow(img);
title('Original picture');

subplot(1,2,2);
imshow(img_3d);
title('Picture after processing');

saveas(gcf, '47Self.jpg')