% Lecture 3, Slide 46

clear all;

% Read the original data from Source/46.jpg
img = imread('46.jpg');
img = imresize(img, [174, 205]);
[y, x, z] = size(img);

% Generate the input image with some noise
ran1 = uint8(randi(50, 174, 205));

for i = 1:20:y
    for j = 1:20:x
        ran1(i, j) = 0;
        ran1(i+1, j) = 0;
        ran1(i, j+1) = 0;
        ran1(i+1, j+1) = 0;
    end
end

img2 = zeros(y, x, z);
img4 = zeros(y, x, z);

imgr = img(:,:,1);
imgg = img(:,:,2);
imgb = img(:,:,3);
imgr_4 = 0.5*imgr;
imgg_4 = 0.5*imgg;
imgb_4 = 0.5*imgb;

for i = 1:205
    imgr_4(:,i) = 2*exp(-(i-103)^2/5000)*imgr_4(:,i);
    imgg_4(:,i) = 2*exp(-(i-103)^2/5000)*imgg_4(:,i);
    imgb_4(:,i) = 2*exp(-(i-103)^2/5000)*imgb_4(:,i);
end

img1 = img;
img2 = uint8(cat(3, imgr+ran1, imgg+ran1, imgb));
img3 = img;
img4 = uint8(cat(3, imgr_4, imgg_4, imgb_4));

% Dimension information of convolution matrix
N1 = 12;
M1 = 12;
N2 = 3;
M2 = 3;
N3 = 10;
M3 = 20;
N4 = 0;
M4 = 20;

C1 = zeros(2*N1+1);
C2 = zeros(2*N2+1);
C3 = zeros(2*N3+1, 2*M3+1);
C4 = zeros(2*N4+1, 2*M4+1);

% Convolution matrix 1
for i = 1:(2*N1+1)
    for j = 1:(2*N1+1)
        C1(i, j) = exp(-(i-(N1+1))^2/30)*exp(-(j-(N1+1))^2/30);
    end
end

Out1 = uint8(convn(img1, C1/sum(C1(:)), 'same'));

% Convolution matrix 2
for i = 1:(2*N2+1)
    for j = 1:(2*N2+1)
        C2(i, j) = exp(-(i-(N2+1))^2/2)*exp(-(j-(N2+1))^2/2);
    end
end

Out2 = uint8(convn(img2, C2/sum(C2(:)), 'same'));

% Convolution matrix 3
C3 = zeros(2*N3, 2*M3);
for i = 1:2*N3
    C3(i, i) = 1;
    C3(i, 2*M3 - i) = 1;
end

Out3 = uint8(convn(img, C3/sum(C3(:)), 'same'));

% Convolution matrix 4
C4 = -0.65*ones(1, 41);
C4(1, 21) = 40;
C4 = C4/40;
    
Out4 = uint8(convn(img4, C4/sum(C4(:)), 'same'));

figure
subplot(2, 6, 1)
imshow(img1)
subplot(2, 6, 2)
imshow(C1)
subplot(2, 6, 3)
imshow(Out1)

subplot(2, 6, 4)
imshow(img2)
subplot(2, 6, 5)
imshow(C2)
subplot(2, 6, 6)
imshow(Out2)

subplot(2, 6, 7)
imshow(img3)
subplot(2, 6, 8)
imshow(C3)
subplot(2, 6, 9)
imshow(Out3)

subplot(2, 6, 10)
imshow(img4)
subplot(2, 6, 11)
plot(C4, 'o')
axis square
xlim([0, 41])
subplot(2, 6, 12)
imshow(Out4)

colormap(flipud(colormap('gray')))
saveas(gcf, '46Self.png')