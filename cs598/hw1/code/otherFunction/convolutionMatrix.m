% This function is for l3s46
% It's function is to separate the original picture into 3 dimensions
% Then enlarge the image for the convolution process

function [Out] = convolutionMatrix(img, C, y, x, N0, M0)

img1 = img(:, :, 1);
img2 = img(:, :, 2);
img3 = img(:, :, 3);

N = y + 2*N0;
M = x + 2*M0;

red = zeros(N, M);
green = zeros(N, M);
blue = zeros(N, M);

for i = (N0+1):(N0+y)
    for j = (M0+1):(M0+x)
        red(i, j) = img1(i-N0, j-M0);
        green(i, j) = img2(i-N0, j-M0);
        blue(i, j) = img3(i-N0, j-M0);
    end
end

Y_r = conv2(red, C/sum(C(:)));
Y_g = conv2(green, C/sum(C(:)));
Y_b = conv2(blue, C/sum(C(:)));
Y_r = Y_r((2*N0+1):(2*N0+y), (2*M0+1):(2*M0+x));
Y_g = Y_g((2*N0+1):(2*N0+y), (2*M0+1):(2*M0+x));
Y_b = Y_b((2*N0+1):(2*N0+y), (2*M0+1):(2*M0+x));

Out = cat(3, uint8(Y_r), uint8(Y_g), uint8(Y_b));

end
