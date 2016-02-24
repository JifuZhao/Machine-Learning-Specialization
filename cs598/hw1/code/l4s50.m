% Lecture 4, Slide 50

clear all;

X = 400;
Y = 300;
Z = 120;
L = (X-Z)/2;
T = (Y-Z)/2;

% Generate the Input picture M1 and M2
M1 = ones(Y, X);
M2 = zeros(Y, X);

for x = L+1-10:(L+Z+10)
    for y = T+1:(T+Z)
        M1(y, x) = 0.25;
        M2(y, x) = 1;
    end
end

for x = (L+1-10):(L+20-10)
    for y = (T+1):(T+20)
        if (x-151)^2 + (y-111)^2 >400
            M1(y, x) = 1;
            M2(y, x) = 0;
        end
    end
    for y = (T+Z-20):(T+Z)
        if (x-151)^2 + (y-190)^2 >400
            M1(y, x) = 1;
            M2(y, x) = 0;
        end
    end
end

for x = (L+Z-10):(L+Z+10)
    for y = (T+1):(T+20)
        if (x-250)^2 + (y-111)^2 >400
            M1(y, x) = 1;
            M2(y, x) = 0;
        end
    end
    for y = (T+Z-20):(T+Z)
        if (x-250)^2 + (y-190)^2 >400
            M1(y, x) = 1;
            M2(y, x) = 0;
        end
    end
end

% Generate the convolution matrix f
f = [-1, -1, -1; -1, 8, -1; -1, -1, -1]/8;

% Convolution
Out1 = conv2(M1, f, 'same');
Out2 = conv2(M2, f, 'same');

% Plot the final result
figure
subplot(2, 2, 1);
imshow(M1)
title('Input');

subplot(2, 2, 2);
imagesc(Out1.^2)
axis off
title('Energy of edge detector');

subplot(2, 2, 3);
imagesc(M2)
axis off
title('Input');

subplot(2, 2, 4);
imagesc(Out2.^2)
axis off
title('Energy of edge detector');

colormap(gray)
saveas(gcf, '50Self.png')