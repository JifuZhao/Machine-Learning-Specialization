% Lecture 3, Slide 21

clear all

X1 = zeros(32);
X2 = zeros(32);
X3 = zeros(32);
X4 = zeros(32);

for j = 1:32
    for k = 1:32
        X1(j, k) = sin(3*(k-1)*2*pi/32);
        X2(j, k) = sin(5*(j-1)*2*pi/32);
        X3(j, k) = sin(3*(j-1 + k-1)*2*pi/32);
        X4(j, k) = sin(3*(j-1 + k-1)*2*pi/32) + sin(6*(k-1)*2*pi/32 + 8*(j-1)*2*pi/32);
    end
end

% Using self-defined function dftMatrix()
F = dftMatrix(32);
one = ones(32);
Y1 = abs(F*X1*F)/16;
Y2 = abs(F*X2*F)/16;
Y3 = abs(F*X3*F)/16;
Y4 = abs(F*X4*F)/16;

% To maker sure that every element in X1,X2,X3,X4 belongs to (0,1)
X1 = (1+X1)/2;
X2 = (1+X2)/2;
X3 = (1+X3)/2;
X4 = (1+X4)/2;

% Plot the figures
figure
subplot(2, 4, 1)
imshow(X1)
title('sin(3x)');
axis on;
subplot(2, 4, 2)
imshow(Y1)
title('2D DFT');
axis on;
subplot(2, 4, 3)
imshow(X2)
title('sin(5y)');
axis on;
subplot(2, 4, 4)
imshow(Y2)
title('2D DFT');
axis on;
subplot(2, 4, 5)
imshow(X3)
title('sin(3(x+y))');
axis on;
subplot(2, 4, 6)
imshow(Y3)
title('2D DFT');
axis on;
subplot(2, 4, 7)
imshow(X4)
title('sin(3(x+y))+sin(6x+8y)', 'FontSize',8);
axis on;
subplot(2, 4, 8)
imshow(Y4)
title('2D DFT');
axis on;

colormap(flipud(colormap(bone)))
saveas(gcf, '21Self.png')
