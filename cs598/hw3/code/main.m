%% Lecture 8, Slide 30
clear all;

[pa, FsPa] = audioread('pa.wav');
[shot_origin, FsShot] = audioread('shot.wav');
lpa = length(pa);
lshot = length(shot_origin);
timePa = (0:(lpa-1))./FsPa;

shot = zeros(lshot, 1);
for i = 1:lshot
    shot(i) = shot_origin(lshot - i + 1);
end

Input = zeros(lpa + lshot - 1, 1);
Filter = zeros(lpa + lshot -1, 1);

Input(1:lpa) = pa;
Filter(1:lshot) = shot;

transform = fft(Input) .* fft(Filter);
output = ifft(transform);
output = output((lshot/2 + 1):(lshot/2 + lpa));
output = abs(output);
N = 1000;
meanfilter = ones(N, 1)/N;
output = conv(output, meanfilter, 'same');
output = exp(output);
output = 2*(output - min(output(:))) ./ (max(output(:)) - min(output(:)));

% plot the result
figure
plot(timePa, pa); hold on;
plot(timePa, output); xlim([0, timePa(end)])
xlabel('Time (sec)'); legend('Input audio', 'Filter output energy'); 
title('Input scene audio/matches');

saveas(gcf, '8s30.png')

%% Lecture 8, Slide 46
clear all;
load one.mat
Template = one;
[rows, cols] = size(Template);
Input1 = zeros(rows+1, 2.*cols+1);
Input1(1:rows, 1:cols) = Template;
Input2 = max(Input1(:)).*ones(rows+1, 2.*cols+1) - Input1;

% calculate the gradient image
deltaX = [1, -1]; deltaY = [1; -1]; S = [1; 1];
gradientTemplate = conv2(conv2(Template, deltaX, 'same'), S, 'same') ...
    + conv2(conv2(Template, deltaY, 'same'), S', 'same');
gradientInput1 = conv2(conv2(Input1, deltaX, 'same'), S, 'same') ...
    + conv2(conv2(Input1, deltaY, 'same'), S', 'same');
gradientInput2 = conv2(conv2(Input2, deltaX, 'same'), S, 'same') ...
    + conv2(conv2(Input2, deltaY, 'same'), S', 'same');
gradientInput1 = gradientInput1(1:rows, 1:2.*cols);
gradientInput2 = gradientInput2(1:rows, 1:2.*cols);

% Convolution
inverseTemplate = zeros(rows, cols);
for i = 1:rows
    for j = 1:cols
       inverseTemplate(i, j) = gradientTemplate(rows-i+1, cols-j+1);
    end
end
Output1 = conv2(gradientInput1, inverseTemplate, 'same');
Output2 = conv2(gradientInput2, inverseTemplate, 'same');

% plot the figure
figure
subplot(3, 3, 1); imagesc(Template); colorbar; axis image; title('Template')
subplot(3, 3, 2); imagesc(Input1); colorbar; axis image; title('Input')
subplot(3, 3, 3); imagesc(Input2); colorbar; axis image; title('Input')
subplot(3, 3, 4); imagesc(gradientTemplate); colorbar; axis image; title('Template gradient')
subplot(3, 3, 5); imagesc(gradientInput1); colorbar; axis image; title('Input gradient')
subplot(3, 3, 6); imagesc(gradientInput2); colorbar; axis image; title('Input gradient')
subplot(3, 3, 7); imagesc(Output1); colorbar; axis image; title('Matched filtered output')
subplot(3, 3, 9); imagesc(Output2); colorbar; axis image; title('Matched filtered output')
colormap(bone); 

saveas(gcf, '8s46.png')

%% Lecture 9, Slide 32
clear all

mu = [3, 3, 6, 6];
sigma = [1, 1, 1, 1];
r = randn(4, 100);
r(1, :) = sigma(1)*r(1, :) + mu(1);
r(2, :) = sigma(2)*r(2, :) + mu(2);
r(3, :) = sigma(3)*r(3, :) + mu(3);
r(4, :) = sigma(4)*r(4, :) + mu(4);

mean1 = 3; mean2 = 3; mean3 = 6; mean4 = 6;
sigma1 = 1; sigma2 = 1;
mu1 = [mean1; mean2]; mu2 = [mean3; mean4];
w1 = mu1 / sigma1.^2; w2 = mu2 / sigma2.^2;

N = 200;
minx = min([r(1, :), r(3, :)]); maxx = max([r(1, :), r(3, :)]);
miny = min([r(2, :), r(4, :)]); maxy = max([r(2, :), r(4, :)]); 
x = linspace(minx, maxx, N); y = linspace(miny, maxy, N);
z = zeros(length(y), length(x));
for i = 1:length(y)
    for j = 1:length(x)
        X = [x(j); y(i)];
        z1 = w1'*X - mu1'*mu1/(2*sigma1.^2);
        z2 = w2'*X - mu2'*mu2/(2*sigma2.^2);
        if z1 >= z2
            z(i, j) = 1;
        end
    end
end

figure; 
contourf(x, y, z, 1); axis image; hold on; title('Lecture 9, Slide 32');
plot(r(1,:),r(2,:),'o'); hold on; plot(r(3,:),r(4,:),'o'); hold on;

for r = 0:0.5:2
    theta = 0:0.1:2*pi;
    x1 = r*cos(theta) + mu(1);
    y1 = r*sin(theta) + mu(2);
    plot(x1, y1, 'k'); hold on;
    x2 = r*cos(theta) + mu(3);
    y2 = r*sin(theta) + mu(4);
    plot(x2, y2, 'k'); hold on;
end

colormap(summer); saveas(gcf, '9s32.png')

%% Lecture 9, Slide 33
clear all

N = 200;
mu = [0, 0, 20, 20];
sigma = [10, 1, 4, 3];
r = randn(4, 100);
theta = pi/4;
M = [cos(theta), -sin(theta); sin(theta), cos(theta)];
r(1, :) = sigma(1)*r(1, :) + mu(1);
r(2, :) = sigma(2)*r(2, :) + mu(2);
r(3, :) = sigma(3)*r(3, :) + mu(3);
r(4, :) = sigma(4)*r(4, :) + mu(4);
r(1:2, :) = M*r(1:2, :);
cov1 = [sigma(1)^2, 0; 0, sigma(2)^2];
cov2 = [sigma(3)^2, 0; 0, sigma(4)^2];
cov1 = M*cov1*M';
minx = min([r(1, :), r(3, :)]); maxx = max([r(1, :), r(3, :)]);
miny = min([r(2, :), r(4, :)]); maxy = max([r(2, :), r(4, :)]); 
x = linspace(minx, maxx, N); y = linspace(miny, maxy, N);
% z = quadraticClassifier2(mu, cov1, cov2, minx, maxx, miny, maxy, N);
z = quadraticClassifier(r(1:2, :), r(3:4, :), N);

figure
contourf(x, y, z, 1); axis image; hold on; title('Lecture 9, Slide 33');
plot(r(1,:),r(2,:),'.'); hold on; plot(r(3,:),r(4,:),'.'); colormap(summer)

for r = 0:0.5:2
    theta = 0:0.1:2*pi;
    x1 = r*cos(theta)*sigma(1) + mu(1);
    y1 = r*sin(theta)*sigma(2) + mu(2);
    X = M*[x1; y1];
    plot(X(1, :), X(2, :), 'k'); hold on;
    x2 = r*cos(theta)*sigma(3) + mu(3);
    y2 = r*sin(theta)*sigma(4) + mu(4);
    plot(x2, y2, 'k'); hold on;
end

saveas(gcf, '9s33.png')

%% Lecture 9, Slide 34
clear all

N = 200;
mu1 = [0, 0, 8, 8];
sigma1 = [10, 1, 3, 2];
r1 = randn(4, 200);
theta1 = pi/4;
M1 = [cos(theta1), -sin(theta1); sin(theta1), cos(theta1)];
r1(1, :) = sigma1(1)*r1(1, :) + mu1(1);
r1(2, :) = sigma1(2)*r1(2, :) + mu1(2);
r1(3, :) = sigma1(3)*r1(3, :) + mu1(3);
r1(4, :) = sigma1(4)*r1(4, :) + mu1(4);
r1(1:2, :) = M1*r1(1:2, :);
cov1 = [sigma1(1)^2, 0; 0, sigma1(2)^2];
cov2 = [sigma1(3)^2, 0; 0, sigma1(4)^2];
cov1 = M1*cov1*M1';
minx1 = min([r1(1, :), r1(3, :)]); maxx1 = max([r1(1, :), r1(3, :)]);
miny1 = min([r1(2, :), r1(4, :)]); maxy1 = max([r1(2, :), r1(4, :)]); 

mu2 = [0, 0, 10, 10];
sigma2 = [10, 9, 3, 2];
r2 = randn(4, 100);
theta2 = pi/4;
r2(1, :) = sigma2(1)*r2(1, :) + mu2(1);
r2(2, :) = sigma2(2)*r2(2, :) + mu2(2);
r2(3, :) = sigma2(3)*r2(3, :) + mu2(3);
r2(4, :) = sigma2(4)*r2(4, :) + mu2(4);
cov3 = [sigma2(1)^2, 0; 0, sigma2(2)^2];
cov4 = [sigma2(3)^2, 0; 0, sigma2(4)^2];
minx2 = min([r2(1, :), r2(3, :)]); maxx2 = max([r2(1, :), r2(3, :)]);
miny2 = min([r2(2, :), r2(4, :)]); maxy2 = max([r2(2, :), r2(4, :)]); 

% z1 = quadraticClassifier2(mu1, cov1, cov2, minx1, maxx1, miny1, maxy1, N);
% z2 = quadraticClassifier2(mu2, cov3, cov4, minx2, maxx2, miny2, maxy2, N);
z1 = quadraticClassifier(r1(1:2, :), r1(3:4, :), N);
z2 = quadraticClassifier(r2(1:2, :), r2(3:4, :), N);


x1 = linspace(minx1, maxx1, N); y1 = linspace(miny1, maxy1, N);
x2 = linspace(minx2, maxx2, N); y2 = linspace(miny2, maxy2, N);

figure
subplot(1, 2, 1)
contourf(x1, y1, z1, 1); axis image; hold on; title('Lecture 9, Slide 34(1)');
plot(r1(1,:), r1(2,:), '.'); hold on; plot(r1(3,:), r1(4,:), '.'); hold on;
for r = 0:0.5:2
    theta = 0:0.1:2*pi;
    xx1 = r*cos(theta)*sigma1(1) + mu1(1);
    yy1 = r*sin(theta)*sigma1(2) + mu1(2);
    X = M1*[xx1; yy1];
    plot(X(1, :), X(2, :), 'k'); hold on;
    xx2 = r*cos(theta)*sigma1(3) + mu1(3);
    yy2 = r*sin(theta)*sigma1(4) + mu1(4);
    plot(xx2, yy2, 'k'); hold on;
end

subplot(1, 2, 2)
contourf(x2, y2, z2, 1); axis image; hold on; title('Lecture 9, Slide 34(2)');
plot(r2(1,:), r2(2,:), '.'); hold on; plot(r2(3,:), r2(4,:), '.'); hold on;
for r = 0:0.5:2
    theta = 0:0.1:2*pi;
    xx3 = r*cos(theta)*sigma2(1) + mu2(1);
    yy3 = r*sin(theta)*sigma2(2) + mu2(2);
    plot(xx3, yy3, 'k'); hold on;
    xx4 = r*cos(theta)*sigma2(3) + mu2(3);
    yy4 = r*sin(theta)*sigma2(4) + mu2(4);
    plot(xx4, yy4, 'k'); hold on;
end

colormap(summer); saveas(gcf, '9s34.png')

%% Lecture 9, Slide 53
clear all;
load face2.mat

face_original = double(XX)/255;
[rows, cols] = size(face_original);
M = mean(face_original, 2);
M = M * ones(1, cols);
face = face_original - M;

[U, S, V] = svd(cov(face'));
W = U(:, 1:2)';
Z = W*face;
for i = 1:length(g)
    if g(i) == 0
        g(i) = -1;
    end
end
temp = Z(2, :); Z(2, :) = Z(1, :); Z(1, :) = temp;

X = [Z; ones(1, length(Z(1, :)))];
w = g*pinv(X);
w = w';
minx = min(Z(1, :)); miny = min(Z(2, :));
maxx = max(Z(1, :)); maxy = max(Z(2, :));
figure
subplot(2, 1, 1)
j = 0; k = 0;
smile = zeros(2, 400); nonsmile = zeros(2, 400);
for i = 1:length(Z(1, :))
    if g(i) == 1
        plot(Z(1, i), Z(2, i), 'b.'); hold on;
        j = j + 1;
        smile(:, j) = Z(:, i);
    else
        plot(Z(1, i), Z(2, i), 'r.'); hold on;
        k = k + 1;
        nonsmile(:, k) = Z(:, i);
    end
end
smile = smile(:, 1:j); nonsmile = nonsmile(:, 1:k);

legend('Smilers', 'Non-Smilers'); grid on;
x = minx:0.1:maxx; y = -(w(1)*x + w(3))/w(2);
plot(x,y);  hold on; xlim([minx, maxx]); ylim([miny, maxy]);

z = quadraticClassifier(smile, nonsmile, 200);
subplot(2, 1, 2)
x = linspace(minx, maxx, 200);
y = linspace(miny, maxy, 200);
contourf(x, y, z); hold on; title('Quadratic Classifier')
for i = 1:length(Z(1, :))
    if g(i) == 1
        plot(Z(1, i), Z(2, i), 'b.'); hold on;
    else
        plot(Z(1, i), Z(2, i), 'r.'); hold on;
    end
end

colormap(summer); saveas(gcf, '9s53.png')

%% Lecture 10, Slide 39
clear all;

N = 20; sigma = 1;
mu = [-3, -3, 3, 3]; r = randn(4, N);
r(1, :) = sigma*r(1, :) + mu(1);
r(2, :) = sigma*r(2, :) + mu(2);
r(3, :) = sigma*r(3, :) + mu(3);
r(4, :) = sigma*r(4, :) + mu(4);
minx = min([r(1, :), r(3, :)]); maxx = max([r(1, :), r(3, :)]);
x = linspace(minx, maxx, N);

X = [r(1:2, :), r(3:4, :)]; X = [X ; ones(1, 2*N)]';
Y = [-ones(N, 1); ones(N, 1)];
A(:, 1) = -Y.*X(:, 1); A(:, 2) = -Y.*X(:, 2); A(:, 3) = -Y.*X(:, 3);
b = -ones(2*N, 1); f = zeros(3, 1);
H = [1, 0, 0; 0, 1, 0; 0, 0, 0];
[W,fval,exitflag,output,lambda] = quadprog(H, f, A, b);
w = (W(1).^2 + W(2).^2).^0.5;
distance = (W(1:2, 1)'*[r(1:2, :), r(3:4 ,:)] + W(3))/w;

figure
subplot(1, 2, 1)
% y1 = -(W(1)*x + W(3)+w)/W(2); y_0 = -(W(1)*x + W(3))/W(2); y_1 = -(W(1)*x + W(3)-w)/W(2);
y1 = -(W(1)*x + W(3)+1)/W(2); y_0 = -(W(1)*x + W(3))/W(2); y_1 = -(W(1)*x + W(3)-1)/W(2);
plot(r(1,:), r(2,:), '.'); hold on; plot(r(3,:), r(4,:), '.'); hold on
plot(x, y_1); hold on; plot(x, y_0); hold on; plot(x, y1); hold on;
axis image; title('Classification result'); grid on;
subplot(1, 2, 2)
stem(distance*5, 'o'); title('SVM output'); axis image; grid on;

saveas(gcf, '10s39.png')


%% Lecture 10, Slide 42
clear all;

N = 30; sigma = 3.0;
mu = [-3, -3, 3, 3]; r = randn(4, N);
r(1, :) = sigma*r(1, :) + mu(1);
r(2, :) = sigma*r(2, :) + mu(2);
r(3, :) = sigma*r(3, :) + mu(3);
r(4, :) = sigma*r(4, :) + mu(4);
minx = min([r(1, :), r(3, :)]); maxx = max([r(1, :), r(3, :)]);
x = linspace(minx, maxx, N);

X = [r(1:2, :), r(3:4, :)]; X = [X ; ones(1, 2*N)]';
Y = [-ones(N, 1); ones(N, 1)];
A(:, 1) = -Y.*X(:, 1); A(:, 2) = -Y.*X(:, 2); A(:, 3) = -Y.*X(:, 3);
b = -ones(2*N, 1); f = zeros(3, 1);
H = [1, 0, 0; 0, 1, 0; 0, 0, 0];
W = quadprog(H, f, A, b);
w = (W(1).^2 + W(2).^2).^0.5;
distance = (W(1:2, 1)'*[r(1:2, :), r(3:4 ,:)] + W(3))/w;

figure
subplot(1, 2, 1)
% y1 = -(W(1)*x + W(3)+w)/W(2); y_0 = -(W(1)*x + W(3))/W(2); y_1 = -(W(1)*x + W(3)-w)/W(2);
y1 = -(W(1)*x + W(3)+1)/W(2); y_0 = -(W(1)*x + W(3))/W(2); y_1 = -(W(1)*x + W(3)-1)/W(2);
plot(r(1,:), r(2,:), '.'); hold on; plot(r(3,:), r(4,:), '.'); hold on
plot(x, y_1); hold on; plot(x, y_0); hold on; plot(x, y1); hold on;
axis image; title('Classification result'); grid on;

subplot(1, 2, 2)
stem(distance*5, 'o'); title('Classifier output'); axis image; grid on;

saveas(gcf, '10s42.png')

%% Lecture 11, Slide 24
clear all;

N = 100; sigma1 = 0.5; sigma2 = 1; sigma3 = 2;
mu = [-1, -1, 1, 1]; r = randn(4, N);
r1(1, :) = sigma1*r(1, :) + mu(1);
r1(2, :) = sigma1*r(2, :) + mu(2);
r1(3, :) = sigma1*r(3, :) + mu(3);
r1(4, :) = sigma1*r(4, :) + mu(4);
r2(1, :) = sigma2*r(1, :) + mu(1);
r2(2, :) = sigma2*r(2, :) + mu(2);
r2(3, :) = sigma2*r(3, :) + mu(3);
r2(4, :) = sigma2*r(4, :) + mu(4);
r3(1, :) = sigma3*r(1, :) + mu(1);
r3(2, :) = sigma3*r(2, :) + mu(2);
r3(3, :) = sigma3*r(3, :) + mu(3);
r3(4, :) = sigma3*r(4, :) + mu(4);

figure
subplot(3, 6, 1)
[x, y, z] = self_knn(r1(1:2, :), r1(3:4, :), 1, N); contourf(x, y, z); hold on;
plot(r1(1,:), r1(2,:), '.'); hold on; plot(r1(3,:), r1(4,:), '.'); title('K = 1')
subplot(3, 6, 2)
[x, y, z] = self_knn(r1(1:2, :), r1(3:4, :), 3, N); contourf(x, y, z); hold on;
plot(r1(1,:), r1(2,:), '.'); hold on; plot(r1(3,:), r1(4,:), '.'); title('K = 3')
subplot(3, 6, 7)
[x, y, z] = self_knn(r1(1:2, :), r1(3:4, :), 5, N); contourf(x, y, z); hold on;
plot(r1(1,:), r1(2,:), '.'); hold on; plot(r1(3,:), r1(4,:), '.'); title('K = 5')
subplot(3, 6, 8)
[x, y, z] = self_knn(r1(1:2, :), r1(3:4, :), 11, N); contourf(x, y, z); hold on;
plot(r1(1,:), r1(2,:), '.'); hold on; plot(r1(3,:), r1(4,:), '.'); title('K = 11')
subplot(3, 6, 13)
[x, y, z] = self_knn(r1(1:2, :), r1(3:4, :), 21, N); contourf(x, y, z); hold on;
plot(r1(1,:), r1(2,:), '.'); hold on; plot(r1(3,:), r1(4,:), '.'); title('K = 21')
subplot(3, 6, 14); 
[x, y, z] = self_knn(r1(1:2, :), r1(3:4, :), 100, N); contourf(x, y, z); hold on;
plot(r1(1,:), r1(2,:), '.'); hold on; plot(r1(3,:), r1(4,:), '.'); title('K = 100')

subplot(3, 6, 3)
[x, y, z] = self_knn(r2(1:2, :), r2(3:4, :), 1, N); contourf(x, y, z); hold on;
plot(r2(1,:), r2(2,:), '.'); hold on; plot(r2(3,:), r2(4,:), '.'); title('K = 1')
subplot(3, 6, 4)
[x, y, z] = self_knn(r2(1:2, :), r2(3:4, :), 3, N); contourf(x, y, z); hold on;
plot(r2(1,:), r2(2,:), '.'); hold on; plot(r2(3,:), r2(4,:), '.'); title('K = 3')
subplot(3, 6, 9)
[x, y, z] = self_knn(r2(1:2, :), r2(3:4, :), 5, N); contourf(x, y, z); hold on;
plot(r2(1,:), r2(2,:), '.'); hold on; plot(r2(3,:), r2(4,:), '.'); title('K = 5')
subplot(3, 6, 10)
[x, y, z] = self_knn(r2(1:2, :), r2(3:4, :), 11, N); contourf(x, y, z); hold on;
plot(r2(1,:), r2(2,:), '.'); hold on; plot(r2(3,:), r2(4,:), '.'); title('K = 11')
subplot(3, 6, 15)
[x, y, z] = self_knn(r2(1:2, :), r2(3:4, :), 21, N); contourf(x, y, z); hold on;
plot(r2(1,:), r2(2,:), '.'); hold on; plot(r2(3,:), r2(4,:), '.'); title('K = 21')
subplot(3, 6, 16); 
[x, y, z] = self_knn(r2(1:2, :), r2(3:4, :), 100, N); contourf(x, y, z); hold on;
plot(r2(1,:), r2(2,:), '.'); hold on; plot(r2(3,:), r2(4,:), '.'); title('K = 100')

subplot(3, 6, 5)
[x, y, z] = self_knn(r3(1:2, :), r3(3:4, :), 1, N); contourf(x, y, z); hold on;
plot(r3(1,:), r3(2,:), '.'); hold on; plot(r3(3,:), r3(4,:), '.'); title('K = 1')
subplot(3, 6, 6)
[x, y, z] = self_knn(r3(1:2, :), r3(3:4, :), 3, N); contourf(x, y, z); hold on;
plot(r3(1,:), r3(2,:), '.'); hold on; plot(r3(3,:), r3(4,:), '.'); title('K = 3')
subplot(3, 6, 11)
[x, y, z] = self_knn(r3(1:2, :), r3(3:4, :), 5, N); contourf(x, y, z); hold on;
plot(r3(1,:), r3(2,:), '.'); hold on; plot(r3(3,:), r3(4,:), '.'); title('K = 5')
subplot(3, 6, 12)
[x, y, z] = self_knn(r3(1:2, :), r3(3:4, :), 11, N); contourf(x, y, z); hold on;
plot(r3(1,:), r3(2,:), '.'); hold on; plot(r3(3,:), r3(4,:), '.'); title('K = 11')
subplot(3, 6, 17)
[x, y, z] = self_knn(r3(1:2, :), r3(3:4, :), 21, N); contourf(x, y, z); hold on;
plot(r3(1,:), r3(2,:), '.'); hold on; plot(r3(3,:), r3(4,:), '.'); title('K = 21')
subplot(3, 6, 18); 
[x, y, z] = self_knn(r3(1:2, :), r3(3:4, :), 100, N); contourf(x, y, z); hold on;
plot(r3(1,:), r3(2,:), '.'); hold on; plot(r3(3,:), r3(4,:), '.'); title('K = 100')

colormap(summer); saveas(gcf, '11s24.png')

%% Extra Credit
clear all;

pool_train = double(imread('pool_train.png'))/255;
pool_test = double(imread('pool_test.png'))/255;

y1 = [1 35 90 145 155 162 204 195 268 315 363 497 425 456];
y2 = [11 45 110 155 170 186 217 220 280 325 373 509 445 473];

x1 = [255 675 745 635 405 435 665 300 205 528 263 13 560 765];
x2 = [265 685 765 655 420 470 680 310 215 538 273 24 572 780];
j = 0;
pool = zeros(300, 10000);
for i = 1:length(y1)
    img = pool_train(y1(i):y2(i), x1(i):x2(i), :);
%     subplot(4, 5, i)
%     imagesc(img); hold on
    for y = 10:length(img(:, 1))
        for x = 10:(length(img(1, :, :)))
            j = j+1;
            pool(:, j) = reshape(img((y-9):y, (x-9):x, :), [], 1);
        end
    end
end

pool = pool(:, 1:j);

delta = 10;
[L, W, H] = size(pool_train);
M = floor(L/delta); N = floor(W/delta);
train = zeros(delta*delta*3, M*N);
for i = 1:M
    for j = 1:N
        y = (i-1)*delta+1; x = (j-1)*delta+1;
        train(:, (i-1)*N+j) = reshape(pool_train(y:(y+delta-1), x:(x+delta-1), :), [], 1);
    end
end
Number = length(train(1, :));

load -ASCII info.txt
positive = info';
train_Positive = pool;
train_Negative = train;
train_Negative(:, positive) = [];
train_Negative = train_Negative(:, 1:6000);
train = [train_Positive, train_Negative];
Lpositive = length(train_Positive(1, :));
Lnegative = length(train_Negative(1, :));
label = zeros(Lpositive+Lnegative, 1);
label(1:Lpositive, 1) = 1;

[L2, W2, H2] = size(pool_test);
M2 = floor(L2/delta); N2 = floor(W2/delta);

test = zeros(delta*delta*3, M2*N2);
for i = 1:M2
    for j = 1:N2
        y = (i-1)*delta+1; x = (j-1)*delta+1;
        test(:, (i-1)*N2+j) = reshape(pool_test(y:(y+delta-1), x:(x+delta-1), :), [], 1);
    end
end

SVMModel = fitcsvm(train',label);
[Label,score] = predict(SVMModel,test');

figure
subplot(1, 2, 1)
imagesc(pool_test); axis image;title('Original map');
subplot(1, 2, 2)
imagesc(pool_test); axis image;title({'SVM method', 'With some error'}); hold on; 

j = 0;
for i = 1:length(Label)
    if Label(i) == 1
        j = j+1;
        y = delta*floor(i/N2);
        x = delta*rem(i, N2);
        plot(x, y, 'r*'); hold on;
    end
end

saveas(gcf, '11sExtraProblem.png')