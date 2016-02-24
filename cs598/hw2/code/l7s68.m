% Lecture 7, Slide 

clear all;

%% Reading mp4 from local file
v = VideoReader('hotlips.mp4');
vFrames = double(read(v))/255;

%% Build the input data matrix
[length, width, height, frames] = size(vFrames);
video = reshape(vFrames, [], frames);
L = length * width;
video = video(1:L, :);
[rows, cols] = size(video);

%% manifold structure analysis
data = video;

%% manifold structure analysis
N = 8;
sigma = 100;

W = zeros(cols, cols);
for i = 1:cols
    for j = 1:cols
        diff = data(:, i) - data(:, j);
        W(i, j) = exp(-diff'*diff./sigma);
        W(j, j) = 0;
    end
end
for i = 1:cols
    s = sort(W(i, :), 'descend');
    threshold = s(N);
    for j = 1:cols
        if W(i, j) < threshold
            W(i, j) = 0;
        end
    end
end
%%
D = diag(ones(1, cols)* W);
L = W - D;
Ln = pinv(D^0.5)*L*pinv(D^0.5);
[Z, S, V] = svd(Ln);
Y = Z'*D^0.5;

%% Plot the figure
figure
colormap(gray)
deltaX = 0.05;
deltaY = 128/96*deltaX;
P = Y((end-2):(end-1), :);
No = frames;
plot(P(2, :), P(1, :));
hold on;
for i = 1:No
    image = reshape(video(:, i), [length, width]);
    x = P(2, i);
    y = P(1, i);
    imagesc([x, x+deltaX], [y, y-deltaY], image);
    hold on;
end
axis image

saveas(gcf, '7s68.png')