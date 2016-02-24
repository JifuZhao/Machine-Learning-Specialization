% Lecture 6, Slide 51

clear all; 

%% Reading mp4 from local file
v = VideoReader('hands.mp4');
vFrames = read(v);
markerSize = 1;

%% Build the input data matrix
[length, width, height, frames] = size(vFrames);
video = reshape(vFrames, [], frames);
L = length * width;
video = double(video(1:L, :));
[rows, cols] = size(video);

%% PCA analysis
[V, D] = eigs(video*video'/cols);
Up = V(:, 1:3);
Sp = D(1:3, 1:3);
Wp = Up';
% Wp = Sp^-0.5*Up';

%%
P1 = reshape(Wp(1, :), [length, width]);
P2 = reshape(Wp(2, :), [length, width]);
P3 = reshape(Wp(3, :), [length, width]);

coeffp = Wp * video;
for i = 1:3
    coeffp(i, :) = (coeffp(i, :) - min(coeffp(i, :))) / (max(coeffp(i, :)) - min(coeffp(i, :)));
end

% plot the figure
figure 
colormap(gray)
subplot(2, 3, 1)
imagesc(P1); axis image; axis off;
subplot(2, 3, 2)
imagesc(P2); axis image; axis off;
subplot(2, 3, 3)
imagesc(P3); axis image; axis off;

subplot(2, 3, [4, 5, 6])
plot(coeffp(1, :)/2+2); hold on
plot(2.25*ones(1, 123), '.', 'MarkerSize', markerSize); hold on
plot(coeffp(2, :)/2+1); hold on
plot(1.25*ones(1, 123), '.', 'MarkerSize', markerSize); hold on
plot(coeffp(3, :)/2); xlim([0, 123])
plot(0.25*ones(1, 123), '.', 'MarkerSize', markerSize); hold on

ha = axes('Position',[0 0 1 1],'Box','off','Visible','off','Units','normalized', 'clipping', 'off');
text(0.5, 1,'\bf PCA Analysis','HorizontalAlignment', 'center','VerticalAlignment', 'top');

saveas(gcf, '6s51.png')
%% ICA analysis
Wp2 = Wp;
video2 = Wp2 * video;
Wi = ica(video2, 0.001);
Y = (Wi*Wp2)';

% recreate the image
I1 = reshape(Y(:, 1), [length, width]);
I2 = reshape(Y(:, 2), [length, width]);
I3 = reshape(Y(:, 3), [length, width]);

coeffi = Y' * video;
for i = 1:3
    coeffi(i, :) = (coeffi(i, :) - min(coeffi(i, :))) / (max(coeffi(i, :)) - min(coeffi(i, :)));
end

% plot the figure
figure
colormap(gray)
subplot(2, 3, 1)
imagesc(I1); axis image; axis off;
subplot(2, 3, 2)
imagesc(I2); axis image; axis off;
subplot(2, 3, 3)
imagesc(I3); axis image; axis off;

subplot(2, 3, [4, 5, 6])
plot(coeffi(1, :)/2+2); hold on
plot(2.25*ones(1, 123), '.', 'MarkerSize', markerSize); hold on
plot(coeffi(2, :)/2+1); hold on
plot(1.25*ones(1, 123), '.', 'MarkerSize', markerSize); hold on
plot(coeffi(3, :)/2); xlim([0, 123])
plot(0.25*ones(1, 123), '.', 'MarkerSize', markerSize); hold on

ha = axes('Position',[0 0 1 1],'Box','off','Visible','off','Units','normalized', 'clipping', 'off');
text(0.5, 1,'\bf ICA Analysis','HorizontalAlignment', 'center','VerticalAlignment', 'top');

saveas(gcf, '6s52.png')
%% NMF analysis
[W, H] = nmf(video, 0.01, 2);

N1 = reshape(W(:, 1), [length, width]);
N2 = reshape(W(:, 2), [length, width]);

coeffn = H;
for i = 1:2
    coeffn(i, :) = (coeffn(i, :) - min(coeffn(i, :))) / (max(coeffn(i, :)) - min(coeffn(i, :)));
end

% Plot the figure
figure
colormap(gray)
subplot(2, 2, 1)
imagesc(N1); axis image; axis off;
subplot(2, 2, 2)
imagesc(N2); axis image; axis off;

subplot(2, 2, [3, 4])
plot(coeffn(1, :)/2); hold on
plot(0.25*ones(1, 123), '.', 'MarkerSize', markerSize); hold on
plot(coeffn(2, :)/2 + 0.5); hold on; xlim([0, 123])
plot(0.75*ones(1, 123), '.', 'MarkerSize', markerSize); hold on

ha = axes('Position',[0 0 1 1],'Box','off','Visible','off','Units','normalized', 'clipping', 'off');
text(0.5, 1,'\bf NMF Analysis','HorizontalAlignment', 'center','VerticalAlignment', 'top');

saveas(gcf, '6s54.png')
