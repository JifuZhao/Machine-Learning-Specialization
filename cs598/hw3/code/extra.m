%% Extra Credit
clear all;

pool_train = double(imread('pool_train.png'))/255;
pool_test = double(imread('pool_test.png'))/255;

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
train_Positive = train(:, positive);
train_Negative = train;
train_Negative(:, positive) = [];
train_Negative = train_Negative(:, 1:200);
train_data = [train_Positive, train_Negative];

% PCA analysis
[coeff,score,latent] = pca(train_data');

eigenVector = coeff(:, 1:2);
projection = eigenVector'*train_data;
L = length(positive);

[x, y, z] = self_knn(projection(1:2, 1:L), projection(1:2, (L+1):end), 11, 100);

figure 
contourf(x, y, z, 1); hold on;
plot(projection(1, 1:L), projection(2, 1:L), 'r.'); hold on; title('Extra Classification (knn method)');
plot(projection(1, (L+1):end), projection(2, (L+1):end), 'b.'); axis image

colormap(summer); saveas(gcf, '11extra.png')

% for n = 1:floor(Number/100)
%     figure
%     for i = ((n-1)*100+1):((n-1)*100+100)
%         j = i - (n-1)*100;
%         img = reshape(train(:, i), [delta, delta, 3]); subplot(10, 10, j);
%         imagesc(img); axis image; hold on; axis off;title(num2str(i))
%     end
% end

%% Extra Credit
clear all;

pool_train = double(imread('pool_train.png'))/255;
pool_test = double(imread('pool_test.png'))/255;

red = reshape(pool_train(:, :, 1), [], 1);
green = reshape(pool_train(:, :, 2), [], 1);
blue = reshape(pool_train(:, :, 3), [], 1);
figure
plot3(red, green, blue, '.')
xlabel('Red'); ylabel('Green'); zlabel('Blue');

red2 = reshape(pool_test(:, :, 1), [], 1);
green2 = reshape(pool_test(:, :, 2), [], 1);
blue2 = reshape(pool_test(:, :, 3), [], 1);
figure
plot3(red2, green2, blue2, '.')
xlabel('Red'); ylabel('Green'); zlabel('Blue');


%% Extra Credit
clear all;

pool_train = double(imread('pool_train.png'))/255;
pool_test = double(imread('pool_test.png'))/255;

delta = 8;
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
train_Positive = train(:, positive);
train_Negative = train;
train_Negative(:, positive) = [];
train_Negative = train_Negative(:, 1:500);
%%
for i = 1:length(train_Positive(1, :))
    red = train_Positive(1:delta, i);
    green = train_Positive((delta+1):(2*delta), i);
    blue = train_Positive((2*delta+1):(3*delta), i);
    plot3(red, green, blue, 'b.'); hold on
end

for i = 1:length(train_Negative(1, :))
    red = train_Negative(1:delta, i);
    green = train_Negative((delta+1):(2*delta), i);
    blue = train_Negative((2*delta+1):(3*delta), i);
    plot3(red, green, blue, 'r.'); hold on
end
