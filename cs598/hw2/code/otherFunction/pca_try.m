A = rand(100, 50);
[rows, cols] = size(A);
average = zeros(rows, 1);
for i = 1:rows
    average(i, 1) = sum(A(i, :));
end

average = average./cols*ones(1, cols);

A2 = A - average;

[U, S, V] = svd(cov(A2'));

M = U(:, 1:25);

W = M'*A2;

A3 = M*W;

diff = A3 + average - A;

max(abs(diff(:)))/max(A(:))
