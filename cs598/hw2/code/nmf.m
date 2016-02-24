function [W, H] = nmf(X, delta, R)

% self-defined NMF analysis function
% input including the original signal X
% the threshold delta to control the accuracy
% the R means the dimension of W and H

[rows, cols] = size(X);
W = rand(rows, R) + 10;
W0 = W;
H = max(pinv(W)*X, 0);
H0 = H;
a = 1;
b = 1;
i = 0;

while ((a > delta) || (b > delta))
    W = max(X*pinv(H), 0);
    H = max(pinv(W)*X, 0);
    
    Wdiff = W - W0;
    Hdiff = H - H0;
    
    a = max(abs(Wdiff(:)));
    b = max(abs(Hdiff(:)));
    
    H0 = H;
    W0 = W;
    i = i + 1;
    if (i > 10000)
        break;
    end
end

end
