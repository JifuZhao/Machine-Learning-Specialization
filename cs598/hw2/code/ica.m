function  [W] = ica(X, delta)

% self-defined ICA analysis function
% input including the original signal X
% the threshold delta to control the accuracy

[rows, cols] = size(X);

W = eye(rows);
Max = 1;
i = 0;

while Max > delta
    if i > 10000
        break;
    end
    i = i + 1;
    Y = W*X;
    Wlast = W;
    deltaW = (cols.*eye(rows) - (Y).^3*Y')*W./cols;
    deltaW = deltaW/norm(deltaW);
    W = W + 0.001*deltaW;
    Wdiff = W - Wlast;
    Max = max(abs(Wdiff(:)));
end

end