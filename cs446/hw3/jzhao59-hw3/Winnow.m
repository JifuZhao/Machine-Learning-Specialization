% Winnow analysis function

function [w, theta, mistakesnumber] = Winnow(iternation, x, y, alpha)
    [rows, cols] = size(x);
    w = ones(cols, 1);
    theta = -cols;
    N = 0;
    mistakesnumber = zeros(iternation*rows, 1);
    for i = 1:iternation
        for j = 1:rows
            if y(j).*(x(j, :)*w + theta) <= 0
                N = N + 1;
                for k = 1:cols
                    w(k) = w(k) * alpha^(y(j)*x(j, k));
                end
            end
            mistakesnumber((i-1)*rows  + j, 1) = N;
        end
    end    

end