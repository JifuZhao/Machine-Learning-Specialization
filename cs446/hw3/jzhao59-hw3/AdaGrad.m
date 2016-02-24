% AdaGrad analysis function

function [w, theta, mistakesnumber] = AdaGrad(iternation, x, y, eta)
    [rows, cols] = size(x);
    x = [x, ones(rows, 1)];
    [rows, cols] = size(x);
    w = zeros(cols, 1);
    G = ones(cols, 1);
    N = 0;
    mistakesnumber = zeros(iternation*rows, 1);
    for i = 1:iternation
        for j = 1:rows
            if y(j).*(x(j, :)*w) <= 0
                N = N + 1;
            end
            if y(j).*(x(j, :)*w) <= 1
                g = -y(j)*x(j, :)';
                G = G + g.^2;
                for k = 1:cols
                    w(k) = w(k) + eta.*y(j).*x(j, k)./(G(k).^0.5);
                end
            end
            mistakesnumber((i-1)*rows  + j, 1) = N;
        end
    end
    
    theta = w(cols, 1);
    w = w(1:(cols - 1), 1);

end