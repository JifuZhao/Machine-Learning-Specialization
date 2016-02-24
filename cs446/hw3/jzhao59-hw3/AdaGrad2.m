% AdaGrad analysis function

function [w, theta, N, R] = AdaGrad(iternation, x, y, eta, threshold)
    [rows, cols] = size(x);
    x = [x, ones(rows, 1)];
    [rows, cols] = size(x);
    w = zeros(cols, 1);
    G = ones(cols, 1);
    N = 0; %
    R = 0; %
    for i = 1:iternation
        if R == threshold  %
            break     %
        end           %
        for j = 1:rows
            if R == threshold %
                break; %
            end        %
            if y(j).*(x(j, :)*w) <= 0
                N = N + 1;
                R = 0;
            else
                R = R + 1;
            end
            if y(j).*(x(j, :)*w) <= 1
                g = -y(j)*x(j, :)';
                G = G + g.^2;
                for k = 1:cols
                    w(k) = w(k) + eta.*y(j).*x(j, k)./(G(k).^0.5);
                end
            end
        end
    end
    theta = w(cols, 1);
    w = w(1:(cols - 1), 1);

end