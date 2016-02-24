% Winnow with margin analysis function

function [w, theta, N, R] = WinnowMargin(iternation, x, y, alpha, gamma, threshold)
    [rows, cols] = size(x);
    w = ones(cols, 1);
    theta = -cols;
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
            if y(j).*(x(j, :)*w + theta) <= 0
                N = N + 1;
                R = 0;
            else
                R = R + 1;
            end
            if y(j).*(x(j, :)*w + theta) <= gamma
                for k = 1:cols
                    w(k) = w(k) * alpha^(y(j)*x(j, k));
                end
            end
        end
    end     

end