% Perception analysis function

function [w, theta, mistakesnumber] = PerceptronModified(iternation, x, y, eta, gammaP, gammaN)
    [rows, cols] = size(x);
    w = zeros(cols, 1);
    theta = 0;
    N = 0;
    mistakesnumber = zeros(iternation*rows, 1);
    for i = 1:iternation
        for j = 1:rows
            if y(j) > 0
                if y(j).*(x(j, :)*w + theta) <= gammaP
                    N = N + 1;
                    w = w + eta.*y(j)*x(j, :)';
                    theta = theta + eta.*y(j);
                end
            else
                if y(j).*(x(j, :)*w + theta) <= gammaN
                    N = N + 1;
                    w = w + eta.*y(j)*x(j, :)';
                    theta = theta + eta.*y(j);
                end
            end
            mistakesnumber((i-1)*rows  + j, 1) = N;
        end
    end

end