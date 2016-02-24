% Perception analysis function

function [w, theta, mistakesnumber] = Perceptron(iternation, x, y)
    [rows, cols] = size(x);
    w = zeros(cols, 1);
    theta = 0;
    eta = 1.0;
    N = 0;
    mistakesnumber = zeros(iternation*rows, 1);
    for i = 1:iternation
        for j = 1:rows
            if y(j).*(x(j, :)*w + theta) <= 0
                N = N + 1;
                w = w + eta.*y(j)*x(j, :)';
                theta = theta + eta.*y(j);
            end
            mistakesnumber((i-1)*rows + j, 1) = N;
        end
    end

end