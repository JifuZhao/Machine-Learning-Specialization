% Perception with margin analysis function

function [w, theta, mistakesnumber] = PerceptronMargin(iternation, x, y, eta)
    [rows, cols] = size(x);
    w = zeros(cols, 1);
    theta = 0;
    gamma = 1;
    N = 0;
    mistakesnumber = zeros(iternation*rows, 1);
    for i = 1:iternation
        for j = 1:rows
            if y(j).*(x(j, :)*w + theta) <= 0
                N = N + 1;
            end
            if y(j).*(x(j, :)*w + theta) <= gamma
                w = w + eta.*y(j)*x(j, :)';
                theta = theta + eta.*y(j);
            end
            mistakesnumber((i-1)*rows  + j, 1) = N;
        end
    end

end