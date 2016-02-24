% Perception with margin analysis function

function [w, theta, N, R] = PerceptronMargin(iternation, x, y, eta, threshold)
    [rows, cols] = size(x);
    w = zeros(cols, 1);
    theta = 0;
    gamma = 1;
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
                R = 0;  %
                N = N + 1;
            else
                R = R + 1;
            end
            if y(j).*(x(j, :)*w + theta) <= gamma
                w = w + eta.*y(j)*x(j, :)';
                theta = theta + eta.*y(j);
            end
        end
    end

end