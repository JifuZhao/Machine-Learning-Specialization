%This function is going to generate a Gabor filter

function [G] = Gabor(size, theta)
    G = zeros(2*size+1);
    sigma_u = 0.03*size;
    sigma_v = 0.3*size;
    lab = 1;

    for m = -size:size
        for n = -size:size
            u = m*cos(theta) + n*sin(theta);
            v = -m*sin(theta) + n*cos(theta);
            G(n+size+1,m+size+1) = exp(-0.5*((u)^2/sigma_u^2 + (v)^2/sigma_v^2));
        end
    end

end