% function to calculate the numbers of mistakes

function [n, accuracy] = mistakeCalculator(w, x, theta, y)
    
    delta = (x*w + theta).* y;
    n = 0;
    for i = 1:length(delta)
        if delta(i) <= 0
            n = n + 1;
        end
    end
    accuracy = 1 - n / length(delta);
end