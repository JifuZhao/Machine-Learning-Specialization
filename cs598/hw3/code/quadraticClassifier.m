function z = quadraticClassifier2(r1, r2, N)
    mean1 = mean(r1(1, :));
    mean2 = mean(r1(2, :));
    mean3 = mean(r2(1, :)); 
    mean4 = mean(r2(2, :));
    covariance1 = cov(r1'); 
    covariance2 = cov(r2'); 
    mu1 = [mean1; mean2];
    mu2 = [mean3; mean4];
    W1 = -0.5*covariance1^-1; 
    W2 = -0.5*covariance2^-1; 
    w1 = covariance1^-1*mu1;
    w2 = covariance2^-1*mu2;
    ww1 = -0.5*mu1'*covariance1^-1*mu1 - 0.5*log(det(covariance1));
    ww2 = -0.5*mu2'*covariance2^-1*mu2 - 0.5*log(det(covariance2));

    minx = min([r1(1, :), r2(1, :)]); miny = min([r1(2, :), r2(2, :)]);
    maxx = max([r1(1, :), r2(1, :)]); maxy = max([r1(2, :), r2(2, :)]);
    x = linspace(minx, maxx, N); y = linspace(miny, maxy, N);
    z = zeros(length(y), length(x));

    for i = 1:length(y)
        for j = 1:length(x)
            X = [x(j); y(i)];
            z1 = X'*W1*X + w1'*X + ww1;
            z2 = X'*W2*X + w2'*X + ww2;
            if z1 >= z2
                z(i, j) = 1;
            end
        end
    end
end