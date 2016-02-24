function [x, y, z] = self_knn(r1, r2, K, N)

    minx = min([r1(1, :), r2(1, :)]); miny = min([r1(2, :), r2(2, :)]);
    maxx = max([r1(1, :), r2(1, :)]); maxy = max([r1(2, :), r2(2, :)]);
    x = linspace(minx, maxx, N); y = linspace(miny, maxy, N);
    z = zeros(length(y), length(x));
    L1 = length(r1(1, :)); 
    L2 = length(r2(1, :));
    distance = zeros(2, (L1 + L2));
    for i = 1:length(y)
        for j = 1:length(x)
            for k = 1:L1
                distance(1, k) = (x(j) - r1(1, k))^2 + (y(i) -r1(2, k)).^2;
                distance(2, k) = 1;
            end
            for k = 1:L2
                distance(1, L1 + k) = (x(j) - r2(1, k))^2 + (y(i) -r2(2, k)).^2;
                distance(2, L1 + k) = -1;
            end
            distance = sortrows(distance')';
            Sum = sum(distance(2, 1:K));
            if Sum >= 0
                z(i, j) = 1;
            end
        end
    end   
end