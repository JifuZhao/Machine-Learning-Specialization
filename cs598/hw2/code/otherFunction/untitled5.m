N = 1000;
t = (3*pi/2)*(1+2*rand(1,N)); 
s = 21*rand(1,N); 
X=[t.*cos(t);s;t.*sin(t)];

scatter3(X(1, :), X(2, :), X(3, :))

% theta = 0:0.05:2*pi;
% r = 0.5*theta;
% L = length(theta);
% z0 = 0.2:0.1:0.8;
% lz = length(z0);
% z = z0'*ones(1, L);
% 
% X = zeros(lz*L, 1);
% Y = zeros(lz*L, 1);
% Z = zeros(lz*L, 1);
% for i = 1:lz
%     X(1+(i-1)*L:i*L, 1) = (r.*cos(-theta))';
%     Y(1+(i-1)*L:i*L, 1) = (r.*sin(-theta))';
%     Z(1+(i-1)*L:i*L, 1) = (z(i, :))';
% end
% 
% X = X + 0.1*rand(lz*L, 1);
% Y = Y + 0.1*rand(lz*L, 1);
% Z = Z + 0.1*rand(lz*L, 1);