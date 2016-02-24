%% create swiss roll data

% % refer to the internet
N = 500; % number of points considered
t = rand(1,N);
t = sort(2*pi*sqrt(t))'; 

z = 1*pi*rand(N,1); 
x = (t+.1).*cos(t);
y = (t+.1).*sin(t);
data = [x,y,z]; 

dlmwrite('swillRollData.txt', data')
