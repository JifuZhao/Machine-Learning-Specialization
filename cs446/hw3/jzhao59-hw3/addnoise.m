% The function adds noise in y and x
% Assume that n is the number of features and k is the number of
% instances.
% y: k-by-1 vector, each element can be 1 or -1
% x: k-by-n matrix,
% noise_y_rate: a number between [0,1]
% noise_x_rate: a number between [0,1]
% ------------------------------------
% new_y: k-by-1 vector, each element can be 1 or -1
% new_x: k-by-n matrix,
function [new_y,new_x] = addnoise(y,x,noise_y_rate,noise_x_rate)

  tmp_y = (y + ones(size(y)))/2;
  noise_y = (rand(size(y)) < noise_y_rate);
  new_y = bitxor(tmp_y,noise_y); % add noise on y
  new_y = 2 * new_y - ones(size(y));


  noise_x = (rand(size(x)) < noise_x_rate);
  new_x = bitxor(x,noise_x);% add noise on x

