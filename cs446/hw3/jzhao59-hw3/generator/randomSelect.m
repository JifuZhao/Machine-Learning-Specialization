% function to randomly select 10% as train data
% and 10% as testing data

function [randomSet1, randomSet2] = randomSelect(x)
  [rows, cols] = size(x);
  N = 0.2 * rows;
  i = 0;
  randomset = zeros(N, 1);
  while true
      num = randi(rows);
      determine = 0;
      for j = 1:N
          if randomset(j) == num
              determine = 1;
          end
      end
      if determine == 0
          i = i + 1;
          randomset(i) = num;
      end
      if i == N
          break
      end
  end
  
  randomSet1 = randomset(1:0.5*N);
  randomSet2 = randomset((0.5*N+1):end);
  
end