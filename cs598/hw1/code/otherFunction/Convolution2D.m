% img_new = zeros(x+N, y+N);
% for i = M+1:x+M
%     for j= M+1:y+M
%         img_new(i, j) = img(i-M, j-M);
%     end
% end

% img1 = conv2(img_new, Q*M_0);
% img2 = conv2(img_new, Q*M_90);
% img3 = conv2(img_new, Q*M_45);
% img4 = conv2(img_new, Q*M_135);
% 
% img1 = img1((N+1):(x+N), (N+1):(y+N));
% img2 = img2((N+1):(x+N), (N+1):(y+N));
% img3 = img3((N+1):(x+N), (N+1):(y+N));
% img4 = img4((N+1):(x+N), (N+1):(y+N));