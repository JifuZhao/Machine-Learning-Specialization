function image = toUint8(img)
% function to transfer image into Uint8

Max = max(img);
Min = min(img);
Len = length(img);

for i = 1:Len
    image(i) = (img(i) - Min)/(Max - Min);
%     image(i) = uint8(255 * (img(i) - Min)/(Max - Min));
end

end