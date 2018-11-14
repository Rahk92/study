clear all
clc

datadir='D:\Data\sr\91'; 
count=0;

f_lst=[];
f_lst=[f_lst; dir(fullfile(datadir, '*.png'))];
f_lst=[f_lst; dir(fullfile(datadir, '*.jpg'))];
f_lst=[f_lst; dir(fullfile(datadir, '*.bmp'))];
% f_lst = 3x1 struct

patch_size = 33; %scale로 나눠서 정수여야함
scale = 3;
stride = 14;

num_lst=numel(f_lst); %numel->배열 요소의 개수를 반환
label_all = uint8(zeros(500000, patch_size, patch_size, 1));
patch_all = uint8(zeros(500000, patch_size, patch_size, 1));

for f_iter = 1:num_lst%numel(f_lst)
    f_info=f_lst(f_iter);
    if f_info.name=='.'
        continue;
    end
    f_path = fullfile(datadir,f_info.name);
    img_raw = imread(f_path);
    img_yuv = rgb2yuv(img_raw);
    img_y = img_yuv(:,:,1);
    img_size = size(img_y);
    img_y = img_y(1:img_size(1) - mod(img_size(1),scale),1:img_size(2) - mod(img_size(2),scale),:);
    img_size = size(img_y); 
    height = img_size(1);
    width = img_size(2); 
    
    img_LR = imresize(imresize(img_y,1/scale, 'bicubic'),[height, width], 'bicubic');
    
    for y = 1: stride : height-patch_size+1
        for x = 1 : stride : width-patch_size+1
            count = count+1;
        end
    end
end
count