clear all
clc

datadir='D:\Data\sr\Set5'; 
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
            label = img_y(y:y+patch_size-1, x:x+patch_size-1);
            patch = img_LR(y:y+patch_size-1, x:x+patch_size-1);
   
            count = count+1;
            label_all(count,:,:,1) = label; 
            patch_all(count,:,:,1) = patch;
            
%             patch = imrotate(img_y(y_coord+1:y_coord+patch_size, x_coord+1:x_coord+patch_size,:), 90);
%             label_all(count,:,:,1) = patch;
%             patch = imrotate(img_LR(y_coord+1:y_coord+patch_size, x_coord+1:x_coord+patch_size), 90);
%             patch_all(count,:,:,1) = patch;
%             count = count+1;
%             
%             patch = imrotate(img_y(y_coord+1:y_coord+patch_size, x_coord+1:x_coord+patch_size,:), 180);
%             label_all(count,:,:,1) = patch;
%             patch = imrotate(img_LR(y_coord+1:y_coord+patch_size, x_coord+1:x_coord+patch_size), 180);
%             patch_all(count,:,:,1) = patch;
%             count = count+1;
%             
%             patch = imrotate(img_y(y_coord+1:y_coord+patch_size, x_coord+1:x_coord+patch_size,:), 270);
%             label_all(count,:,:,1) = patch;
%             patch = imrotate(img_LR(y_coord+1:y_coord+patch_size, x_coord+1:x_coord+patch_size), 270);
%             patch_all(count,:,:,1) = patch;
%             count = count+1;
%             
%             patch = fliplr(imrotate(img_y(y_coord+1:y_coord+patch_size, x_coord+1:x_coord+patch_size,:),90));
%             label_all(count,:,:,1) = patch;
%             patch = fliplr(imrotate(img_LR(y_coord+1:y_coord+patch_size, x_coord+1:x_coord+patch_size,:),90));
%             patch_all(count,:,:,1) = patch;
%             count = count+1;
        end
    end
end
order = randperm(count);
label_all = label_all(order,:,:,1);
patch_all = patch_all(order,:,:,1);

label_name='val_label.h5';
if exist(label_name, 'file')
    fprintf('Warning: replacing existing file %s \n', label_name);
    delete(label_name);
end
h5create(label_name,'/label',size(label_all),'Datatype','uint8');
h5write(label_name,'/label',label_all);

patch_name='val_patch.h5';
if exist(patch_name, 'file')
    fprintf('Warning: replacing existing file %s \n', patch_name);
    delete(patch_name);
end
h5create(patch_name,'/patch',size(patch_all),'Datatype','uint8');
h5write(patch_name,'/patch',patch_all);
    