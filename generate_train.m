
clear all
clc

datadir='D:\Data\sr\91'; 
count=1;
cnt=0;

f_lst=[];
f_lst=[f_lst; dir(fullfile(datadir, '*.png'))];
f_lst=[f_lst; dir(fullfile(datadir, '*.jpg'))];
f_lst=[f_lst; dir(fullfile(datadir, '*.bmp'))];
% f_lst = 3x1 struct

patch_size=32; %scale로 나눠서 정수여야함 64->41
%scale4=4;
%scale2=2;
num_lst=numel(f_lst); %numel->배열 요소의 개수를 반환
label_all=uint8(zeros(500000, patch_size, patch_size, 1)); % gt 10000x32x32x1
patch_all=uint8(zeros(500000, patch_size, patch_size, 1)); % input 10000x32x32x1

for f_iter = 1:num_lst%numel(f_lst)
    f_info=f_lst(f_iter);
    if f_info.name=='.'
        continue;
    end
    f_path=fullfile(datadir,f_info.name);
    img_raw=imread(f_path);
    img_raw=img_raw(:,:,1);
    img_size=size(img_raw); width=img_size(2); height=img_size(1);
    img_raw=img_raw(1:height-mod(height,12),1:width-mod(width,12),:);
    img_size=size(img_raw); width=img_size(2); height=img_size(1);
    
    img_2=imresize(imresize(img_raw,1/2, 'bicubic'),[img_size(1), img_size(2)], 'bicubic');
    %img_3=imresize(imresize(img_raw,1/3, 'bicubic'),[img_size(1), img_size(2)], 'bicubic');
    %img_4=imresize(imresize(img_raw,1/4, 'bicubic'),[img_size(1), img_size(3)], 'bicubic');
    
    img_HR=size(img_raw);
    patch_HR=patch_size; %32
    stride_HR=patch_size; %32
    x_size=(img_HR(2)-patch_HR)/stride_HR+1;
    y_size=(img_HR(1)-patch_HR)/stride_HR+1;
    
    for y=0:y_size-1
        for x=0:x_size-1
            x_coord=x*stride_HR; y_coord=y*stride_HR;
            
            patch=img_raw(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:);
            label_all(count,:,:,1)=patch;
            patch=img_2(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR);
            patch_all(count,:,:,1)=patch;
            count=count+1;
            
            patch=imrotate(img_raw(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:), 90);
            label_all(count,:,:,1)=patch;
            patch=imrotate(img_2(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR), 90);
            patch_all(count,:,:,1)=patch;
            count= count+1;
            
            patch=imrotate(img_raw(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:), 180);
            label_all(count,:,:,1)=patch;
            patch=imrotate(img_2(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR), 180);
            patch_all(count,:,:,1)=patch;
            count= count+1;
            
            patch=imrotate(img_raw(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:), 270);
            label_all(count,:,:,1)=patch;
            patch=imrotate(img_2(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR), 270);
            patch_all(count,:,:,1)=patch;
            count= count+1;
            
            patch=fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),90));
            label_all(count,:,:,1)=patch;
            patch=fliplr(imrotate(img_2(y_coord+1:y_coord+patch_HR, x_coord+1:x_coord+patch_HR,:),90));
            patch_all(count,:,:,1)=patch;
            count=count+1;
        end
    end
    
    cnt=cnt+1;
    if mod(cnt,100)==0
        display(100*cnt/numel(f_lst));display('percent complete');
    end
    
end
label_all=label_all(1:count-1,:,:,:);
patch_all=patch_all(1:count-1,:,:,:);

order=randperm(count-1);
label_all=label_all(order,:,:,:);
patch_all=patch_all(order,:,:,:);

patch_name='label_all_interpolation.h5';
h5create(patch_name,'/label_all',size(label_all),'Datatype','uint8');
h5write(patch_name,'/label_all',label_all);

patch_name='patch_all_interpolation.h5';
h5create(patch_name,'/patch_all',size(patch_all),'Datatype','uint8');
h5write(patch_name,'/patch_all',patch_all);
    