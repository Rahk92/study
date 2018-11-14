from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
from skimage.measure import compare_psnr, compare_ssim
import numpy as np
import PIL.Image as Image
import scipy.misc as misc
import os, glob

test_path = "D:\Data\sr\B100"
results_path = './results/'

def predict_model():
    SRCNN = Sequential()
    SRCNN.add(Conv2D(64, (9, 9), padding='same', activation='relu', input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(32, (1, 1), padding='same', activation='relu'))
    SRCNN.add(Conv2D(1, (5, 5), padding='same', activation='linear'))

    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN

def predict():
    srcnn_model = predict_model()
    srcnn_model.load_weights("200_epoch_weights.h5")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    psnr = []
    file_list = glob.glob('{}/*.bmp'.format(test_path))
    for file in file_list:
        img = misc.imread(file, mode='YCbCr')
        h, w, c = img.shape
        img_y = img[:h-(h%3), :w-(w%3), 0]

        img_test = misc.imresize(img_y, 1.0/3.0, 'bicubic')
        img_test = misc.imresize(img_test, 3.0, 'bicubic')

        x_test = img_test.reshape(1, img_test.shape[0], img_test.shape[1], 1);
        y_predict = srcnn_model.predict(x_test, batch_size=1)

        img_out = y_predict.reshape(img_test.shape)
        img_out = np.clip(img_out, 0, 255)

        filename = os.path.basename(file)
        filename = filename.split('.')[0]
        psnr_test = compare_psnr(img_y, img_out)
        psnr_bicubic = compare_psnr(img_y, img_test)
        print('{0:} : bicubic PSNR = {1:.2f}, SRCNN PSNR = {2:.2f}'.format(filename, psnr_bicubic, psnr_test))
        psnr.append(psnr_test)

        img_gt = Image.fromarray((img_y).astype('uint8'))
        img_gt.save(results_path + filename +'_groundtruth.png')
        img_test = Image.fromarray((img_test).astype('uint8'))
        img_test.save(results_path + filename +'_bicubic.png')
        img_out = Image.fromarray((img_out).astype('uint8'))
        img_out.save(results_path + filename + '_srcnn.png')

    psnr_avg = sum(psnr)/len(psnr)
    print('Average PSNR = {0:.2f}'.format(psnr_avg))


if __name__ =="__main__":
    predict()