import os
import cv2
import torch
import numpy as np
import pandas as pd
from time import time
from sklearn import decomposition
from torchvision.transforms import transforms
from gabor_filter import GaborFilters

if __name__ == '__main__':
    data_name = 'livevqc'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    if data_name == 'konvid1k':
        data_path = '/mnt/lustre/lliao/Dataset/KoNViD_1k/KoNViD_1k_videos/'
    elif data_name == 'livevqc':
        data_path = '/mnt/lustre/lliao/Dataset/LIVE-VQC/Video'
    else:
        raise NotImplementedError

    width = 480
    height = 270

    feat_path = './features'
    save_path = os.path.join(feat_path, data_name + str(width))
    if not os.path.exists(save_path): os.makedirs(save_path)
    meta_data = pd.read_csv(
        os.path.join(feat_path, data_name + '_metadata.csv'))
    video_num = len(meta_data)

    scale = 6
    orientations = 8
    kernel_size = 39
    row_downsample = 4
    column_downsample = 4

    trasform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((height, width))])

    

    for vn in range(video_num):
        if data_name == 'konvid1k':
            video_name = os.path.join(data_path,
                                      '{}.mp4'.format(meta_data.flickr_id[vn]))
        elif data_name == 'livevqc':
            video_name = os.path.join(data_path, meta_data.File[vn])

        video_capture = cv2.VideoCapture(video_name)
        frame_num = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        v1_features = torch.zeros(
            frame_num,
            (scale * orientations * round(width / column_downsample) *
             round(height / row_downsample)))

        start_time = time()
        count = 0
        while True:
            success, frame = video_capture.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = trasform(frame)
            frame_imag = torch.zeros(frame.size())
            frame = torch.stack((frame, frame_imag), 3)
            frame = torch.view_as_complex(frame)
            frame = frame[None, :, :, :]
            frame = frame.to(device)

            v1_features[count, :] = gb(frame).detach().cpu()

            count += 1

        v1_features = torch.nan_to_num(v1_features)
        v1_features = v1_features.numpy()
        pca = decomposition.PCA()
        v1_features = pca.fit_transform(v1_features)

        end_time = time()
        print('Video {}, {}s elapsed running in {}'.format(
            vn, end_time - start_time, device))

        np.save(
            os.path.join(save_path,
                         os.path.split(video_name)[-1] + '.npy'), v1_features)
