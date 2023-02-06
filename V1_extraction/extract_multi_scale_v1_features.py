import os
import cv2
import torch
import numpy as np
import pandas as pd
from time import time
from sklearn import decomposition
from torchvision.transforms import transforms
from gabor_filter import GaborFilters

# resolutioin: 960*540，480*270，240*135，120*67
# downsample rate: 1.0, 0.5, 0.25, 0.125

if __name__ == '__main__':
    data_name = 'konvid1k'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    if data_name == 'konvid1k':
        data_path = '/mnt/lustre/lliao/Dataset/KoNViD_1k/KoNViD_1k_videos/'
    elif data_name == 'livevqc':
        data_path = '/mnt/lustre/lliao/Dataset/LIVE-VQC/Video'
    else:
        raise NotImplementedError

    feat_path = './features'
    save_path = os.path.join(feat_path, data_name + 'multi_scale')
    if not os.path.exists(save_path): os.makedirs(save_path)
    meta_data = pd.read_csv(
        os.path.join(feat_path, data_name + '_metadata.csv'))
    video_num = len(meta_data)

    width_list = [960, 480, 240, 120]
    height_list = [540, 270, 135, 67]
    #downsample_rate_list = [1.0, 0.5, 0.25, 0.125]

    scale = 5
    orientations = 8
    kernel_size = 19
    row_downsample = 4
    column_downsample = 4

    pca_d = 10

    gb = GaborFilters(scale,
                      orientations, (kernel_size - 1) // 2,
                      row_downsample,
                      column_downsample,
                      device=device)

    for vn in range(video_num):
        start_time = time()

        if data_name == 'konvid1k':
            video_name = os.path.join(data_path,
                                      '{}.mp4'.format(meta_data.flickr_id[vn]))
        elif data_name == 'livevqc':
            video_name = os.path.join(data_path, meta_data.File[vn])

        video_capture = cv2.VideoCapture(video_name)
        frame_num = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        v1_features = []
        transform_list = []
        for i in range(len(width_list)):
            width = width_list[i]
            height = height_list[i]
            v1_features.append(
                torch.zeros(
                    frame_num,
                    (scale * orientations * round(width / column_downsample) *
                     round(height / row_downsample))))
            transform_list.append(
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((height, width))
                ]))

        # for i in range(len(downsample_rate_list)):
            # width = int(video_width * downsample_rate_list[i])
            # height = int(video_height * downsample_rate_list[i])
            # v1_features.append(
                # torch.zeros(
                    # frame_num,
                    # (scale * orientations * round(width / column_downsample) *
                     # round(height / row_downsample))))
            # transform_list.append(
                # transforms.Compose([
                    # transforms.ToTensor(),
                    # transforms.Resize((height, width))
                # ]))

        count = 0
        while True:
            success, frame = video_capture.read()
            if not success:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for i in range(len(width_list)):# + len(downsample_rate_list)):
                frame = transform_list[i](frame_gray)
                frame_imag = torch.zeros(frame.size())
                frame = torch.stack((frame, frame_imag), 3)
                frame = torch.view_as_complex(frame)
                frame = frame[None, :, :, :]
                frame = frame.to(device)

                v1_features[i][count, :] = gb(frame).detach().cpu()

            count += 1

        for i in range(len(width_list)):# + len(downsample_rate_list)):
            v1_features[i] = torch.nan_to_num(v1_features[i])
            v1_features[i] = v1_features[i].numpy()
            pca = decomposition.PCA(pca_d)
            v1_features[i] = pca.fit_transform(v1_features[i])
            np.save(
                os.path.join(
                    save_path,
                    '{}_{}.npy'.format(i, os.path.split(video_name)[-1])),
                v1_features[i])

        end_time = time()
        print('Video {}, {}s elapsed running in {}'.format(
            vn, end_time - start_time, device))
