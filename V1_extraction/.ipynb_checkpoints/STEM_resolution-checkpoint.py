import math

import scipy.io
import numpy as np
import warnings
import os
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
import time
from utilities import *
import pickle

time_cost = 0
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    data_name = 'KoNViD'
    
    if data_name == 'KoNViD':
        meta_data = pd.read_csv('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/konvid1k_metadata.csv')
        flickr_ids = meta_data.flickr_id
        mos = meta_data.mos.to_numpy()
    elif data_name == 'LiveVQC':
        meta_data = pd.read_csv('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/livevqc_metadata.csv')
        flickr_ids = meta_data.File
        mos = meta_data.MOS.to_numpy()
        
    data_length = mos.shape[0]

    pca_d = 10
    k = 6

    tem_quality = np.zeros((data_length, 1))
    fused_quality = np.zeros((data_length, 1))
    fused_quality2 = np.zeros((data_length, 1))

    lgn_quality = np.zeros((data_length, 1))
    V1_quality = np.zeros((data_length, 1))

    niqe_quality = np.zeros((data_length, 1))

    for v in range(data_length):
        time_start = time.time()
        if data_name == 'KoNViD':
            lgn_feature_mat = scipy.io.loadmat('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/lgn/KoNViD/' + str(flickr_ids[v]) + '.mp4.mat')
            lgn_feature = lgn_feature_mat['LGN_features_level6']
            
            V1_feature = np.load('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/resolution/konvid1k120/' + str(flickr_ids[v]) + '.mp4.npy')
            
            niqe_score_mat = scipy.io.loadmat('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/NIQE/KoNViD/'+str(flickr_ids[v])+'.mat')
            niqe_score = niqe_score_mat['features_norm22']
            
        elif data_name == 'LiveVQC':
            lgn_feature_mat = scipy.io.loadmat('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/lgn/LiveVQC/' + flickr_ids[v] + '.mat')
            lgn_feature = lgn_feature_mat['LGN_features']
                
            V1_feature = np.load('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/resolution/livevqc120/' + flickr_ids[v] + '.npy')

            
            niqe_score_mat = scipy.io.loadmat('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/NIQE/LiveVQC/'+ flickr_ids[v] + '.mat')
            niqe_score = niqe_score_mat['features_norm22']
            
        lgn_feature = np.asarray(lgn_feature, dtype=np.float)
        lgn_feature = clear_data(lgn_feature)

        V1_feature = np.asarray(V1_feature, dtype=np.float)
        V1_feature = clear_data(V1_feature)


        pca = PCA(n_components=pca_d)
        pca.fit(lgn_feature)
        lgn_PCA = pca.transform(lgn_feature)

        pca = PCA(n_components=pca_d)
        pca.fit(V1_feature)
        V1_PCA = pca.transform(V1_feature)


        lgn_score = compute_lgn_curvature(lgn_PCA)
        v1_score = compute_v1_curvature(V1_PCA)


        lgn_quality[v] = math.log(np.mean(lgn_score))
        V1_quality[v] = math.log(np.mean(v1_score))

        niqe_quality[v] = np.mean(niqe_score)

        time_end = time.time()
        #print('Video {}, overall {} seconds elapsed...'.format(
        #    v, time_end - time_start))

    print(120)
    temporal_quality = V1_quality + lgn_quality
    data = temporal_quality.squeeze()
    data = data[~np.isnan(data)]
    data = data[~np.isposinf(data)]
    temporal_quality = data[~np.isneginf(data)]

    mu = np.mean(temporal_quality)
    sigma = np.std(temporal_quality)

    mu_niqe = np.mean(niqe_quality)
    sigma_niqe = np.std(niqe_quality)

    niqe_quality = (niqe_quality-mu_niqe)/sigma_niqe*sigma+mu
    print(mu_niqe, sigma_niqe, sigma, mu, len(temporal_quality))

    fused_quality = (V1_quality + lgn_quality) * niqe_quality
    fused_quality2 = (V1_quality) * niqe_quality


    curvage_mos = fused_quality
    tem_mos = mos
    curvage_mos = fit_curve(curvage_mos.squeeze(), tem_mos)
    print('overall:', compute_metrics(tem_mos.squeeze(), curvage_mos.squeeze(), haveFit=True))

    curvage_mos = (V1_quality + lgn_quality)
    tem_mos = mos
    curvage_mos = fit_curve(curvage_mos.squeeze(), tem_mos)
    print('temporal:', compute_metrics(tem_mos.squeeze(), curvage_mos.squeeze(), haveFit=True))
    #
    curvage_mos = lgn_quality
    fused_mos = mos
    curvage_mos = fit_curve(curvage_mos.squeeze(), fused_mos)
    print('lgn_quality:', compute_metrics(tem_mos.squeeze(), curvage_mos.squeeze(), haveFit=True))
    #
    curvage_mos = V1_quality
    tem_mos = mos
    curvage_mos = fit_curve(curvage_mos.squeeze(), tem_mos)
    print('V1_quality:', compute_metrics(tem_mos.squeeze(), curvage_mos.squeeze(), haveFit=True))




    for v in range(data_length):
        time_start = time.time()
        if data_name == 'KoNViD':
            lgn_feature_mat = scipy.io.loadmat('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/lgn/KoNViD/' + str(flickr_ids[v]) + '.mp4.mat')
            lgn_feature = lgn_feature_mat['LGN_features_level6']
            
            V1_feature = np.load('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/resolution/konvid1k240/' + str(flickr_ids[v]) + '.mp4.npy')
            
            niqe_score_mat = scipy.io.loadmat('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/NIQE/KoNViD/'+str(flickr_ids[v])+'.mat')
            niqe_score = niqe_score_mat['features_norm22']
            
        elif data_name == 'LiveVQC':
            lgn_feature_mat = scipy.io.loadmat('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/lgn/LiveVQC/' + flickr_ids[v] + '.mat')
            lgn_feature = lgn_feature_mat['LGN_features']
                
            V1_feature = np.load('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/resolution/livevqc240/' + flickr_ids[v] + '.npy')

            
            niqe_score_mat = scipy.io.loadmat('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/NIQE/LiveVQC/'+ flickr_ids[v] + '.mat')
            niqe_score = niqe_score_mat['features_norm22']
            
        lgn_feature = np.asarray(lgn_feature, dtype=np.float)
        lgn_feature = clear_data(lgn_feature)

        V1_feature = np.asarray(V1_feature, dtype=np.float)
        V1_feature = clear_data(V1_feature)


        pca = PCA(n_components=pca_d)
        pca.fit(lgn_feature)
        lgn_PCA = pca.transform(lgn_feature)

        pca = PCA(n_components=pca_d)
        pca.fit(V1_feature)
        V1_PCA = pca.transform(V1_feature)


        lgn_score = compute_lgn_curvature(lgn_PCA)
        v1_score = compute_v1_curvature(V1_PCA)


        lgn_quality[v] = math.log(np.mean(lgn_score))
        V1_quality[v] = math.log(np.mean(v1_score))

        niqe_quality[v] = np.mean(niqe_score)

        time_end = time.time()
        #print('Video {}, overall {} seconds elapsed...'.format(
        #    v, time_end - time_start))
    print(240)
    temporal_quality = V1_quality + lgn_quality
    data = temporal_quality.squeeze()
    data = data[~np.isnan(data)]
    data = data[~np.isposinf(data)]
    temporal_quality = data[~np.isneginf(data)]

    mu = np.mean(temporal_quality)
    sigma = np.std(temporal_quality)

    mu_niqe = np.mean(niqe_quality)
    sigma_niqe = np.std(niqe_quality)

    niqe_quality = (niqe_quality-mu_niqe)/sigma_niqe*sigma+mu
    print(mu_niqe, sigma_niqe, sigma, mu, len(temporal_quality))

    fused_quality = (V1_quality + lgn_quality) * niqe_quality
    fused_quality2 = (V1_quality) * niqe_quality


    curvage_mos = fused_quality
    tem_mos = mos
    curvage_mos = fit_curve(curvage_mos.squeeze(), tem_mos)
    print('overall:', compute_metrics(tem_mos.squeeze(), curvage_mos.squeeze(), haveFit=True))

    curvage_mos = (V1_quality + lgn_quality)
    tem_mos = mos
    curvage_mos = fit_curve(curvage_mos.squeeze(), tem_mos)
    print('temporal:', compute_metrics(tem_mos.squeeze(), curvage_mos.squeeze(), haveFit=True))
    #
    curvage_mos = lgn_quality
    fused_mos = mos
    curvage_mos = fit_curve(curvage_mos.squeeze(), fused_mos)
    print('lgn_quality:', compute_metrics(tem_mos.squeeze(), curvage_mos.squeeze(), haveFit=True))
    #
    curvage_mos = V1_quality
    tem_mos = mos
    curvage_mos = fit_curve(curvage_mos.squeeze(), tem_mos)
    print('V1_quality:', compute_metrics(tem_mos.squeeze(), curvage_mos.squeeze(), haveFit=True))
    
    
    
    
    
    
    for v in range(data_length):
        time_start = time.time()
        if data_name == 'KoNViD':
            lgn_feature_mat = scipy.io.loadmat('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/lgn/KoNViD/' + str(flickr_ids[v]) + '.mp4.mat')
            lgn_feature = lgn_feature_mat['LGN_features_level6']
            
            V1_feature = np.load('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/resolution/konvid1k480/' + str(flickr_ids[v]) + '.mp4.npy')
            
            niqe_score_mat = scipy.io.loadmat('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/NIQE/KoNViD/'+str(flickr_ids[v])+'.mat')
            niqe_score = niqe_score_mat['features_norm22']
            
        elif data_name == 'LiveVQC':
            lgn_feature_mat = scipy.io.loadmat('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/lgn/LiveVQC/' + flickr_ids[v] + '.mat')
            lgn_feature = lgn_feature_mat['LGN_features']
                
            V1_feature = np.load('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/resolution/livevqc480/' + flickr_ids[v] + '.npy')

            
            niqe_score_mat = scipy.io.loadmat('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/NIQE/LiveVQC/'+ flickr_ids[v] + '.mat')
            niqe_score = niqe_score_mat['features_norm22']
            
        lgn_feature = np.asarray(lgn_feature, dtype=np.float)
        lgn_feature = clear_data(lgn_feature)

        V1_feature = np.asarray(V1_feature, dtype=np.float)
        V1_feature = clear_data(V1_feature)


        pca = PCA(n_components=pca_d)
        pca.fit(lgn_feature)
        lgn_PCA = pca.transform(lgn_feature)

        pca = PCA(n_components=pca_d)
        pca.fit(V1_feature)
        V1_PCA = pca.transform(V1_feature)


        lgn_score = compute_lgn_curvature(lgn_PCA)
        v1_score = compute_v1_curvature(V1_PCA)


        lgn_quality[v] = math.log(np.mean(lgn_score))
        V1_quality[v] = math.log(np.mean(v1_score))

        niqe_quality[v] = np.mean(niqe_score)

        time_end = time.time()
        #print('Video {}, overall {} seconds elapsed...'.format(
        #    v, time_end - time_start))

    temporal_quality = V1_quality + lgn_quality
    data = temporal_quality.squeeze()
    data = data[~np.isnan(data)]
    data = data[~np.isposinf(data)]
    temporal_quality = data[~np.isneginf(data)]

    mu = np.mean(temporal_quality)
    sigma = np.std(temporal_quality)

    mu_niqe = np.mean(niqe_quality)
    sigma_niqe = np.std(niqe_quality)

    niqe_quality = (niqe_quality-mu_niqe)/sigma_niqe*sigma+mu
    print(mu_niqe, sigma_niqe, sigma, mu, len(temporal_quality))

    fused_quality = (V1_quality + lgn_quality) * niqe_quality
    fused_quality2 = (V1_quality) * niqe_quality

    print(480)
    curvage_mos = fused_quality
    tem_mos = mos
    curvage_mos = fit_curve(curvage_mos.squeeze(), tem_mos)
    print('overall:', compute_metrics(tem_mos.squeeze(), curvage_mos.squeeze(), haveFit=True))

    curvage_mos = (V1_quality + lgn_quality)
    tem_mos = mos
    curvage_mos = fit_curve(curvage_mos.squeeze(), tem_mos)
    print('temporal:', compute_metrics(tem_mos.squeeze(), curvage_mos.squeeze(), haveFit=True))
    #
    curvage_mos = lgn_quality
    fused_mos = mos
    curvage_mos = fit_curve(curvage_mos.squeeze(), fused_mos)
    print('lgn_quality:', compute_metrics(tem_mos.squeeze(), curvage_mos.squeeze(), haveFit=True))
    #
    curvage_mos = V1_quality
    tem_mos = mos
    curvage_mos = fit_curve(curvage_mos.squeeze(), tem_mos)
    print('V1_quality:', compute_metrics(tem_mos.squeeze(), curvage_mos.squeeze(), haveFit=True))
    
    
    


    for v in range(data_length):
        time_start = time.time()
        if data_name == 'KoNViD':
            lgn_feature_mat = scipy.io.loadmat('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/lgn/KoNViD/' + str(flickr_ids[v]) + '.mp4.mat')
            lgn_feature = lgn_feature_mat['LGN_features_level6']
            
            V1_feature = np.load('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/resolution/konvid1k960/' + str(flickr_ids[v]) + '.mp4.npy')
            
            niqe_score_mat = scipy.io.loadmat('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/NIQE/KoNViD/'+str(flickr_ids[v])+'.mat')
            niqe_score = niqe_score_mat['features_norm22']
            
        elif data_name == 'LiveVQC':
            lgn_feature_mat = scipy.io.loadmat('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/lgn/LiveVQC/' + flickr_ids[v] + '.mat')
            lgn_feature = lgn_feature_mat['LGN_features']
                
            V1_feature = np.load('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/resolution/livevqc960/' + flickr_ids[v] + '.npy')

            
            niqe_score_mat = scipy.io.loadmat('/mnt/lustre/lliao/Sourcecode/TPAMI-VQA/V1_extraction/features/NIQE/LiveVQC/'+ flickr_ids[v] + '.mat')
            niqe_score = niqe_score_mat['features_norm22']
            
        lgn_feature = np.asarray(lgn_feature, dtype=np.float)
        lgn_feature = clear_data(lgn_feature)

        V1_feature = np.asarray(V1_feature, dtype=np.float)
        V1_feature = clear_data(V1_feature)


        pca = PCA(n_components=pca_d)
        pca.fit(lgn_feature)
        lgn_PCA = pca.transform(lgn_feature)

        pca = PCA(n_components=pca_d)
        pca.fit(V1_feature)
        V1_PCA = pca.transform(V1_feature)


        lgn_score = compute_lgn_curvature(lgn_PCA)
        v1_score = compute_v1_curvature(V1_PCA)


        lgn_quality[v] = math.log(np.mean(lgn_score))
        V1_quality[v] = math.log(np.mean(v1_score))

        niqe_quality[v] = np.mean(niqe_score)

        time_end = time.time()
        #print('Video {}, overall {} seconds elapsed...'.format(
        #    v, time_end - time_start))

    temporal_quality = V1_quality + lgn_quality
    data = temporal_quality.squeeze()
    data = data[~np.isnan(data)]
    data = data[~np.isposinf(data)]
    temporal_quality = data[~np.isneginf(data)]

    mu = np.mean(temporal_quality)
    sigma = np.std(temporal_quality)

    mu_niqe = np.mean(niqe_quality)
    sigma_niqe = np.std(niqe_quality)

    niqe_quality = (niqe_quality-mu_niqe)/sigma_niqe*sigma+mu
    print(mu_niqe, sigma_niqe, sigma, mu, len(temporal_quality))

    fused_quality = (V1_quality + lgn_quality) * niqe_quality
    fused_quality2 = (V1_quality) * niqe_quality

    print(960)
    curvage_mos = fused_quality
    tem_mos = mos
    curvage_mos = fit_curve(curvage_mos.squeeze(), tem_mos)
    print('overall:', compute_metrics(tem_mos.squeeze(), curvage_mos.squeeze(), haveFit=True))

    curvage_mos = (V1_quality + lgn_quality)
    tem_mos = mos
    curvage_mos = fit_curve(curvage_mos.squeeze(), tem_mos)
    print('temporal:', compute_metrics(tem_mos.squeeze(), curvage_mos.squeeze(), haveFit=True))
    #
    curvage_mos = lgn_quality
    fused_mos = mos
    curvage_mos = fit_curve(curvage_mos.squeeze(), fused_mos)
    print('lgn_quality:', compute_metrics(tem_mos.squeeze(), curvage_mos.squeeze(), haveFit=True))
    #
    curvage_mos = V1_quality
    tem_mos = mos
    curvage_mos = fit_curve(curvage_mos.squeeze(), tem_mos)
    print('V1_quality:', compute_metrics(tem_mos.squeeze(), curvage_mos.squeeze(), haveFit=True))