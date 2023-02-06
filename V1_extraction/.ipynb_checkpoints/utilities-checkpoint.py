import numpy as np
from sklearn import linear_model
import scipy
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import cv2
from scipy import signal
from sklearn.preprocessing import StandardScaler

def compute_v1_curvature(features):
    len = features.shape[0]
    curvatures = np.zeros((len-2, 1))
    theta = np.zeros((len-2, 1))
    distance = np.zeros((len-2, 1))

    for fn in range(1, len - 1):
        prev = features[fn - 1, :]
        cur = features[fn, :]
        next = features[fn + 1, :]
        numerator = np.dot((next - cur).T, (cur-prev)).squeeze()
        denominator = np.linalg.norm(next - cur) * np.linalg.norm(cur - prev)

        if denominator<0.0001 or np.abs(numerator)<0.0001:
            theta = 3.141592/2
        else:
            theta = np.arccos(numerator / (0.000001+denominator))

        cos_alpha = theta*np.power(np.linalg.norm(next - prev), 1)
        #cos_alpha = np.arccos(numerator / (0.000001+denominator))*np.power(np.linalg.norm(next - prev), 0.5)

        curvatures[fn - 1] = cos_alpha
        # theta[fn - 1] = np.arccos(numerator / (0.000001+denominator))
        # distance[fn - 1] = np.linalg.norm(next - prev)
        #
        # if np.isnan(theta[fn - 1])| np.isposinf(theta[fn - 1]) | np.isneginf(theta[fn - 1]):
        #     theta = 0

    # mu = np.mean(distance)
    # sigma = np.std(distance)
    #
    # mu_niqe = np.mean(theta)
    # sigma_niqe = np.std(theta)
    #
    # theta = (theta-mu_niqe)/sigma_niqe*sigma+mu
    # for fn in range(1, len - 1):
    #     curvatures[fn - 1] = (theta[fn - 1]-np.min(theta))*distance[fn - 1]
    # for fn in range(1, len - 1):
    #     curvatures[fn - 1] = theta[fn - 1]*distance[fn - 1]/np.mean(distance)
    return curvatures



def compute_discrete_v1_curvature(features):
    len = features.shape[0]
    curvatures = np.zeros((len//3, 1))
    theta = np.zeros((len//3, 1))
    distance = np.zeros((len//3, 1))

    for fn in range(0, len//3):
        prev = features[fn*3]
        cur = features[fn*3+1]
        next = features[fn*3+2]
        numerator = np.dot((next - cur).T, (cur-prev)).squeeze()
        denominator = np.linalg.norm(next - cur) * np.linalg.norm(cur - prev)

        if denominator<0.0001 or np.abs(numerator)<0.0001:
            theta = 3.141592/2
        else:
            theta = np.arccos(numerator / (0.000001+denominator))

        cos_alpha = theta*np.power(np.linalg.norm(next - prev), 1)
        #cos_alpha = np.arccos(numerator / (0.000001+denominator))*np.power(np.linalg.norm(next - prev), 0.5)

        curvatures[fn - 1] = cos_alpha
        # theta[fn - 1] = np.arccos(numerator / (0.000001+denominator))
        # distance[fn - 1] = np.linalg.norm(next - prev)
        #
        # if np.isnan(theta[fn - 1])| np.isposinf(theta[fn - 1]) | np.isneginf(theta[fn - 1]):
        #     theta = 0

    # mu = np.mean(distance)
    # sigma = np.std(distance)
    #
    # mu_niqe = np.mean(theta)
    # sigma_niqe = np.std(theta)
    #
    # theta = (theta-mu_niqe)/sigma_niqe*sigma+mu
    # for fn in range(1, len - 1):
    #     curvatures[fn - 1] = (theta[fn - 1]-np.min(theta))*distance[fn - 1]
    # for fn in range(1, len - 1):
    #     curvatures[fn - 1] = theta[fn - 1]*distance[fn - 1]/np.mean(distance)
    return curvatures

def compute_lgn_curvature(features):
    len = features.shape[0]
    curvatures = np.zeros((len-2, 1))
    theta = np.zeros((len-2, 1))
    distance = np.zeros((len-2, 1))

    for fn in range(1, len - 1):
        prev = features[fn - 1, :]
        cur = features[fn, :]
        next = features[fn + 1, :]
        numerator = np.dot((next - cur).T, (cur-prev)).squeeze()
        denominator = np.linalg.norm(next - cur) * np.linalg.norm(cur - prev)

        if denominator<0.0001 or np.abs(numerator)<0.0001:
            theta = 3.141592/2
        else:
            theta = np.arccos(numerator / (0.000001+denominator))

        cos_alpha = np.power(theta*np.power(np.linalg.norm(next - prev), 0.25), 1)
        #cos_alpha = np.arccos(numerator / (0.000001+denominator))*np.power(np.linalg.norm(next - prev), 0.5)

        curvatures[fn - 1] = cos_alpha
    return curvatures

def linear_pred(features, k):
    len = features.shape[0]
    linear_error = np.zeros((len - k, 1))

    for fn in range(1, len - k):
        prev = features[fn:fn+k, :].T
        cur = features[fn+k, :]
        lr = linear_model.LinearRegression()
        lr.fit(prev, cur)
        pred = lr.predict(prev)

        linear_error[fn] = np.linalg.norm(cur-pred, 1)

    return linear_error

def extract_img_features(video_name, rate):
    vid_cap = cv2.VideoCapture(video_name)

    img_features = []
    while 1:
        ret, img = vid_cap.read()
        if not ret:
            break
        width = int(img.shape[1] * rate)
        height = int(img.shape[0] * rate)
        img = cv2.resize(img, dsize=(width,height))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_norm = cv2.normalize(img_gray,
                                      None,
                                      alpha=0,
                                      beta=1,
                                      norm_type=cv2.NORM_MINMAX,
                                      dtype=cv2.CV_32F)
        img_features.append(
            img_gray_norm.reshape(
                img_gray_norm.shape[0] * img_gray_norm.shape[1], 1))

    img_features = np.array(img_features).squeeze()

    return img_features

def geometric_mean2(data):
    total = 1
    for i in data:
        total*=i
    return pow(total, 1/len(data))

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    # 4-parameter logistic function
    logisticPart = 1 + np.exp(
        np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def compute_metrics(y_pred, y, haveFit=False):
    '''
  compute metrics btw predictions & labels
  '''
    # compute SRCC & KRCC
    SRCC = scipy.stats.spearmanr(y, y_pred)[0]
    try:
        KRCC = scipy.stats.kendalltau(y, y_pred)[0]
    except:
        KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]

    if not haveFit:
        # logistic regression btw y_pred & y
        beta_init = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
        popt, _ = curve_fit(logistic_func,
                            y_pred,
                            y,
                            p0=beta_init,
                            maxfev=int(1e8))
        y_pred_logistic = logistic_func(y_pred, *popt)
    else:
        y_pred_logistic = y_pred

    # compute  PLCC RMSE
    PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
    RMSE = np.sqrt(mean_squared_error(y, y_pred_logistic))
    return [SRCC, PLCC, KRCC, RMSE]


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    # 4-parameter logistic function
    logisticPart = 1 + np.exp(
        np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def plot_scatter(input_x,
                 input_y,
                 save_path,
                 xlabel='MOS',
                 ylabel='Curvatures',
                 haveFit=False):
    # 可视化
    p = np.polyfit(input_x, input_y, 1).squeeze()
    min_val = np.min(input_x)
    max_val = np.max(input_x)
    x = np.linspace(min_val, max_val, 1000)
    f = np.poly1d(p)
    y = f(x)

    srcc, plcc, krcc, rmse = compute_metrics(input_x.squeeze(),
                                             input_y.squeeze(),
                                             haveFit=haveFit)
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.scatter(input_x, input_y, s=7.5, c='b', marker='D')
    plt.plot(x, y, c='r')
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title('SRCC: {} | PLCC: {} |  RMSE: {}'.format(
        round(srcc, 3), round(plcc, 3), round(rmse, 3)), fontsize=20)
    # plt.xlim(2.37, 3.78)  # 确定横轴坐标范围
    # plt.ylim(2.37, 3.78)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    plt.savefig(save_path)
    plt.clf()

def clear_data(data):

    data[np.isnan(data) | np.isposinf(data)
                 | np.isneginf(data)] = 0
    return data

def clear_mos(data, mos):
    data = data.squeeze()
    mos = mos[~np.isnan(data)]
    data = data[~np.isnan(data)]
    mos = mos[~np.isposinf(data)]
    data = data[~np.isposinf(data)]
    mos = mos[~np.isneginf(data)]
    data = data[~np.isneginf(data)]
    return data, mos

def fit_curve(x, y):
    """fit x to y"""
    # logistic regression
    beta_init = [np.max(y), np.min(y), np.mean(x), 0.5]
    popt, _ = curve_fit(logistic_func, x, y, p0=beta_init, maxfev=int(1e8))
    y_logistic = logistic_func(x, *popt)

    return y_logistic


def hysteresis_pooling(chunk):
    '''parameters'''
    tau = 8 # 2-sec * 30 fps
    comb_alpha = 0.2 # weighting
    ''' function body '''
    chunk = np.asarray(chunk, dtype=np.float64)
    chunk_length = len(chunk)
    l = np.zeros(chunk_length)
    m = np.zeros(chunk_length)
    q = np.zeros(chunk_length)
    for t in range(chunk_length):
        ''' calculate l[t] - the memory component '''
        if t == 0: # corner case
            l[t] = chunk[t]
        else:
            # get previous frame indices
            idx_prev = slice(max(0, t-tau), max(0, t-1)+1)
            # print(idx_prev)
            # calculate min scores
            l[t] = min(chunk[idx_prev])
        # print("l[t]:", l[t])
        ''' compute m[t] - the current component '''
        if t == chunk_length - 1: # corner case
            m[t] = chunk[t]
        else:
            # get next frame indices
            idx_next = slice(t, min(t + tau, chunk_length))
            # print(idx_next)
            # sort ascend order
            v = np.sort(chunk[idx_next])
            # generated Gaussian weight
            win_len = len(v) * 2.0 - 1.0
            win_sigma = win_len / 6.0
            # print(win_len, win_sigma)
            gaussian_win = signal.gaussian(win_len, win_sigma)
            gaussian_half_win = gaussian_win[len(v)-1:]
            # normalize gaussian descend kernel
            gaussian_half_win = np.divide(gaussian_half_win, np.sum(gaussian_half_win))
            # print(gaussian_half_win)
            m[t] = sum([x * y for x, y in zip(v, gaussian_half_win)])
        # print("m[t]:", m[t])
    ''' combine l[t] and m[t] into one q[t] '''
    q = comb_alpha * l + (1.0 - comb_alpha) * m
    # print(q)
    # print(np.mean(q))
    return q, np.mean(q)