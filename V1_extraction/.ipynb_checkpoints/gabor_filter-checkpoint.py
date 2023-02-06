import math
import cmath
import torch
import torch.nn as nn


class GaborFilters(nn.Module):

    def __init__(self,
                 n_scale=5,
                 n_orientation=8,
                 kernel_radius=9,
                 row_downsample=4, 
                 column_downsample=4, 
                 device='cpu'):
        super().__init__()
        self.kernel_size = kernel_radius * 2 + 1
        self.kernel_radius = kernel_radius
        self.n_scale = n_scale
        self.n_orientation = n_orientation
        self.row_downsample = row_downsample
        self.column_downsample = column_downsample

        self.to(device)
        self.gb = self.make_gabor_filters().to(device)

    def make_gabor_filters(self):
        kernel_size = self.kernel_size
        n_scale = self.n_scale
        n_orientation = self.n_orientation

        gb = torch.zeros((n_scale * n_orientation, kernel_size, kernel_size),
                         dtype=torch.cfloat)

        fmax = 0.25
        gama = math.sqrt(2)
        eta = math.sqrt(2)

        for i in range(n_scale):
            fu = fmax / (math.sqrt(2)**i)
            alpha = fu / gama
            beta = fu / eta
            for j in range(n_orientation):
                tetav = (j / n_orientation) * math.pi
                g_filter = torch.zeros((kernel_size, kernel_size),
                                       dtype=torch.cfloat)
                for x in range(1, kernel_size + 1):
                    for y in range(1, kernel_size + 1):
                        xprime = (x - (
                            (kernel_size + 1) / 2)) * math.cos(tetav) + (y - (
                                (kernel_size + 1) / 2)) * math.sin(tetav)
                        yprime = -(x - (
                            (kernel_size + 1) / 2)) * math.sin(tetav) + (y - (
                                (kernel_size + 1) / 2)) * math.cos(tetav)
                        g_filter[x - 1][
                            y -
                            1] = (fu**2 / (math.pi * gama * eta)) * math.exp(-(
                                (alpha**2) * (xprime**2) + (beta**2) *
                                (yprime**2))) * cmath.exp(
                                    1j * 2 * math.pi * fu * xprime)
                gb[i * n_orientation + j] = g_filter

        return gb

    def forward(self, x):
        batch_size = x.size(0)
        cn = x.size(1)
        sy = x.size(2)
        sx = x.size(3)

        assert cn == 1

        gb = self.gb
        gb = gb[:, None, :, :]

        res = nn.functional.conv2d(input=x, weight=gb, padding='same')

        res = res.view(batch_size, -1, sy, sx)

        res = torch.abs(res)
        res = res[:, :, ::self.row_downsample, :]
        res = res[:, :, :, ::self.column_downsample]
        res = res.reshape(batch_size, res.size(1), -1)
        res = (res - torch.mean(res, 2, keepdim=True)) / torch.std(
            res, 2, keepdim=True)
        res = res.view(batch_size, -1)

        return res


if __name__ == "__main__":
    import time
    from PIL import Image
    from torchvision.transforms import transforms

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    img = Image.open(
        '/mnt/d/datasets/KonIQ-10k/images_512x384/826373.jpg').convert('L')
    img = transforms.ToTensor()(img)
    img_imag = torch.zeros(img.size())
    img = torch.stack((img, img_imag), 3)
    img = torch.view_as_complex(img)
    img = img[None, :, :, :]

    gb = GaborFilters(device=device)
    img = img.to(device)
    start_time = time.time()
    res = gb(img)
    end_time = time.time()
    print(res.shape)
    print('{}s elapsed running in {}'.format(end_time - start_time, device))
