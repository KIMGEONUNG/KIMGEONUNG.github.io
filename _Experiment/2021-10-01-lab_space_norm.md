---
title: 'LAB color space normalization'
classes: wide
---

## Motivation
 
&nbsp; RGB 영상 데이터를 딥러닝 모델에 학습시킬때, 0 부터 255 사이의 값을 
그대로 사용하지 않고, 0과 1사이, 또는 그에 상응하는 값으로 정규화해서 사용한다. 
직관적으로 이해하기를, 인공신경망의 각 레이어마다 통과하는 값의 분포가 크게 
다르면 학습에 불리하다고 알고있다.
<br>
<center><i>그렇다면, LAB color space에서는 normalization이 유효할까?</i></center>
<br>
&nbsp; 흑백 영상을 색상을 가진 영상으로 바꾸는 colorization 문제를 풀어야해서 
LAB color space를 사용할 일이 있었다. LAB space는 어떻게 정규화하는 것이 좋을지
궁금해서 실험을 진행했다.

## Experiment

&nbsp; 특정 이미지의 L값을 받아서 AB값을 추정하는 가벼운 CNN 모델을 학습시키는데,
정규화 되지 않은 데이터와 정규화된 데이터를 각각 학습시켜서 어떤 모델이 더 잘
AB값을 복원하는지 비교한다. 여기서 모델의 learning capacity가 너무 크면, 복원
결과가 비슷하게 나오기 때문에 적당하게 작은 모델을 찾아야했다. 여러 모델과 
iteration으로 실험해본 결과, **항상 정규화된 데이터가 좋은 결과**를 보여줬다.
아래 이미지는 정석적인 비교를 보여주고 있는데, 왼쪽에 위치한 영상이 정규화된
데이터를 사용해서 학습된 모델의 추정 결과다. 오른쪽 영상과 비교해 볼 때, 색상
복원 품질이 더 좋은 것을 확인할 수 있다.

|![Image](/assets/images/lab-color-space-normalization/recon_norm_True.png) |![Image](/assets/images/lab-color-space-normalization/sample01.png) |![Image](/assets/images/lab-color-space-normalization/recon_norm_False.png)|
|:--:|:--:|:--:|
|W/ Norm.| Original | W/O Norm. | 

## Plots

&nbsp; 아래 그래프에서 Distance는 단순한 픽셀값의 차이를 모두 더한 것이다.
약 1200정도의 차이를 보여주고 있는데, 모델의 learning capacity가 크고 iteration
횟수가 많을 수록 distance의 값은 줄어들었다. 

|![Image of Yaktocat](/assets/images/lab-color-space-normalization/plot_norm_True.png)| ![Image of Yaktocat](/assets/images/lab-color-space-normalization/plot_norm_False.png)|
|:--:|:--:|
|W/ Norm.|W/O Norm.| 

## Code 

```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from skimage import color
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import random
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--iter', type=int, default=2000)
    parser.add_argument('--size_ch', type=int, default=6)
    parser.add_argument('--path_image', type=str, default='./sample01.jpg')
    return parser.parse_args()


def prep_lab(x):
    x[:, :1, :, :] /= 100
    x[:, 1:, :, :] /= 128
    return x


def post_lab(x):
    x[:, :1, :, :] *= 100
    x[:, 1:, :, :] *= 128
    return x


def colorize(args, normalize):
    print('func.colorize(normalize=%s)' % str(normalize))

    im = Image.open(args.path_image)
    im = transforms.ToTensor()(im)
    rgb_original = im.clone()
    im = im.permute(1, 2, 0)
    lab = color.rgb2lab(im)
    lab = torch.from_numpy(lab).permute(2, 0, 1).unsqueeze(0)

    if normalize:
        lab = prep_lab(lab)

    x_l = lab[:, :1, :, :].cuda()
    x_ab = lab[:, 1:, :, :].cuda()

    model = nn.Sequential(
            nn.Conv2d(1, args.size_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(args.size_ch, args.size_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(args.size_ch, 2, kernel_size=3, padding=1),
            ).cuda()

    optimizer = optim.Adam(model.parameters())
    loss_f = nn.MSELoss()

    losses = []
    for i in tqdm(range(args.iter)):
        y = model(x_l)
        loss = loss_f(x_ab, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    lab = torch.cat([x_l, y], dim=1)
    if normalize:
        lab = post_lab(lab)
    lab = lab.squeeze().permute(1, 2, 0)
    rgb = color.lab2rgb(lab.cpu().detach())
    rgb = torch.from_numpy(rgb).permute(2, 0, 1)

    diff = rgb_original.sub(rgb).pow(2).sum()
    log_distance = 'distance:' + str(diff.item())
    log_normalize = 'normalize:' + str(normalize)
    log = log_distance + ' | ' + log_normalize

    im = transforms.ToPILImage()(rgb)
    im.save('./recon_norm_%s.png' % (str(normalize)))

    plt.clf()
    plt.plot(torch.linspace(1, args.iter, args.iter), losses)
    plt.title(log)
    plt.savefig('./plot_norm_%s.png' % (str(normalize)), dpi=300)


if __name__ == '__main__':
    args = parse()

    if args.seed > -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)

    colorize(args, normalize=True)
    colorize(args, normalize=False)
```
