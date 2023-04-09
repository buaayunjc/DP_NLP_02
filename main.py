# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy import stats


def em(h, mu1, sigmal, w1, mu2, sigma2, w2):
    d = 1
    n = len(h)  # 样本长度

    # E-step
    # 计算响应
    # p1=w1*flot(stats.norm(mu1,sigmal))
    p1 = w1 * stats.norm(mu1, sigmal).pdf(h)
    p2 = w2 * stats.norm(mu2, sigma2).pdf(h)
    # p1, p2权重 * 男女生的后验概率
    R1i = p1 / (p1 + p2)
    R2i = p2 / (p1 + p2)

    # M-step
    # mu的更新
    mu1 = np.sum(R1i * h) / np.sum(R1i)
    mu2 = np.sum(R2i * h) / np.sum(R2i)
    # sigmal的更新
    sigmal = np.sqrt(np.sum(R1i * np.square(h - mu1)) / (d * np.sum(R1i)))
    sigma2 = np.sqrt(np.sum(R2i * np.square(h - mu2)) / (d * np.sum(R2i)))
    # w的更新
    w1 = np.sum(R1i) / n
    w2 = np.sum(R2i) / n

    return mu1, sigmal, w1, mu2, sigma2, w2


np.random.seed(100)

male = np.random.normal(176,5,1500)
female = np.random.normal(164,3,500)
h=list(male)
h.extend(female)
h=np.array(h)

mu1=170;sigmal=10;w1=0.7
mu2=160;sigma2=10;w2=0.3

d=1
n = len(h)  # 样本长度
# 开始EM算法的主循环
for iteration in range(100):
    mu1,sigmal,w1,mu2,sigma2,w2=em(h,mu1,sigmal,w1,mu2,sigma2,w2)

print(mu1,sigmal,w1,mu2,sigma2,w2)
t=np.linspace(120,220,550)
m = stats.norm.pdf(t,loc=mu1, scale=sigmal)
f = stats.norm.pdf(t,loc=mu2, scale=sigma2)
mix=w1*m+w2*f
plt.plot(t, m, color='b')
plt.plot(t, f, color='r')
plt.plot(t, mix, color='k')

plt.title('Probability density curve for boys and girls and mixed height')

plt.legend(["male","female","mixing"],loc='upper right')
plt.xlabel('height/cm')
plt.ylabel('Probability')
plt.show()