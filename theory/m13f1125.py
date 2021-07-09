from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
im_orig = Image.open('marinesnow3.jpg')

im_bw = im_orig.convert('L')
im = np.array(im_bw)

U, s, V = np.linalg.svd(im)

components = 39
Ur = U[:, :components]
Vr = V[:components, :]

#ima = im-np.mean(im)

sm = np.zeros([len(U), len(V)])

for j in range(0, len(U)):
    sm[j, j] = s[j]

sm = sm[:components, :components]


im_recon = np.matmul(np.matmul(Ur, sm), Vr)

plt.imshow(im_recon, cmap='gray')


# 이미지는 평균을 안빼줌
# 해수 온도 데이터는 첫번째 주성분이 얼마나 중요한지 정량화 하기 위해서 평균을 빼줌
# 람다에서 뺀다는게 아니라 원 데이터에서 빼준다는것
# 그래야 람다끼리 비교 가능
# 람다는 아이겐 벡터의 첫번째 열부터 아이겐 벨류의 첫뻔째 주성분부터 나오게된다?


a11 = 1
a12 = 2
a13 = 3
a21 = 4
a22 = 5
a23 = 6

A = np.array([[a11, a12, a13], [a21, a22, a23]])
B = A.T

C = np.dot(A, B)
D = np.dot(B, A)
