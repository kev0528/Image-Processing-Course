import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 影像讀取(.convert('L')保證了後續處理在單一亮度通道上進行，而不是在 RGB 格式上操作。)
p1=np.array(Image.open('Cameraman.bmp').convert('L'))
p2=np.array(Image.open('Jetplane.bmp').convert('L'))
p3=np.array(Image.open('Lake.bmp').convert('L'))
p4=np.array(Image.open('Peppers.bmp').convert('L'))

# 初始化直方圖，大小為 256
H1=np.zeros(256)
H2=np.zeros(256)
H3=np.zeros(256)
H4=np.zeros(256)

# 計算每張影像的直方圖
for y in range(p1.shape[0]): # 遍歷影像的行
    for x in range(p1.shape[1]): # 遍歷影像的列
        H1[p1[y,x]]+=1 # 增加對應灰度值的計數
        H2[p2[y,x]]+=1
        H3[p3[y,x]]+=1
        H4[p4[y,x]]+=1

# 計算累積直方圖
h1=np.cumsum(H1)
h2=np.cumsum(H2)
h3=np.cumsum(H3)
h4=np.cumsum(H4)

# 直方圖均衡化
F1=np.zeros_like(p1,dtype=np.uint8) # 初始化均衡化後的影像，與原影像大小相同
F2=np.zeros_like(p2,dtype=np.uint8)
F3=np.zeros_like(p3,dtype=np.uint8)
F4=np.zeros_like(p4,dtype=np.uint8)

for y in range(p1.shape[0]):
    for x in range(p1.shape[1]):
        # 計算均衡化後的像素值，並轉換為 uint8
        F1[y,x]=((h1[p1[y,x]]-h1.min())/(h1.max()-h1.min())*255).astype(np.uint8)
        F2[y,x]=((h2[p2[y,x]]-h2.min())/(h2.max()-h2.min())*255).astype(np.uint8)
        F3[y,x]=((h3[p3[y,x]]-h3.min())/(h3.max()-h3.min())*255).astype(np.uint8)
        F4[y,x]=((h4[p4[y,x]]-h4.min())/(h4.max()-h4.min())*255).astype(np.uint8)

# 計算均衡化後的直方圖
f1=np.zeros(256)
f2=np.zeros(256)
f3=np.zeros(256)
f4=np.zeros(256)

for y in range(F1.shape[0]): # 遍歷均衡化後的影像
    for x in range(F1.shape[1]):
        f1[F1[y,x]]+=1 # 增加對應灰度值的計數
        f2[F2[y,x]]+=1
        f3[F3[y,x]]+=1
        f4[F4[y,x]]+=1

# 顯示原始影像、直方圖、均衡化後的影像及直方圖
fig,axs=plt.subplots(4,4,figsize=(15,15))

# Cameraman
axs[0,0].imshow(p1,cmap='gray')
axs[0,0].set_title('Original Cameraman')
axs[0,0].axis('off')
axs[0,1].bar(range(256),H1)
axs[0,1].set_title('Histogram')
axs[0,2].imshow(F1,cmap='gray')
axs[0,2].set_title('Histogram Equalized')
axs[0,2].axis('off')
axs[0,3].bar(range(256),f1)
axs[0,3].set_title('Equalized Histogram')

# Jetplane
axs[1,0].imshow(p2,cmap='gray')
axs[1,0].set_title('Original Jetplane')
axs[1,0].axis('off')
axs[1,1].bar(range(256),H2)
axs[1,1].set_title('Histogram')
axs[1,2].imshow(F2,cmap='gray')
axs[1,2].set_title('Histogram Equalized')
axs[1,2].axis('off')
axs[1,3].bar(range(256),f2)
axs[1,3].set_title('Equalized Histogram')

# Lake
axs[2,0].imshow(p3,cmap='gray')
axs[2,0].set_title('Original Lake')
axs[2,0].axis('off')
axs[2,1].bar(range(256),H3)
axs[2,1].set_title('Histogram')
axs[2,2].imshow(F3,cmap='gray')
axs[2,2].set_title('Histogram Equalized')
axs[2,2].axis('off')
axs[2,3].bar(range(256),f3)
axs[2,3].set_title('Equalized Histogram')

# Peppers
axs[3,0].imshow(p4,cmap='gray')
axs[3,0].set_title('Original Peppers')
axs[3,0].axis('off')
axs[3,1].bar(range(256),H4)
axs[3,1].set_title('Histogram')
axs[3,2].imshow(F4,cmap='gray')
axs[3,2].set_title('Histogram Equalized')
axs[3,2].axis('off')
axs[3,3].bar(range(256),f4)
axs[3,3].set_title('Equalized Histogram')

plt.tight_layout(pad=3)
plt.show()