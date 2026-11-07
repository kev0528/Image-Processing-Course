import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# RGB 轉換為 HSI 
def rgb2hsi(image):
    """
    將 RGB 影像轉換為 HSI 色彩空間
    :param image: 輸入影像 (numpy array)
    :return: HSI 影像 (numpy array)
    """
    image=image.astype(np.float32)/255.0 # 正規化至 [0,1]
    r,g,b=image[:,:,0],image[:,:,1],image[:,:,2]

    # 計算色調 (H)
    num=0.5*((r-g)+(r-b))
    den=np.sqrt((r-g)**2+(r-b)*(g-b))+1e-10
    theta=np.arccos(num/den)
    H=np.where(b>g,2*np.pi-theta,theta)
    H/=2*np.pi # 正規化至 [0,1]

    # 計算飽和度 (S)
    min_rgb=np.minimum(np.minimum(r,g),b)
    sum_rgb=r+g+b+1e-10
    S=1-3*min_rgb/sum_rgb
    S[sum_rgb==0]=0 # 處理邊界情況

    # 計算亮度 (I)
    I=sum_rgb/3.0

    return np.stack((H,S,I),axis=-1)

# HSI 直方圖均衡化函數
def histogram(hsi):
    """
    對 HSI 影像的亮度 (I) 通道進行直方圖均衡化
    :param hsi: HSI 影像 (numpy array)
    :return: 均衡化後的 HSI 影像
    """
    H,S,I=hsi[:,:,0],hsi[:,:,1],hsi[:,:,2]
    flat_I=I.flatten()
    sorted_I=np.sort(flat_I)
    unique,counts=np.unique(sorted_I,return_counts=True)

    # 累積分布
    cumulative=np.cumsum(counts)
    normalized_cumulative=(cumulative-cumulative.min())/(cumulative.max()-cumulative.min())

    # 將原始亮度映射至均衡亮度
    eq_map=dict(zip(unique,normalized_cumulative))
    I_eq=np.vectorize(eq_map.get)(I)

    # 組合 H,S 和均衡後的 I
    hsi_eq=np.stack((H,S,I_eq),axis=-1)
    return hsi_eq

# HSI 轉換為 RGB 函數
def hsi2rgb(hsi):
    """
    將 HSI 影像轉換回 RGB 色彩空間
    :param hsi: HSI 影像 (numpy array)
    :return: RGB 影像 (numpy array)
    """
    H,S,I=hsi[:,:,0]*2*np.pi,hsi[:,:,1],hsi[:,:,2]
    R,G,B=np.zeros_like(H),np.zeros_like(H),np.zeros_like(H)

    # RG 區域
    rg_mask=(0<=H)&(H<2*np.pi/3)
    B[rg_mask]=I[rg_mask]*(1-S[rg_mask])
    R[rg_mask]=I[rg_mask]*(1+S[rg_mask]*np.cos(H[rg_mask])/np.cos(np.pi/3-H[rg_mask]))
    G[rg_mask]=3*I[rg_mask]-(R[rg_mask]+B[rg_mask])

    # GB 區域
    gb_mask=(2*np.pi/3<=H)&(H<4*np.pi/3)
    R[gb_mask]=I[gb_mask]*(1-S[gb_mask])
    G[gb_mask]=I[gb_mask]*(1+S[gb_mask]*np.cos(H[gb_mask]-2*np.pi/3)/np.cos(np.pi-H[gb_mask]))
    B[gb_mask]=3*I[gb_mask]-(R[gb_mask]+G[gb_mask])

    # BR 區域
    br_mask=(4*np.pi/3<=H)&(H<=2*np.pi)
    G[br_mask]=I[br_mask]*(1-S[br_mask])
    B[br_mask]=I[br_mask]*(1+S[br_mask]*np.cos(H[br_mask]-4*np.pi/3)/np.cos(5*np.pi/3-H[br_mask]))
    R[br_mask]=3*I[br_mask]-(G[br_mask]+B[br_mask])

    rgb=np.clip(np.stack((R,G,B),axis=-1),0,1) # 確保值在 [0,1] 範圍內
    return (rgb*255).astype(np.uint8)

plt.figure(figsize=(12,8))

p1=np.array(Image.open('house.jpg'))
p2=np.array(Image.open('kitchen.jpg'))
p3=np.array(Image.open('aloe.jpg'))
p4=np.array(Image.open('church.jpg'))

hsi1=rgb2hsi(p1)
hsi2=rgb2hsi(p2)
hsi3=rgb2hsi(p3)
hsi4=rgb2hsi(p4)

hsi1_eq=histogram(hsi1)
hsi2_eq=histogram(hsi2)
hsi3_eq=histogram(hsi3)
hsi4_eq=histogram(hsi4)

rgb1=hsi2rgb(hsi1_eq)
rgb2=hsi2rgb(hsi2_eq)
rgb3=hsi2rgb(hsi3_eq)
rgb4=hsi2rgb(hsi4_eq)

# House
plt.subplot(2,4,1)
plt.imshow(p1)
plt.title('Original house')
plt.axis('off')
plt.subplot(2,4,2)
plt.imshow(rgb1)
plt.title('Enhanced house')
plt.axis('off')

# Kitchen
plt.subplot(2,4,3)
plt.imshow(p2)
plt.title('Original Kitchen')
plt.axis('off')
plt.subplot(2,4,4)
plt.imshow(rgb2)
plt.title('Enhanced Kitchen')
plt.axis('off')

# Aloe
plt.subplot(2,4,5)
plt.imshow(p3)
plt.title('Original Aloe')
plt.axis('off')
plt.subplot(2,4,6)
plt.imshow(rgb3)
plt.title('Enhanced Aloe')
plt.axis('off')

# Church
plt.subplot(2,4,7)
plt.imshow(p4)
plt.title('Original Church')
plt.axis('off')
plt.subplot(2,4,8)
plt.imshow(rgb4)
plt.title('Enhanced Church')
plt.axis('off')

plt.tight_layout()
plt.show()