import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 頻率域 High-Boost Filtering
def frequency_high_boost(A,D0,image):
    m,n=image.shape

    # 傅立葉變換
    fft_image=np.fft.fft2(image)
    fft_image_shifted=np.fft.fftshift(fft_image)

    # 濾波器
    H=np.zeros((m,n))
    HH=np.zeros((m,n))
    D=np.zeros((m,n))
    for u in range(m):
        for v in range(n):
            D[u,v]=np.sqrt((u-m/2)**2+(v-n/2)**2)
            if D[u,v]<=D0:
                H[u,v]=1
            else:
                H[u,v]=0
            HH[u,v]=A-H[u,v] # High-Boost Filtering

    # 濾波運算
    G=fft_image_shifted*HH
    G_shifted_back=np.fft.ifftshift(G)
    high_boost=np.fft.ifft2(G_shifted_back)
    high_boost=np.abs(high_boost)

    # 增強並正規化
    high_boost=high_boost/np.max(high_boost)
    result=image+high_boost
    result=np.clip(result,0,1) 
    return result

# 讀取影像並正規化
p1=np.array(Image.open('skeleton_orig.bmp').convert('L'))/255.0
p2=np.array(Image.open('blurry_moon.tif').convert('L'))/255.0

# 頻率域處理
H1=frequency_high_boost(3,15,p1)
H2=frequency_high_boost(3,15,p2)

# 結果顯示
plt.figure(figsize=(12,8))
    
plt.subplot(2,2,1)
plt.imshow(p1,cmap='gray')
plt.axis('off')
plt.title('Original: skeleton orig')
    
plt.subplot(2,2,2)
plt.imshow(H1,cmap='gray')
plt.axis('off')
plt.title('High-Boost (Frequency Domain)')
    
plt.subplot(2,2,3)
plt.imshow(p2,cmap='gray')
plt.axis('off')
plt.title('Original: blurry moon')
    
plt.subplot(2,2,4)
plt.imshow(H2,cmap='gray')
plt.axis('off')
plt.title('High-Boost (Frequency Domain)')
    
plt.tight_layout()
plt.show()