import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 頻率域 Unsharp Masking
def frequency_unsharp(D0,image):
    m,n=image.shape
    
    # 傅立葉轉換
    fft_image=np.fft.fft2(image)
    fft_image_shifted=np.fft.fftshift(fft_image)
    
    # 理想低通濾波器
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
            HH[u,v]=1-H[u,v] # Unsharp Masking
    
    # 濾波運算
    G=fft_image_shifted*HH
    G_shifted_back=np.fft.ifftshift(G)
    sharpened=np.fft.ifft2(G_shifted_back)
    sharpened=np.abs(sharpened)
    
    # 增強並正規化
    sharpened=sharpened/np.max(sharpened)
    result=image+sharpened
    result=np.clip(result,0,1) 
    return result

# 讀取影像並正規化
p1=np.array(Image.open('skeleton_orig.bmp').convert('L'))/255.0
p2=np.array(Image.open('blurry_moon.tif').convert('L'))/255.0
    
# 頻率域處理，(D0=15)為截止頻率
H1=frequency_unsharp(15,p1)
H2=frequency_unsharp(15,p2)

# 結果顯示
plt.figure(figsize=(12,8))
    
plt.subplot(2,2,1)
plt.imshow(p1,cmap='gray')
plt.axis("off")
plt.title('Original: skeleton orig')
    
plt.subplot(2,2,2)
plt.imshow(H1,cmap='gray')
plt.axis("off")
plt.title('Unsharp (Frequency Domain)')
    
plt.subplot(2,2,3)
plt.imshow(p2,cmap='gray')
plt.axis("off")
plt.title('Original: blurry moon')
    
plt.subplot(2,2,4)
plt.imshow(H2,cmap='gray')
plt.axis("off")
plt.title('Unsharp (Frequency Domain)')
    
plt.tight_layout()
plt.show()