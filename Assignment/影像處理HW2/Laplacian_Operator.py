import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 頻率域拉普拉斯運算
def frequency_laplacian(image):
    m,n=image.shape
    fft_image=np.fft.fft2(image)
    fft_image_shifted=np.fft.fftshift(fft_image)
    
    # 濾波器
    H=np.zeros((m,n))
    for u in range(m):
        for v in range(n):
            H[u,v]=-4*(np.pi**2)*((u-m/2)**2+(v-n/2)**2)
    
    # 頻域濾波運算
    G=fft_image_shifted*H
    G_shifted_back=np.fft.ifftshift(G)
    result=np.fft.ifft2(G_shifted_back)
    result=np.abs(result) 
    result=result/np.max(result) 
    
    # 銳化結果加回原影像
    result=image+result
    result=np.clip(result,0,1) # 正規化至 [0,1]
    return result

# 讀取影像並轉換為灰階
p1=np.array(Image.open('skeleton_orig.bmp').convert('L'))/255.0
p2=np.array(Image.open('blurry_moon.tif').convert('L'))/255.0
    
L1_frequency=frequency_laplacian(p1)
L2_frequency=frequency_laplacian(p2)
    
# 結果顯示
plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.imshow(p1,cmap='gray')
plt.axis('off')
plt.title('Original: skeleton orig')
    
plt.subplot(2,2,2)
plt.imshow(L1_frequency,cmap='gray')
plt.axis('off')
plt.title('Laplacian (Frequency Domain)')
    
plt.subplot(2,2,3)
plt.imshow(p2,cmap='gray')
plt.axis('off')
plt.title('Original: blurry moon')
    
plt.subplot(2,2,4)
plt.imshow(L2_frequency,cmap='gray')
plt.axis('off')
plt.title('Laplacian (Frequency Domain)')
    
plt.tight_layout()
plt.show()