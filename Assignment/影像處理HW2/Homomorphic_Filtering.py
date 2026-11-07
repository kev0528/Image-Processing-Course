import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def homomorphic_filter(image,gamma_h=2.0,gamma_l=0.5,d0=30,c=1.0):
    """
    參數：
        image: 影像，2D numpy 陣列 (灰階影像，值域歸一化到 [0,1])
        gamma_h: 高頻增益 (通常>1)
        gamma_l: 低頻增益 (通常<1)
        d0: 高斯濾波器的截止頻率
        c: 濾波器過渡區的銳度調節參數
        
    回傳：
        filtered_image: 處理後的影像，2D numpy 陣列 (值域歸一化到 [0,1])
    """
    # 對影像進行對數變換
    image_log=np.log1p(image) # 計算 log(1+image)
    
    # 傅立葉轉換
    fft_image=np.fft.fft2(image_log) 
    fft_image_shifted=np.fft.fftshift(fft_image) # 平移零頻到中心

    # Homomorphic 濾波器 H(u,v)
    m,n=image.shape
    rows,cols=np.meshgrid(np.arange(-m//2,m//2),np.arange(-n//2,n//2),indexing='ij')
    d_squared=rows**2+cols**2 
    h=(gamma_h-gamma_l)*(1-np.exp(-c*(d_squared/(d0**2))))+gamma_l 

    # 應用濾波器
    filtered_fft=fft_image_shifted*h 
    
    # 逆傅立葉轉換 
    filtered_fft_shifted_back=np.fft.ifftshift(filtered_fft) 
    image_filtered_log=np.fft.ifft2(filtered_fft_shifted_back) 
    image_filtered=np.abs(image_filtered_log) 

    # 對逆對數進行指數變換，恢復原影像範圍
    filtered_image=np.expm1(image_filtered) 
    
    # 歸一化影像到 [0,1] 範圍
    filtered_image=np.clip(filtered_image/np.max(filtered_image),0,1)
    
    return filtered_image

# 讀取影像並轉換為灰階
p1=np.array(Image.open('skeleton_orig.bmp').convert('L'))/255.0 
p2=np.array(Image.open('blurry_moon.tif').convert('L'))/255.0 

# 應用 Homomorphic Filter
filtered_p1=homomorphic_filter(p1,gamma_h=2.0,gamma_l=0.5,d0=30,c=1.0)
filtered_p2=homomorphic_filter(p2,gamma_h=2.0,gamma_l=0.5,d0=30,c=1.0)

# 結果顯示
plt.figure(figsize=(10,5))

plt.subplot(2,2,1)
plt.imshow(p1,cmap='gray')
plt.axis('off')
plt.title('Original: skeleton orig')

plt.subplot(2,2,2)
plt.imshow(filtered_p1,cmap='gray')
plt.axis('off')
plt.title('Filtered: skeleton orig')

plt.subplot(2,2,3)
plt.imshow(p2,cmap='gray')
plt.axis('off')
plt.title('Original: blurry moon')  

plt.subplot(2,2,4)
plt.imshow(filtered_p2,cmap='gray')
plt.axis('off')
plt.title('Filtered: blurry moon')

plt.tight_layout()
plt.show()