import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 高斯模糊
def gaussian_blur(image,kernel_size=5,sigma=1.4):
    # 建立高斯核
    ax=np.arange(-kernel_size//2+1.,kernel_size//2+1.)
    xx,yy=np.meshgrid(ax,ax)
    kernel=np.exp(-(xx**2+yy**2)/(2.*sigma**2))
    kernel=kernel/np.sum(kernel)
    
    # 將影像與高斯核進行卷積
    M,N=image.shape
    padded_image=np.pad(image,((kernel_size//2,kernel_size//2),(kernel_size//2,kernel_size//2)),mode='constant')
    smoothed_image=np.zeros_like(image)
    for i in range(M):
        for j in range(N):
            smoothed_image[i,j]=np.sum(kernel*padded_image[i:i+kernel_size,j:j+kernel_size])
    return smoothed_image

# 梯度計算 (Sobel Operator)
def sobel_filters(image):
    Kx=np.array([[-1, 0, 1], 
                 [-2, 0, 2], 
                 [-1, 0, 1]]) # Sobel 橫向運算子
    Ky=np.array([[ 1, 2, 1], 
                 [ 0, 0, 0], 
                 [-1,-2,-1]]) # Sobel 縱向運算子
    
    M,N=image.shape
    Sx=np.zeros((M,N))
    Sy=np.zeros((M,N))
    
    # 對影像進行邊界填充
    padded_image=np.pad(image,((1,1),(1,1)),mode='constant')
    for i in range(1,M+1):
        for j in range(1,N+1):
            Sx[i-1,j-1]=np.sum(Kx*padded_image[i-1:i+2,j-1:j+2]) # 計算 Sx
            Sy[i-1,j-1]=np.sum(Ky*padded_image[i-1:i+2,j-1:j+2]) # 計算 Sy
    
    gradient_magnitude=np.sqrt(Sx**2+Sy**2) # 梯度大小
    gradient_direction=np.arctan2(Sy,Sx) # 梯度方向
    
    return gradient_magnitude,gradient_direction

# Non-Maximum Suppression
def non_maximum_suppression(gradient_magnitude,gradient_direction):
    M,N=gradient_magnitude.shape
    Z=np.zeros((M,N),dtype=np.float32)
    angle=gradient_direction*180.0 / np.pi # 將方向轉換為角度
    angle[angle<0]+=180  # 將負角度轉為正角度
    
    for i in range(1,M-1):
        for j in range(1,N-1):
            q=255
            r=255

            # 判斷角度範圍
            if (0<=angle[i,j]<22.5) or (157.5<=angle[i,j]<=180): # 0 度
                q = gradient_magnitude[i,j+1]
                r = gradient_magnitude[i,j-1]
            elif 22.5<=angle[i,j]<67.5: # 45 度
                q = gradient_magnitude[i+1,j-1]
                r = gradient_magnitude[i-1,j+1]
            elif 67.5<=angle[i,j]<112.5: # 90 度
                q=gradient_magnitude[i+1,j]
                r=gradient_magnitude[i-1,j]
            elif 112.5<=angle[i,j]<157.5: # 135 度
                q=gradient_magnitude[i-1,j-1]
                r=gradient_magnitude[i+1,j+1]
            
            # 僅保留局部最大值
            if gradient_magnitude[i,j]>=q and gradient_magnitude[i,j]>=r:
                Z[i,j]=gradient_magnitude[i,j]
            else:
                Z[i,j]=0
    return Z

# Double Thresholding
def double_threshold(image,low_ratio=0.05,high_ratio=0.15):
    high_threshold=image.max()*high_ratio # 計算高閾值
    low_threshold=high_threshold*low_ratio # 計算低閾值
    
    strong=255 # 強邊緣
    weak=25 # 弱邊緣
    
    result=np.zeros_like(image,dtype=np.uint8)
    strong_i,strong_j=np.where(image>=high_threshold) # 強邊緣位置
    weak_i,weak_j=np.where((image<=high_threshold) & (image>=low_threshold)) # 弱邊緣位置
    
    result[strong_i,strong_j]=strong
    result[weak_i,weak_j]=weak
    return result,weak, strong

# Edge Tracking by Hysteresis
def edge_tracking_by_hysteresis(image,weak,strong=255):
    M,N=image.shape
    for i in range(1,M-1):
        for j in range(1,N-1):
            if image[i,j]==weak: # 對弱邊緣進行處理
                if (strong in image[i-1:i+2,j-1:j+2]): # 判斷是否與強邊緣相連
                    image[i,j]=strong
                else:
                    image[i,j]=0 # 非連接的弱邊緣置為 0
    return image

# 整合 Canny 邊緣檢測
def canny_edge_detection(image,sigma=1.4,low_threshold_ratio=0.05,high_threshold_ratio=0.15):
    smoothed_image=gaussian_blur(image,sigma=sigma) # 高斯模糊
    gradient_magnitude,gradient_direction=sobel_filters(smoothed_image) # 梯度計算
    non_max_image=non_maximum_suppression(gradient_magnitude,gradient_direction) # Non-Maximum Suppression
    threshold_image,weak,strong=double_threshold(non_max_image,low_threshold_ratio,high_threshold_ratio) # Double Thresholding
    final_image=edge_tracking_by_hysteresis(threshold_image,weak,strong) # Edge Tracking by Hysteresis
    return final_image

# 讀取影像
def load_and_convert_image(image_path):
    image=Image.open(image_path).convert('L') # 將影像轉為灰階
    data=np.array(image)/255.0 # 將像素值從 [0,255] 正規化到 [0,1]
    return data,image

# 讀取影像
r1,itemp1=load_and_convert_image('image1.jpg')
r2,itemp2=load_and_convert_image('image2.jpg')
r3,itemp3=load_and_convert_image('image3.jpg')

# 執行 Canny 邊緣檢測
C1=canny_edge_detection(r1)
C2=canny_edge_detection(r2)
C3=canny_edge_detection(r3)

plt.figure(figsize=(12,8))

# Image 1
plt.subplot(3,2,1)
plt.imshow(itemp1,cmap='gray')
plt.axis('off')
plt.title('Original Image 1')
plt.subplot(3,2,2)
plt.imshow(C1,cmap='gray')
plt.axis('off')
plt.title('Canny Edge Detection 1')

# Image 2
plt.subplot(3,2,3)
plt.imshow(itemp2,cmap='gray')
plt.axis('off')
plt.title('Original Image 2')
plt.subplot(3,2,4)
plt.imshow(C2,cmap='gray')
plt.axis('off')
plt.title('Canny Edge Detection 2')

# Image 3
plt.subplot(3,2,5)
plt.imshow(itemp3,cmap='gray')
plt.axis('off')
plt.title('Original Image 3')
plt.subplot(3,2,6)
plt.imshow(C3,cmap='gray')
plt.axis('off')
plt.title('Canny Edge Detection 3')

plt.tight_layout()
plt.show()