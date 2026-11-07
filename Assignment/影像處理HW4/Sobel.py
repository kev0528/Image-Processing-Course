import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Sobel 邊緣檢測函數
def s_sobel(data):
    M,N=data.shape
    Sx=np.zeros((M,N))
    Sy=np.zeros((M,N))
    
    for y in range(1,N-1):
        for x in range(1,M-1):
            Sx[x,y]=(+data[x-1,y-1]+2*data[x,y-1]+data[x+1,y-1]
                        -data[x-1,y+1]-2*data[x,y+1]-data[x+1,y+1])
            Sy[x,y]=(-data[x-1,y-1]-2*data[x-1,y]-data[x-1,y+1]
                        +data[x+1,y-1]+2*data[x+1,y]+data[x+1,y+1])
    
    G=np.sqrt(Sx**2+Sy**2) # 計算梯度強度
    return G

# 讀取影像
def load_and_convert_image(image_path):
    image=Image.open(image_path).convert('L') # 轉為灰階
    data=np.array(image)/255.0 # 正規化到 [0,1]
    return data,image

r1,itemp1=load_and_convert_image('image1.jpg')
r2,itemp2=load_and_convert_image('image2.jpg')
r3,itemp3=load_and_convert_image('image3.jpg')

# Sobel 邊緣檢測
L1=s_sobel(r1)
L2=s_sobel(r2)
L3=s_sobel(r3)

# 顯示結果
plt.figure(figsize=(12,8))

# Image 1
plt.subplot(3,2,1)
plt.imshow(itemp1,cmap='gray')
plt.axis('off')
plt.title('Original Image 1')
plt.subplot(3,2,2)
plt.imshow(L1,cmap='gray')
plt.axis('off')
plt.title('Sobel Edge Detection 1')

# Image 2
plt.subplot(3,2,3)
plt.imshow(itemp2,cmap='gray')
plt.axis('off')
plt.title('Original Image 2')
plt.subplot(3,2,4)
plt.imshow(L2,cmap='gray')
plt.axis('off')
plt.title('Sobel Edge Detection 2')

# Image 3
plt.subplot(3,2,5)
plt.imshow(itemp3,cmap='gray')
plt.axis('off')
plt.title('Original Image 3')
plt.subplot(3,2,6)
plt.imshow(L3,cmap='gray')
plt.axis('off')
plt.title('Sobel Edge Detection 3')

plt.tight_layout()
plt.show()