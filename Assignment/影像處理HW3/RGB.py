import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Power Law 函數
def power_law(image,gamma,c):
    """
    Power Law  
    :image: 原始影像 
    :gamma: Gamma 值
    :c: 增益常數
    :return: 增強後影像 
    """
    r=image.astype(np.float32) 
    g=c*(r**gamma) # Power Law 
    g=np.clip(g,0,255) # 限制像素值範圍為 [0,255]
    return g.astype(np.uint8) 

# 載入影像
p1=np.array(Image.open('house.jpg'))
p2=np.array(Image.open('kitchen.jpg'))
p3=np.array(Image.open('aloe.jpg'))
p4=np.array(Image.open('church.jpg'))

r1=power_law(p1,0.95,1) 
r2=power_law(p2,0.95,1) 
r3=power_law(p3,1.5,1) 
r4=power_law(p4,1.5,1) 

plt.figure(figsize=(12,8))

# House
plt.subplot(2,4,1)
plt.imshow(p1)
plt.title('Original House')
plt.axis('off')
plt.subplot(2,4,2)
plt.imshow(r1)
plt.title('Enhanced House')
plt.axis('off')

# Kitchen
plt.subplot(2,4,3)
plt.imshow(p2)
plt.title('Original Kitchen')
plt.axis('off')
plt.subplot(2,4,4)
plt.imshow(r2)
plt.title('Enhanced Kitchen')
plt.axis('off')

# Aloe
plt.subplot(2,4,5)
plt.imshow(p3)
plt.title('Original Aloe')
plt.axis('off')
plt.subplot(2,4,6)
plt.imshow(r3)
plt.title('Enhanced Aloe')
plt.axis('off')

# Church
plt.subplot(2,4,7)
plt.imshow(p4)
plt.title('Original Church')
plt.axis('off')
plt.subplot(2,4,8)
plt.imshow(r4)
plt.title('Enhanced Church')
plt.axis('off')

plt.tight_layout()
plt.show()