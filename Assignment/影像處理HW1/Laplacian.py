import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 影像讀取(.convert('L')保證了後續處理在單一亮度通道上進行，而不是在 RGB 格式上操作，並正規化至[0,1])
k=np.array(Image.open('Cameraman.bmp').convert('L'))/255.0
j=np.array(Image.open('Jetplane.bmp').convert('L'))/255.0
l=np.array(Image.open('Lake.bmp').convert('L'))/255.0
p=np.array(Image.open('Peppers.bmp').convert('L'))/255.0

# Laplacian 卷積核
laplacian_filter=np.array([[-1,-1,-1],
                           [-1, 8,-1],
                           [-1,-1,-1]])

# 卷積應用函數(np.pad()在圖像的邊緣處填充零，防止邊緣效應，然後對圖像進行逐像素卷積運算，得到濾波後的圖像)
def apply_laplacian_filter(image,filter_kernel):
    filtered_image=np.zeros_like(image)
    padded_image=np.pad(image,pad_width=1,mode='constant',constant_values=0)
    for y in range(1,image.shape[0]+1):
        for x in range(1,image.shape[1]+1):
            region=padded_image[y-1:y+2,x-1:x+2]
            filtered_image[y-1,x-1]=np.sum(region*filter_kernel)
    return filtered_image

resp3=apply_laplacian_filter(k,laplacian_filter)
resp2=apply_laplacian_filter(j,laplacian_filter)
resp1=apply_laplacian_filter(l,laplacian_filter)
resp=apply_laplacian_filter(p,laplacian_filter)

# 正規化濾波後的影像
def normalize_image(image):
    min_val=np.min(image)
    max_val=np.max(image)
    return (image-min_val)/(max_val-min_val)

resp3=normalize_image(resp3)
resp2=normalize_image(resp2)
resp1=normalize_image(resp1)
resp=normalize_image(resp)

# 銳化影像
sharpened3=k+resp3
sharpened2=j+resp2
sharpened1=l+resp1
sharpened=p+resp

# 正規化銳化後的影像
sharpened3=normalize_image(sharpened3)
sharpened2=normalize_image(sharpened2)
sharpened1=normalize_image(sharpened1)
sharpened=normalize_image(sharpened)

# 顯示原始影像、Laplacian後的影像和銳化後的影像
fig,axes=plt.subplots(4,3,figsize=(15,15))

# Cameraman
axes[0,0].imshow(k,cmap='gray')
axes[0,0].set_title('Original Cameraman')
axes[0,0].axis('off')
axes[0,1].imshow(resp3,cmap='gray')
axes[0,1].set_title('Laplacian filtered image')
axes[0,1].axis('off')
axes[0,2].imshow(sharpened3,cmap='gray')
axes[0,2].set_title('Sharpened image')
axes[0,2].axis('off')

# Jetplane
axes[1,0].imshow(j,cmap='gray')
axes[1,0].set_title('Original Jetplane')
axes[1,0].axis('off')
axes[1,1].imshow(resp2,cmap='gray')
axes[1,1].set_title('Laplacian filtered image')
axes[1,1].axis('off')
axes[1,2].imshow(sharpened2,cmap='gray')
axes[1,2].set_title('Sharpened image')
axes[1,2].axis('off')

# Lake
axes[2,0].imshow(l,cmap='gray')
axes[2,0].set_title('Original Lake')
axes[2,0].axis('off')
axes[2,1].imshow(resp1,cmap='gray')
axes[2,1].set_title('Laplacian filtered image')
axes[2,1].axis('off')
axes[2,2].imshow(sharpened1,cmap='gray')
axes[2,2].set_title('Sharpened image')
axes[2,2].axis('off')

# Peppers
axes[3,0].imshow(p,cmap='gray')
axes[3,0].set_title('Original Peppers')
axes[3,0].axis('off')
axes[3,1].imshow(resp,cmap='gray')
axes[3,1].set_title('Laplacian filtered image')
axes[3,1].axis('off')
axes[3,2].imshow(sharpened,cmap='gray')
axes[3,2].set_title('Sharpened image')
axes[3,2].axis('off')

plt.tight_layout(pad=3)
plt.show()