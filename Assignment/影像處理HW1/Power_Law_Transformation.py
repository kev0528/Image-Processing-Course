import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 影像讀取(.convert('L')保證了後續處理在單一亮度通道上進行，而不是在 RGB 格式上操作。)
p1=np.array(Image.open('Cameraman.bmp').convert('L'))
p2=np.array(Image.open('Jetplane.bmp').convert('L'))
p3=np.array(Image.open('Lake.bmp').convert('L'))
p4=np.array(Image.open('Peppers.bmp').convert('L'))

# 正規化至[0,1]
g1=p1/255.0
g2=p2/255.0
g3=p3/255.0
g4=p4/255.0

# 設置gamma轉換參數
c=1
gamma=0.2

# 應用 Power-Law 轉換(𝑠=𝑐𝑟^𝛾)
s1=c*(g1**gamma)
s2=c*(g2**gamma)
s3=c*(g3**gamma)
s4=c*(g4**gamma)

fig,axs=plt.subplots(4,2,figsize=(10,15))

# 顯示原始圖片和處理後的圖片
axs[0,0].imshow(p1,cmap='gray')
axs[0,0].set_title('Original Cameraman')
axs[0,0].axis('off')

axs[0,1].imshow(s1,cmap='gray')
axs[0,1].set_title('Power Law Transformation')
axs[0,1].axis('off')

axs[1,0].imshow(p2,cmap='gray')
axs[1,0].set_title('Original Jetplane')
axs[1,0].axis('off')

axs[1,1].imshow(s2,cmap='gray')
axs[1,1].set_title('Power Law Transformation')
axs[1,1].axis('off')

axs[2,0].imshow(p3,cmap='gray')
axs[2,0].set_title('Original Lake')
axs[2,0].axis('off')

axs[2,1].imshow(s3,cmap='gray')
axs[2,1].set_title('Power Law Transformation')
axs[2,1].axis('off')

axs[3,0].imshow(p4,cmap='gray')
axs[3,0].set_title('Original Peppers')
axs[3,0].axis('off')

axs[3,1].imshow(s4,cmap='gray')
axs[3,1].set_title('Power Law Transformation')
axs[3,1].axis('off')

plt.show()