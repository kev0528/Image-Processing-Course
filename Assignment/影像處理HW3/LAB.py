from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def rgb2lab(data):
    # data為uint8的RGB影像
    # 歸一化至[0,1]
    R=data[:,:,0].astype(np.float64)/255.0
    G=data[:,:,1].astype(np.float64)/255.0
    B=data[:,:,2].astype(np.float64)/255.0

    M,N=R.shape
    s=M*N

    # 將R,G,B展平成一維向量
    RGB=np.stack([R.flatten(),G.flatten(),B.flatten()],axis=0)

    # XYZ轉換矩陣
    MAT=np.array([[0.412453,0.357580,0.180423],
                  [0.212671,0.715160,0.072169],
                  [0.019334,0.119193,0.950227]])

    XYZ=MAT@RGB

    # 正規化
    X=XYZ[0,:]/0.950456
    Y=XYZ[1,:]
    Z=XYZ[2,:]/1.088754

    T=0.008856
    XT=X>T
    YT=Y>T
    ZT=Z>T

    Y3=np.power(Y,1/3)

    fX=XT*(X**(1/3))+(~XT)*(7.787*X+16/116)
    fY=YT*Y3+(~YT)*(7.787*Y+16/116)
    fZ=ZT*(Z**(1/3))+(~ZT)*(7.787*Z+16/116)

    L=YT*(116*Y3-16)+(~YT)*(903.3*Y)
    a=500*(fX-fY)
    b=200*(fY-fZ)

    L=L.reshape(M,N)
    a=a.reshape(M,N)
    b=b.reshape(M,N)

    Lab=np.stack([L,a,b],axis=2)
    return Lab

def lab2rgb(g):
    # g為Lab影像(M,N,3)
    L=g[:,:,0]
    a=g[:,:,1]
    b=g[:,:,2]

    M,N=L.shape
    s=M*N

    Lf=L.flatten()
    af=a.flatten()
    bf=b.flatten()

    # 將Lab轉回XYZ
    # Y
    fY=((Lf+16)/116.0)**3
    T1=0.008856
    YT=fY>T1
    fY=YT*fY+(~YT)*(Lf/903.3)
    Y=fY

    fYc=YT*(fY**(1/3))+(~YT)*(7.787*fY+16/116)

    # X
    fX=af/500.0+fYc
    T2=0.206893
    XT=fX>T2
    X=XT*(fX**3)+(~XT)*((fX-16/116)/7.787)

    # Z
    fZ=fYc-bf/200.0
    ZT=fZ>T2
    Z=ZT*(fZ**3)+(~ZT)*((fZ-16/116)/7.787)

    X=X*0.950456
    Z=Z*1.088754

    # XYZ轉RGB
    MAT=np.array([[ 3.240479,-1.537150,-0.498535],
                  [-0.969256, 1.875992, 0.041556],
                  [ 0.055648,-0.204043, 1.057311]])

    XYZ=np.stack([X,Y,Z],axis=0)
    RGB=MAT@XYZ
    RGB=np.clip(RGB,0,1)

    R=RGB[0,:].reshape(M,N)
    G=RGB[1,:].reshape(M,N)
    B=RGB[2,:].reshape(M,N)

    data=np.round(np.stack([R,G,B],axis=2)*255).astype(np.uint8)
    return data

def power_law(data,y,c,L,H):
    # data: Lab
    # y: gamma
    # c: c值
    # L,H:閾值
    ta=data[:,:,0]
    M,N=ta.shape
    g=np.copy(data)
    h=ta/100.0
    for i in range(M):
        for j in range(N):
            if (ta[i,j]>L) and (ta[i,j]<H):
                g[i,j,0]=c*(h[i,j]**y)*100
            else:
                g[i,j,0]=h[i,j]*100
    return g

def histogram(data,z):
    z_idx=z-1
    M,N=data[:,:,z_idx].shape
    ta=data[:,:,z_idx]

    # 將像素值排序
    ht=np.sort(ta.flatten())
    # 找出 unique 值及對應次數
    ha,counts=np.unique(ht,return_counts=True)
    hr=np.cumsum(counts)

    # 正規化
    hz=((hr-hr.min())/(hr.max()-hr.min()))*100

    g=np.copy(data)
    for i in range(M):
        for j in range(N):
            val=ta[i,j]
            # 找出val在ha中的位置
            m=np.searchsorted(ha,val)
            g[i,j,z_idx]=hz[m]

    return g

itemp1=np.array(Image.open('house.jpg'))
itemp2=np.array(Image.open('kitchen.jpg'))
itemp3=np.array(Image.open('aloe.jpg'))
itemp4=np.array(Image.open('church.jpg'))

t1=rgb2lab(itemp1)
t2=rgb2lab(itemp2)
t3=rgb2lab(itemp3)
t4=rgb2lab(itemp4)

r1=histogram(t1,1)
r2=histogram(t2,1)
r3=power_law(t3,0.35,1,0,70)
r4=power_law(t4,0.35,1,0,70)

g1=lab2rgb(r1)
g2=lab2rgb(r2)
g3=lab2rgb(r3)
g4=lab2rgb(r4)

plt.figure(figsize=(12,8))

# House
plt.subplot(2,4,1)
plt.imshow(itemp1.astype(np.uint8))
plt.axis('off')
plt.title('Original House')
plt.subplot(2,4,2)
plt.imshow(g1)
plt.axis('off')
plt.title('Sharpened House')

# Kitchen
plt.subplot(2,4,3)
plt.imshow(itemp2.astype(np.uint8))
plt.axis('off')
plt.title('Original Kitchen')
plt.subplot(2,4,4)
plt.imshow(g2)
plt.axis('off')
plt.title('Sharpened Kitchen')

# Aloe
plt.subplot(2,4,5)
plt.imshow(itemp3.astype(np.uint8))
plt.axis('off')
plt.title('Original Aloe')
plt.subplot(2,4,6)
plt.imshow(g3)
plt.axis('off')
plt.title('Sharpened Aloe')

# Church
plt.subplot(2,4,7)
plt.imshow(itemp4.astype(np.uint8))
plt.axis('off')
plt.title('Original Church')
plt.subplot(2,4,8)
plt.imshow(g4)
plt.axis('off')
plt.title('Sharpened Church')

plt.tight_layout()
plt.show()