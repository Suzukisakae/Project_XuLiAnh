import cv2
import numpy as np
import matplotlib.pyplot as plt
L = 256

# CHuong 3
def Negative(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    for x in range(M):
        for y in range(0, N):
            r = imgin[x, y]
            s = L - 1 - r
            imgout[x, y] = np.uint8(s)
    return imgout

def NegativeColor(imgin):
    M, N = imgin.shape
    C = 3
    imgout = np.zeros((M, N, C), np.uint8) + 255
    for x in range(0, M):
        for y in range(0, N):
            b = imgin[x, y, 0]
            g = imgin[x, y, 1]
            r = imgin[x, y, 2]

            b = L - 1 - b
            g = L - 1 - g
            r = L - 1 - r

            imgout[x, y, 0] = np.uint8(b)
            imgout[x, y, 1] = np.uint8(g)
            imgout[x, y, 2] = np.uint8(r)

    return imgout

def Logarit(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8) + 255
    c = (L-1)/np.log(1.0*L)
    for x in range(0,M):
        for y in range(0, N):
            r = imgin[x,y]
            s = c*np.log(1.0 + r)
            imgout[x,y] = np.uint8(s)
    return imgout

def Power(imgin):
    gamma = 5.0
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8) + 255
    c = np.power(L-1.0,1.0 - gamma)
    for x in range(0,M):
        for y in range(0, N):
            r = imgin[x,y]
            s = c*np.power(1.0*r,gamma)
            imgout[x,y] = np.uint8(s)
    return imgout

def PiecewiseLinear(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8) + 255
    rmin, rmax, _, _ = cv2.minMaxLoc(imgin)
    r1 = rmin
    s1 = 0
    r2 = rmax
    s2 = L - 1
    for x in range(0,M):
        for y in range(0, N):
            r = imgin[x,y]
            # Giai doan 1
            if r < r1:
                s = s1*r/r1
            elif r < r2:
                s = (s2 - s1)*(r - r1)/(r2 - r1) + s1
            # Giai doan 2
            else:
                s = (L-1 - s2)*(r - r2)/(L-1 - r2) + s2
            imgout[x,y] = np.uint8(s)
    return imgout

def Histogram(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, L), np.uint8) + 255
    h = np.zeros(L, np.uint)
    for x in range(0,M):
        for y in range(0, N):
            r = imgin[x,y]
            h[r] = h[r] + 1
    p = h/(M*N)
    scale = 3000
    for r in range(0,L):
        cv2.line(imgout, (r, M -1), (r,M -1 - int(scale*p[r])), (0, 0, 0))
    return imgout

def HistEqual(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8) + 255
    h = np.zeros(L, np.int32)
    for x in range(0,M):
        for y in range(0, N):
            r = imgin[x,y]
            h[r] = h[r] + 1
    p = h/(M*N) 
    s = np.zeros(L, np.float64)
    for k in range(0,L):
        for j in range(0, k +1):
            s[k] = s[k] + p[j]
        s[k] = (L-1)*s[k]
    for x in range(0,M):
        for y in range(0, N):
            r = imgin[x,y]
            imgout[x,y] = np.uint8(s[r])
    return imgout

def LocalHist(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8) + 255

    m = 3
    n = 3
    a = m // 2
    b = n // 2
    w = np.zeros((m,n), np.uint8)

    for x in range(a, M - a):
        for y in range(b, N - b):
            w = imgin[x - a:x + a + 1, y - b:y + b + 1]
            # for s in range(-a, a + 1):
            #     for t in range(-b, b + 1):
            #         w[s + a, t + b] = imgin[x + s, y + t]
            w = cv2.equalizeHist(w)
            imgout[x, y] = w[a, b]
    return imgout

def HistStat(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8) + 255
    # sum = 0.0
    # for x in range(0,M):
    #     for y in range(0, N):
    #         r = imgin[x,y]
    #         sum = sum + r
    # mean = sum/(M*N)

    # variance = 0.0
    # for x in range(0,M):
    #     for y in range(0, N):
    #         r = imgin[x,y]
    #         variance = variance + (r - mean)**2
    # variance = variance/(M*N)

    # sigma = np.sqrt(variance)
    # print("mean = ", mean)
    # print("sigma = ", sigma)

    mean, stdDev = cv2.meanStdDev(imgin)
    # print("mean = ", mean)
    # print("stdDev = ", stdDev)
    mG = mean[0,0]
    sigmaG = stdDev[0,0]

    m = 3
    n = 3
    a = m // 2
    b = n // 2
    w = np.zeros((m,n), np.uint8)
    C = 22.8
    k0 = 0.0
    k1 = 0.1
    k2 = 0.0
    k3 = 0.1

    for x in range(a, M - a):
        for y in range(b, N - b):
            w = imgin[x - a:x + a + 1, y - b:y + b + 1]
            mean, stdDev = cv2.meanStdDev(w)
            msxy = mean[0,0]
            sigmasxy = stdDev[0,0]
            if (k0*mG <= msxy <= k1*mG) and (k2*sigmaG <= sigmasxy <= k3*sigmaG):
                r = imgin[x,y]
                imgout[x,y] = np.uint8(C*r)
            else:
                imgout[x,y] = imgin[x,y]

    return imgout

def MyFilter2D(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8) + 255
    m = 11
    n = 11
    w = np.zeros((m,n), np.float32) + 1.0/(m*n)
    a = m // 2
    b = n // 2
    for x in range(a, M - a):
        for y in range(b, M - b):            
            r = 0.0
            for s in range(-a, a + 1):
                for t in range(-b, b + 1):
                    x_moi = (x + s)%M
                    y_moi = (y + t)%N
                    r = r + w[s + a, t + b]*imgin[x_moi, y_moi]
            if r < 0:
                r = 0
            if r > L - 1:
                r = L - 1
            imgout[x, y] = np.uint8(r)
            
    return imgout

def MySmooth(imgin):
    m = 11
    n = 11
    w = np.zeros((m,n), np.float32) + 1.0/(m*n)
    imgout = cv2.filter2D(imgin, cv2.CV_8UC1, w)
    return imgout

def MyMedianFilter(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    m = 3
    n = 3                              
    a = m // 2
    b = n // 2
    for x in range(a, M - a):
        for y in range(b, N - b):  # Corrected the range for y
            w = imgin[x - a : x + a + 1, y - b : y + b + 1]
            w = np.sort(w.reshape((1, m*n)))
            imgout[x, y] = w[0, m*n // 2]        
    return imgout

def OnSharpen(imgin):
    w = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.int32)
    temp = cv2.filter2D(imgin, cv2.CV_32FC1, w)
    result = imgin - temp
    result = np.clip(result, 0, L - 1)
    imgout = result.astype(np.uint8)
    return imgout

def Gradient(imgin):
    wx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.int32)
    wy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
    gx = cv2.filter2D(imgin, cv2.CV_32FC1, wx);
    gy = cv2.filter2D(imgin, cv2.CV_32FC1, wy);
    g = abs(gx) + abs(gy)
    imgout = np.clip(g, 0, L - 1).astype(np.uint8)
    return imgout


# CHuong 4
def Spectrum(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    
    # Bước 1 và 2: 
    # Tạo ảnh mới có kích thước PxQ
    # và thêm số 0 và phần mở rộng
    fp = np.zeros((P,Q), np.float64)
    fp[:M,:N] = imgin
    fp = fp/(L-1)

    # Bước 3:
    # Nhân (-1)^(x+y) để dời vào tâm ảnh
    for x in range(0, M):
        for y in range(0, N):
            if (x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]

    # Bước 4:
    # Tính DFT    
    F = cv2.dft(fp, flags = cv2.DFT_COMPLEX_OUTPUT)

    # Tính spectrum
    S = np.sqrt(F[:,:,0]**2 + F[:,:,1]**2)
    S = np.clip(S, 0, L-1)
    S = S.astype(np.uint8)
    return S

def HighpassFilter(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    
    # Bước 1 và 2: 
    # Tạo ảnh mới có kích thước PxQ
    # và thêm số 0 vào phần mở rộng
    fp = np.zeros((P,Q), np.float32)
    fp[:M,:N] = imgin

    # Bước 3:
    # Nhân (-1)^(x+y) để dời vào tâm ảnh
    for x in range(0, M):
        for y in range(0, N):
            if (x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]
    # Bước 4:
    # Tính DFT    
    F = cv2.dft(fp, flags = cv2.DFT_COMPLEX_OUTPUT)

    # Bước 5: 
    # Tạo bộ lọc H thực High Pass Butterworth
    H = np.zeros((P,Q), np.float32)
    D0 = 60
    n = 2
    for u in range(0, P):
        for v in range(0, Q):
            Duv = np.sqrt((u-P//2)**2 + (v-Q//2)**2)
            if Duv > 0:
                H[u,v] = 1.0/(1.0 + np.power(D0/Duv,2*n))
    # Bước 6:
    # G = F*H nhân từng cặp
    G = F.copy()
    for u in range(0, P):
        for v in range(0, Q):
            G[u,v,0] = F[u,v,0]*H[u,v]
            G[u,v,1] = F[u,v,1]*H[u,v]
    
    # Bước 7:
    # IDFT
    g = cv2.idft(G, flags = cv2.DFT_SCALE)
    # Lấy phần thực
    gp = g[:,:,0]
    # Nhân với (-1)^(x+y)
    for x in range(0, P):
        for y in range(0, Q):
            if (x+y)%2 == 1:
                gp[x,y] = -gp[x,y]
    # Bước 8:
    # Lấy kích thước ảnh ban đầu
    imgout = gp[0:M,0:N]
    imgout = np.clip(imgout,0,L-1)
    imgout = imgout.astype(np.uint8)
    return imgout

def CreateNotchRejectFilter(P,Q):

    u1, v1 = 44, 58
    u2, v2 = 40, 119
    u3, v3 = 86, 59
    u4, v4 = 82, 119

    D0 = 10
    n = 2
    H = np.ones((P,Q), np.complex128)
    for u in range(0, P):
        for v in range(0, Q):
            h = 1.0
            # Bộ lọc u1, v1
            Duv = np.sqrt((u-u1)**2 + (v-v1)**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0
            Duv = np.sqrt((u-(P-u1))**2 + (v-(Q-v1))**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0

            # Bộ lọc u2, v2
            Duv = np.sqrt((u-u2)**2 + (v-v2)**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0
            Duv = np.sqrt((u-(P-u2))**2 + (v-(Q-v2))**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0

            # Bộ lọc u3, v3
            Duv = np.sqrt((u-u3)**2 + (v-v3)**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0
            Duv = np.sqrt((u-(P-u3))**2 + (v-(Q-v3))**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0

            # Bộ lọc u4, v4
            Duv = np.sqrt((u-u4)**2 + (v-v4)**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0
            Duv = np.sqrt((u-(P-u4))**2 + (v-(Q-v4))**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0
            H[u,v] = h
    return H

def DrawNotchRejectFilter():
    H = CreateNotchRejectFilter(250,180)
    H = H*(L-1)
    H = H.astype(np.uint8)
    return H

def RemoveMoire(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)

    # Zero-padding
    fp = np.zeros((P, Q), np.complex128)
    fp[:M, :N] = imgin

    # Shift to center
    fp *= (-1) ** np.fromfunction(lambda x, y: x + y, (P, Q))

    # DFT
    F = np.fft.fft2(fp)

    # Create and apply Notch Reject Filter
    H = CreateNotchRejectFilter(P, Q)  # Assuming CreateNotchRejectFilter is implemented
    G = F * H

    # IDFT
    g = np.fft.ifft2(G)

    # Shift back
    g *= (-1) ** np.fromfunction(lambda x, y: x + y, (P, Q))

    # Crop to the original size
    imgout = np.real(g[:M, :N])

    # Clip and convert to uint8
    imgout = np.clip(imgout, 0, 255).astype(np.uint8)

    return imgout

# CHuong 5
def CreateMotionfilter(M, N):
    H = np.zeros((M,N), np.complex128)
    a = 0.1
    b = 0.1
    T = 1
    for u in range(0, M):
        for v in range(0, N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            if np.abs(phi) < 1.0e-6:
                RE = T*np.cos(phi)
                IM = -T*np.sin(phi)
            else:
                RE = T*np.sin(phi)/phi*np.cos(phi)
                IM = -T*np.sin(phi)/phi*np.sin(phi)
            H.real[u,v] = RE
            H.imag[u,v] = IM
    return H

def CreateMotionNoise(imgin):
    M, N = imgin.shape
    f = imgin.astype(np.float64)
    # Buoc 1: DFT
    F = np.fft.fft2(f)
    # Buoc 2: Shift vao the center of the image
    F = np.fft.fftshift(F)

    # Buoc 3: Tao bo loc H
    H = CreateMotionfilter(M, N)

    # Buoc 4: Nhan F voi H
    G = F*H

    # Buoc 5: Shift return
    G = np.fft.ifftshift(G)

    # Buoc 6: IDFT
    g = np.fft.ifft2(G)
    g = g.real
    g = np.clip(g, 0, L-1)
    g = g.astype(np.uint8)
    return g

def CreateInverseMotionfilter(M, N):
    H = np.zeros((M,N), np.complex128)
    a = 0.1
    b = 0.1
    T = 1
    phi_prev = 0
    for u in range(0, M):
        for v in range(0, N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            if np.abs(phi) < 1.0e-6:
                RE = np.cos(phi)/T
                IM = np.sin(phi)/T
            else:
                if np.abs(np.sin(phi)) < 1.0e-6:
                    phi = phi_prev
                RE = phi/(T*np.sin(phi))*np.cos(phi)
                IM = phi/(T*np.sin(phi))*np.sin(phi)
            H.real[u,v] = RE
            H.imag[u,v] = IM
            phi_prev = phi
    return H

def DenoiseMotion(imgin):
    M, N = imgin.shape
    f = imgin.astype(np.float64)
    # Buoc 1: DFT
    F = np.fft.fft2(f)
    # Buoc 2: Shift vao the center of the image
    F = np.fft.fftshift(F)

    # Buoc 3: Tao bo loc H
    H = CreateInverseMotionfilter(M, N)

    # Buoc 4: Nhan F voi H
    G = F*H

    # Buoc 5: Shift return
    G = np.fft.ifftshift(G)

    # Buoc 6: IDFT
    g = np.fft.ifft2(G)
    g = g.real
    g = np.clip(g, 0, L-1)
    g = g.astype(np.uint8)
    return g

# CHuong 9
def ConnectedComponent(imgin, text):
    ret, temp = cv2.threshold(imgin, 200, L-1, cv2.THRESH_BINARY)
    temp = cv2.medianBlur(temp, 7)
    dem, label = cv2.connectedComponents(temp)
    print('Co %d thanh phan lien thong' % (dem-1))
    text = text + 'Có %d thành phần liên thông' % (dem-1)
    a = np.zeros(dem, np.int64)
    M, N = label.shape
    color = 150
    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            a[r] += 1
            if r > 0:
                label[x, y] += color
    for r in range(1, dem):
        print('%4d %10d' % (r, a[r]))
    return label.astype(np.uint8), text

# trả về số hạt gạo
def CountRice1(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
    temp = cv2.morphologyEx(imgin, cv2.MORPH_TOPHAT, w)
    ret, temp = cv2.threshold(temp, 100, L-1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    temp = cv2.medianBlur(temp, 3)
    dem, label = cv2.connectedComponents(temp)
    return dem


def CountRice(imgin, text):
    w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
    temp = cv2.morphologyEx(imgin, cv2.MORPH_TOPHAT, w)
    ret, temp = cv2.threshold(temp, 100, L-1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    temp = cv2.medianBlur(temp, 3)
    dem, label = cv2.connectedComponents(temp)
    print('Co %d hat gao' % dem)
    text = text + 'Có %d hạt gạo' % dem
    print(text)
    a = np.zeros(dem, np.int64)
    M, N = label.shape
    color = 150
    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            a[r] += 1
            if r > 0:
                label[x, y] += color
    for r in range(0, dem):
        print('%4d %10d' % (r, a[r]))

    max_val = a[1]
    rmax = 1
    for r in range(2, dem):
        if a[r] > max_val:
            max_val = a[r]
            rmax = r

    xoa = np.array([], np.int64)
    for r in range(1, dem):
        if a[r] < 0.5 * max_val:
            xoa = np.append(xoa, r)

    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            if r > 0:
                r -= color
                if r in xoa:
                    label[x, y] = 0
    return label.astype(np.uint8), text