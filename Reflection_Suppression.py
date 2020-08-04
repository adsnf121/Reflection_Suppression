import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

image = cv2.imread("/content/drive/My Drive/画像/dog.png")
image2 = cv2.imread("/content/drive/My Drive/画像/rabbit.png")
H = 300
W = 350
image = 0.2 * cv2.resize(image,(W,H)) + 0.8 * cv2.resize(image2,(W,H))
image = cv2.resize(image,(image.shape[1]//2,image.shape[0]//2))
k = 2.0
lam = 5*10**(-2)
S = image.copy()/image.max()
img = S.copy()
betamax = 10**5
beta = 2*lam
it = 0

alpha = 0.001
beta_1 = 0.9
beta_2 = 0.999
epsilon = 10**(-8)
iteration = 100

def Laplacian(img):
  kernel = np.array([0,1,0,1,-4,1,0,1,0]).reshape(3,3)
  h,w = img.shape[0],img.shape[1]
  L = img.copy()
  for i in range(h):
    for j in range(w):
      if (2<=i) and (2<=j):
        L[i][j] = np.sum(img[i-2:i+1,j-2:j+1]*kernel)
  return L

def gradientSGL(S,I,h,v,beta):
  delta_h = np.hstack([S[:,1:,:]-S[:,0:-1,:],S[:,0:1,:]-S[:,-1:,:]]) - h
  delta_v = np.vstack([S[1:,:,:]-S[0:-1,:,:],S[0:1,:,:]-S[-1:,:,:]]) - v

  transposed_h = np.hstack([delta_h[:,-1:,:]-delta_h[:,0:1,:],-delta_h[:,1:,:]+delta_h[:,0:-1,:]])
  transposed_v = np.vstack([delta_v[-1:,:,:]-delta_v[0:1,:,:],-delta_v[1:,:,:]+delta_v[0:-1,:,:]])

  S_1 = np.zeros(S.shape)
  I_1 = np.zeros(S.shape)
  S_1_t = np.zeros(S.shape)
  I_1_t = np.zeros(S.shape)

  for d in range(3):
    S_1[:,:,d] = cv2.Laplacian(S[:,:,d],cv2.CV_64F,ksize=3)
    I_1[:,:,d] = cv2.Laplacian(I[:,:,d],cv2.CV_64F,ksize=3)
    S_1_t[:,:,d] = cv2.Laplacian(S_1[:,:,d],cv2.CV_64F,ksize=3)
    I_1_t[:,:,d] = cv2.Laplacian(I_1[:,:,d],cv2.CV_64F,ksize=3)

  grad = 2*(S_1_t - I_1_t) + 2*beta*(transposed_h + transposed_v)
  return grad



while beta<betamax:
  it +=1
  #update h-v
  h = np.hstack([S[:,1:,:]-S[:,0:-1,:],S[:,0:1,:]-S[:,-1:,:]])
  v = np.vstack([S[1:,:,:]-S[0:-1,:,:],S[0:1,:,:]-S[-1:,:,:]])
  t = h**2 + v**2
  h[t<lam/beta] = 0
  v[t<lam/beta] = 0

  #update S
  c_1 = 0
  c_2 = 0

  for i in range(1,iteration+1):
    grad = gradientSGL(S,img,h,v,beta)
    c_1 = beta_1 * c_1 + (1-beta_1) * grad
    c_2 = beta_2 * c_2 + (1-beta_2) * (grad**2)
    C = c_1/(1-beta_1**i)
    D = c_2/(1-beta_2**i)
    S = S - alpha * C / (np.sqrt(D)+epsilon)
  beta*=k
  if it%3==0:
    image = cv2.hconcat([image,S/S.max()*255])
