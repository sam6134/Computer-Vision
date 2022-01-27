from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# camera parameters
X_center=10.9
Y_center=10.7
Z_center=43.4
worldPoints=np.array([[X_center,Y_center,Z_center],
                       [5.5,3.9,46.8],
                       [14.2,3.9,47.0],
                       [22.8,3.9,47.4],
                       [5.5,10.6,44.2],
                       [14.2,10.6,43.8],
                       [22.8,10.6,44.8],
                       [5.5,17.3,43],
                       [14.2,17.3,42.5],
                       [22.8,17.3,44.4]], dtype=np.float32)

cx = 628
cy = 342



imagePoints=np.array([[cx,cy],
                       [502,185],
                       [700,197],
                       [894,208],
                       [491,331],
                       [695,342],
                       [896,353],
                       [478,487],
                       [691,497],
                       [900,508]], dtype=np.float32)

A = []
for i in range(6):
    A.append(imagePoints[i][0])
for i in range(6):
    A.append(imagePoints[i][1])
A = np.array(A)

D = []
for i in range(6):
    row= []
    row.extend(worldPoints[i]) # X1,Y1,Z1
    row.append(1) #1
    row.extend([0]*4) # 0,0,0,0
    row.extend(-1*A[i]*worldPoints[i]) # -x1*X1,-x1*Y1,-x1*Z1
    D.append(row)

for i in range(6,12):
    row= []
    row.extend([0]*4) # 0,0,0,0
    row.extend(worldPoints[i-6]) # X1,Y1,Z1
    row.append(1) #1
    row.extend(-1*A[i]*worldPoints[i-6]) # -y1*X1,-y1*Y1,-y1*Z1
    D.append(row)

D = np.array(D)

from numpy.linalg import inv
Q = inv(((D.transpose()).dot(D))).dot(D.transpose()).dot(A)
Q = np.array(Q)
M= np.zeros((3,4))
for i in range(11):
    M[i//4][i%4]=Q[i]
M[2][3]=1

q1 = np.array(M[0])
q2 = np.array(M[1])
q3 = np.array(M[2])

ox = q1.transpose().dot(q3)
oy = q2.transpose().dot(q3)
print(ox,oy)

fx = np.sqrt(abs(q1.transpose().dot(q1) - ox**2))
fy = np.sqrt(abs(q2.transpose().dot(q2) - oy**2))
print(fx,fy)

r11 = (ox*M[2][0] - M[0][0])/fx
r12 = (ox*M[2][1] - M[0][1])/fx
r13 = (ox*M[2][2] - M[0][2])/fx
print(r11, r12, r13)

r21 = (oy*M[2][0] - M[1][0])/fy
r22 = (oy*M[2][1] - M[1][1])/fy
r23 = (oy*M[2][2] - M[1][2])/fy
print(r21, r22, r23)

Tx = (ox*M[2][3] - M[0][3])/fx
Ty = (oy*M[2][3] - M[1][3])/fy
print(Tx, Ty)