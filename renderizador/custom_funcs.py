import math
import numpy as np


def fazNormal(vec):
    a = vec[0]
    b = vec[1]
    return (b,-a)

def produtoEscalar(vec0,vec1):
    return vec0[0]*vec1[0]+vec0[1]*vec1[1]

def L(p0,p1,p):
    vd = (p1[0]-p0[0],p1[1]-p0[1])
    vn = (vd[1],-vd[0])
    vp = (p[0]-p0[0],p[1]-p0[1])
    return produtoEscalar(vp,vn) >= 0

def dentro(p0,p1,p2,p):
    l0 = L(p0,p1,p)
    l1 = L(p1,p2,p)
    l2 = L(p2,p0,p)
    return l0 and l1 and l2

def diamond(px,py,x,y):
    within_pixel = x>=px and x<px+1 and y>=py and y<py+1
    dec_x = x%1
    dec_y = y%1
    print(dec_x-dec_y>=-0.5)
    return within_pixel and dec_x+dec_y<=1.5 and dec_x+dec_y>=0.5 and dec_x-dec_y<=0.5 and dec_x-dec_y>=-0.5

def cameraToScreen(p,d = 1):
    """  p = [x,y,z] in camera space"""
    x = p[0]
    y = p[1]
    z = p[2]

    u = d*x/z
    v = d*y/z

    return [u,v]


def rotate_quat(u,theta):
    """ p = point
    u = rotation axis
    theta = angle """
    u = np.array([0,0,1]) # eixo z

    qi = u[0]*np.sin(theta/2)
    qj = u[1]*np.sin(theta/2)
    qk = u[2]*np.sin(theta/2)
    qr = np.cos(theta/2)

    rm = [[1-2*(qj**2+qk**2), 2*(qi*qj - qk*qr), 2*(qi*qk + qj*qr), 0],
        [2*(qi*qj + qk*qr), 1-2*(qi**2 + qk**2), 2*(qj*qk - qi*qr), 0],
        [2*(qi*qk),2*(qj*qk+qi*qr),1-2*(qi**2 + qj**2),0],
        [0,0,0,1]]
    
    return rm

def NDCToScreenMatrix(w,h):
    return np.array([[w/2,0,0,w/2],
                     [0,-h/2,0,h/2],
                     [0,0,1,0],
                     [0,0,0,1]])

def translationMatrix(x,y,z):
    return np.array([[1,0,0,x],
                    [0,1,0,y],
                    [0,0,1,z],
                    [0,0,0,1]])



if __name__ == "__main__":
    """ vec = (1,2)
    print(fazNormal(vec))

    vec0 = (3,3)
    vec1 = (0,5)
    print(produtoEscalar(vec0,vec1))

    p0 = (0,0)
    p1 = (1,1)
    p = (2,2)

    print(L(p0,p1,p))

    p0 = (0,0)
    p1 = (0,5)
    p2 = (5,0)
    p = (0,0)

    print(dentro(p0,p1,p2,p)) """

    px = 5
    py = 5
    x = 5
    y = 5.5
    print(diamond(px,py,x,y))
