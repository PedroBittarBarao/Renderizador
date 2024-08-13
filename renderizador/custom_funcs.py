import math


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
