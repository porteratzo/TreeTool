import numpy as np

def similarize(vector,target):
    Nvector = np.array(vector)
    assert len(Nvector) == 3,'vector must be dim 3'
    angle = angle_b_vectors(Nvector,target)
    if angle > np.pi/2:
        Nvector = -Nvector
    return Nvector

def angle_b_vectors(a,b):
    value = np.sum(np.multiply(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    if (value<-1) | (value>1):
        value = np.sign(value)
    angle = np.arccos(value)
    return angle

def makecylinder(model=[0,0,0,1,0,0,1],length = 1,dense=10):
    radius = model[6]
    X,Y,Z = model[:3]
    direction = model[3:6]/np.linalg.norm(model[3:6])
    n = np.arange(0,360,int(360/dense))
    height = np.arange(0,length,length/dense)
    n = np.deg2rad(n)
    x,z = np.meshgrid(n,height)
    x = x.flatten()
    z = z.flatten()
    cyl = np.vstack([np.cos(x)*radius,np.sin(x)*radius,z]).T
    rotation = rotation_matrix_from_vectors([0,0,1],model[3:6])
    rotatedcyl = np.matmul(rotation,cyl.T).T + np.array([X,Y,Z])
    return rotatedcyl   

def DistPoint2Line(point,linepoint1, linepoint2=np.array([0,0,0])): #get minimum destance from a point to a line
    return np.linalg.norm(np.cross((point-linepoint2),(point-linepoint1)))/np.linalg.norm(linepoint1 - linepoint2)


def rotation_matrix_from_vectors(vec1, vec2):
    if all(np.abs(vec1)==np.abs(vec2)):
        return np.eye(3)
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix