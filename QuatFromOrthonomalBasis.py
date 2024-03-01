import numpy as np
import sys
sys.path.append(r'../bvh_utils')
from Quaternions_old import Quaternions


def LookRotation(forward, up):
    
    
    # forward = Vector3.Normalize(forward)
    forward = forward / np.linalg.norm(forward)
    print(f'forward: {forward}')
    # print(f'forward: {forward}')
    # Vector3 right = Vector3.Normalize(Vector3.Cross(up, forward));
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    print(f'right: {right}')

    # print(f'right: {right}')
    
    # up = Vector3.Cross(forward, right);
    up = np.cross(forward, right)
    print(f'up: {up}')

    # print(f'up: {up}')

    m00 = right[0]
    m01 = right[1]
    m02 = right[2]
    m10 = up[0]
    m11 = up[1]
    m12 = up[2]
    m20 = forward[0]
    m21 = forward[1]
    m22 = forward[2]


    tr = m00 + m11 + m22
    # quaternion = Quaternions()
    if (tr > 0):
        '''
        If the trace of the matrix is less than or equal 
        to zero then identify which major diagonal element has the greatest value.
        '''
        
        S = 0.5 / np.sqrt(tr+ 1) 
        w = 0.25 / S 
        
        x = (m21 - m12) * S
        y = (m02 - m20) * S
        z = (m10 - m01) * S
        # quaternion = Quaternions(np.array([w,x,y,z]))
    
    elif ((m00 >= m11) and (m00 >= m22)):
    
        S = 2 * np.sqrt(1 + m00 - m11 - m22)
        w = (m12 - m21) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
        
        # return quaternion;
    
    elif (m11 > m22):
    
        S = 2 * np.sqrt(1 + m11 - m00 - m22)
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
        
        # return quaternion;
    else:
        S = 2 * np.sqrt(1 + m22 - m00 - m11)
        w = (m01 - m10) / S
        x = (m02 - m20) / S
        y = (m12 - m21) / S
        z = 0.25 * S
    # return quaternion;

    return Quaternions(np.array([w,x,y,z]))

if __name__ == '__main__':
    xpositive = np.array([1,0,0])
    ypositive = np.array([0,1,0])
    zpositive = np.array([0,0,1])
    # print(LookRotation(xpositive, -zpositive))
    a = Quaternions(np.array([1,0,0,0]))
    b = LookRotation(xpositive, -zpositive)
    print(a)
    print(b)
    print(b * -a)
    