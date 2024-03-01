import numpy as np
import sys
sys.path.append(r'../bvh_utils')
from Quaternions_old import Quaternions


def LookRotation(forward, up):
    
    
    # forward = Vector3.Normalize(forward)
    forward = forward / np.linalg.norm(forward)
    # print(f'forward: {forward}')
    # Vector3 right = Vector3.Normalize(Vector3.Cross(up, forward));
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    # print(f'right: {right}')
    
    # up = Vector3.Cross(forward, right);
    up = np.cross(forward, right)
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


    num8 = (m00 + m11) + m22
    # quaternion = Quaternions()
    if (num8 > 0):
    
        # var num = (float)Math.Sqrt(num8 + 1f);
        num = np.sqrt(num8+ 1)
        # quaternion.w = num * 0.5f;
        w = num * 0.5
        num = 0.5 / num
        x = (m12 - m21) * num
        y = (m20 - m02) * num
        z = (m01 - m10) * num
        # quaternion = Quaternions(np.array([w,x,y,z]))
    
    elif ((m00 >= m11) and (m00 >= m22)):
    
        num7 = np.sqrt(((1 + m00) - m11) - m22)
        num4 = 0.5 / num7
        x = 0.5 * num7
        y = (m01 + m10) * num4
        z = (m02 + m20) * num4
        w = (m12 - m21) * num4
        # return quaternion;
    
    elif (m11 > m22):
    
        num6 = np.sqrt(((1 + m11) - m00) - m22)
        num3 = 0.5 / num6
        x = (m10 + m01) * num3
        y = 0.5 * num6
        z = (m21 + m12) * num3
        w = (m20 - m02) * num3
        # return quaternion;
    else:
        num5 = np.sqrt(((1 + m22) - m00) - m11)
        num2 = 0.5 / num5
        x = (m20 + m02) * num2
        y = (m21 + m12) * num2
        z = 0.5 * num5
        w = (m01 - m10) * num2
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
    