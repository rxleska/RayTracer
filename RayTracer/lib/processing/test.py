import matplotlib.pyplot as plt
import numpy as np
import random as rd

pnts = []

arbitraryVector = np.array([2, 1, -1])
# normalize arbitraryVector
arbitraryVector = arbitraryVector / np.linalg.norm(arbitraryVector)

r = np.cross(arbitraryVector, np.array([0, 0, 1]))
rcos = np.dot(arbitraryVector, np.array([0, 0, 1]))
rcos = rcos / np.linalg.norm(r)
rtheta = np.arccos(rcos)
rtheta = -rtheta

for i in range(1000):
    h0 = rd.uniform(0, 1)
    h1 = rd.uniform(0, 1)
    theta = np.arccos(h0) 
    phi = 2 * np.pi * h1
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    hemi = np.array([x, y, z])

    hemi = hemi*np.cos(rtheta) + np.cross(r, hemi)*np.sin(rtheta) + r*np.dot(r, hemi)*(1-np.cos(rtheta))

    pnts.append(hemi)


# Plot the points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pnts = np.array(pnts)
ax.scatter(pnts[:, 0], pnts[:, 1], pnts[:, 2])

# draw line from origin to arbitraryVector
ax.plot([0, arbitraryVector[0]], [0, arbitraryVector[1]], [0, arbitraryVector[2]], color='r')

plt.show()




