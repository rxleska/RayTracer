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


def rnd_on_hemisphere():
    h0 = rd.uniform(0, 1)
    h1 = rd.uniform(0, 1)
    theta = np.arccos(h0)
    phi = 2 * np.pi * h1

    hemi = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    return hemi*rcos + r*np.dot(r, hemi)*(1-rcos) + np.cross(r, hemi)*np.sin(rtheta)

def rnd_on_hemisphere_power_weighted(a):
    h0 = rd.uniform(0, 1)
    h1 = rd.uniform(0, 1)
    theta = np.arccos(pow(h0, 1/(a+1)))
    phi = 2 * np.pi * h1

    hemi = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    return hemi*rcos + r*np.dot(r, hemi)*(1-rcos) + np.cross(r, hemi)*np.sin(rtheta)

def rnd_on_hemisphere_beckmann(a):
    h0 = rd.uniform(0, 1)
    h1 = rd.uniform(0, 1)
    theta = np.arctan(np.sqrt(-(a**2) * np.log(1-h0)))
    phi = 2 * np.pi * h1

    hemi = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    return hemi*rcos + r*np.dot(r, hemi)*(1-rcos) + np.cross(r, hemi)*np.sin(rtheta)


def rnd_on_hemisphere_blinn_phong(a):
    h0 = rd.uniform(0, 1)
    h1 = rd.uniform(0, 1)
    theta = np.arccos(pow(h0, 1/(a+2)))
    phi = 2 * np.pi * h1

    hemi = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    return hemi*rcos + r*np.dot(r, hemi)*(1-rcos) + np.cross(r, hemi)*np.sin(rtheta)


# create 4 plots 
fig = plt.figure()


# plot 1
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
for i in range(1000):
    pnts.append(rnd_on_hemisphere())
pnts = np.array(pnts)
ax1.scatter(pnts[:, 0], pnts[:, 1], pnts[:, 2])
ax1.set_title('uniform')    

# plot 2
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
pnts = []
for i in range(1000):
    pnts.append(rnd_on_hemisphere_power_weighted(2))
pnts = np.array(pnts)
ax2.scatter(pnts[:, 0], pnts[:, 1], pnts[:, 2])
ax2.set_title('power weighted')

# plot 3
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
pnts = []
for i in range(1000):
    pnts.append(rnd_on_hemisphere_beckmann(0.5))
pnts = np.array(pnts)
ax3.scatter(pnts[:, 0], pnts[:, 1], pnts[:, 2])
ax3.set_title('beckmann')

# plot 4
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
pnts = []
for i in range(1000):
    pnts.append(rnd_on_hemisphere_blinn_phong(0.5))
pnts = np.array(pnts)
ax4.scatter(pnts[:, 0], pnts[:, 1], pnts[:, 2])
ax4.set_title('blinn-phong')

plt.show()


# for i in range(1000):
#     h0 = rd.uniform(0, 1)
#     h1 = rd.uniform(0, 1)
#     # theta = np.arccos(h0) 
#     a = -0.5
#     # theta = np.arctan(np.sqrt(-a**2 * np.log(1-h0/np.pi)))
#     theta = np.cos(pow(h0, 1/(a+1)))
#     phi = 2 * np.pi * h1
#     x = np.sin(theta) * np.cos(phi)
#     y = np.sin(theta) * np.sin(phi)
#     z = np.cos(theta)

#     hemi = np.array([x, y, z])

#     hemi = hemi*np.cos(rtheta) + np.cross(r, hemi)*np.sin(rtheta) + r*np.dot(r, hemi)*(1-np.cos(rtheta))

#     pnts.append(hemi)


# # Plot the points
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# pnts = np.array(pnts)
# ax.scatter(pnts[:, 0], pnts[:, 1], pnts[:, 2])

# # draw line from origin to arbitraryVector
# ax.plot([0, arbitraryVector[0]], [0, arbitraryVector[1]], [0, arbitraryVector[2]], color='r')

# plt.show()




