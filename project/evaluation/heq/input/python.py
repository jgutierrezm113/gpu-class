import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img3.jpg',0)

hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
#plt.show()
plt.savefig('img3-hist.jpg')

# cdf_m = np.ma.masked_equal(cdf,0)
# cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
# cdf = np.ma.filled(cdf_m,0).astype('uint8')

# img2 = cdf[img]

# hist,bins = np.histogram(img2.flatten(),256,[0,256])

# cdf = hist.cumsum()
# cdf_normalized = cdf * hist.max()/ cdf.max()

# plt.plot(cdf_normalized, color = 'b')
# plt.hist(img2.flatten(),256,[0,256], color = 'r')
# plt.xlim([0,256])
# plt.legend(('cdf','histogram'), loc = 'upper left')
#plt.show()

#cv2.imshow("output", img2)
#cv2.imwrite('img0-hist.png',img2)
