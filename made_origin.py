import cv2
import numpy
import matplotlib.pyplot as plt

ori = cv2.imread("test.png")
ori = numpy.array(ori,dtype='uint8')
ori = ori[:,:,::-1]
plt.title("Origin")
plt.imshow(ori)
plt.savefig("Origin.png",dpi=150)
plt.close()