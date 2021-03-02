import cv2
import numpy

pic = numpy.random.rand(321, 481, 3)
pic = pic * 255
cv2.imwrite("input.bmp", pic)