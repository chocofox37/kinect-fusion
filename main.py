import numpy as np
import cv2

print("Hello, world!")

identity = np.identity(3)
print(identity)

lena = cv2.imread("lena.png")
cv2.imshow("Lena", lena)
cv2.waitKey(0)
