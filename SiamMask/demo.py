import numpy as np
import PIL.Image
import siammask
import cv2

sm = siammask.SiamMask()

# Weight files are automatically retrieved from GitHub Releases
sm.load_weights()

# Adjust this parameter for the better mask prediction
sm.box_offset_ratio = 1.5

path1 = r'C:\PersonalStuff\PersonalProjects\Python\SiamMask\data\MyDataset\mask.jpg'
image1 = cv2.imread(path1)
cv2.imshow('cat1', image1)

try:
    init_rect = cv2.selectROI('cat1', image1, False, False)
    x, y, w, h = init_rect
    init_rect1 = np.array([[x, y], [(x+w), (y+h)]])
except:
    print("code exit")
    exit()

img_next = np.array(PIL.Image.open('data/MyDataset/2.jpg'))[..., ::-1]

path2 = r'C:\PersonalStuff\PersonalProjects\Python\SiamMask\data\MyDataset\2.jpg'
image2 = cv2.imread(path2)
cv2.imshow('cat2', image2)

box, mask = sm.predict(image1, init_rect1, img_next, debug=True)
mask2 = cv2.merge((mask,mask,mask))

cv2.imshow('mask', mask2)

alpha = 0.5
beta = (1.0 - alpha)
dst = cv2.addWeighted(image2, alpha, mask2, beta, 0.0)
dst = np.uint8(alpha*(image2)+beta*(mask2))

cv2.imshow('blend', dst)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
