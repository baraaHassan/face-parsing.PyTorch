import segment
import numpy as np
import cv2 as cv
import os

res_dir = './res/blur/'
dspth = './data/ilyassCloseUp/'
images = list()
for image_path in os.listdir(dspth):
    images.append(cv.imread(os.path.join(dspth, image_path)))
shape = (images[0].shape[1], images[0].shape[0])
parsings = segment.evaluate(dspth='./data/ilyassCloseUp/', cp='79999_iter.pth', return_parsings=True)

for i in range(len(parsings)):
    parsing = parsings[i].copy().astype(np.uint8)
    parsing = cv.resize(parsing, shape, interpolation=cv.INTER_NEAREST)
    # region = np.where(parsing == 0)
    blurred_image = cv.GaussianBlur(images[i], ksize=(11, 11), sigmaX=6)
    mask = parsing != 0
    blurred_image[mask] = images[i][mask]
    cv.imwrite(f'{res_dir}{i}.jpg', blurred_image)




