from PIL import Image, ImageEnhance
import glob
import numpy as np
from scipy import misc, ndimage
import cv2


def change_contrast(filepath, level):

    img = Image.open(filepath)
    img.load()
    #img = misc.img(gray=True).astype(float)

    factor = (259 * (level+255)) / (255 * (259-level))
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            color = img.getpixel((x, y))
            new_color = tuple(int(factor * (c-128) + 128) for c in color)
            img.putpixel((x, y), new_color)

    
    blurred_f = ndimage.gaussian_filter(img, 3)	
    filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
    alpha = 20
    sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
 
    im = ndimage.distance_transform_bf(sharpened)
    im_noise = im + 0.2 * np.random.randn(*im.shape)
    im_med = ndimage.median_filter(im_noise, 3)

    return im_med

def clahe (image_path):

	bgr = cv2.imread(image_path)
	lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
	lab_planes = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
	lab_planes[0] = clahe.apply(lab_planes[0])
	lab = cv2.merge(lab_planes)
	bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
	return bgr


for filepath in glob.iglob('data/blueCircle/*.jpg'):

    result = change_contrast(filepath, 150)
    #result.save(filepath)

    result = clahe(filepath)
    cv2.imwrite(filepath, result)


print('done')
