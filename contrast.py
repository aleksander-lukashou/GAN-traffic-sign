from PIL import Image
import glob
import numpy as np
from scipy import misc, ndimage



def change_contrast(filepath, level):

    img = Image.open(filepath)
    img.load()



    #img = misc.img(gray=True).astype(float)
    blurred_f = ndimage.gaussian_filter(img, 3)	
    filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
    alpha = 30
    sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)

    im = ndimage.distance_transform_bf(sharpened)
    im_noise = im + 0.2 * np.random.randn(*im.shape)
    im_med = ndimage.median_filter(im_noise, 3)

    factor = (259 * (level+255)) / (255 * (259-level))
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            color = img.getpixel((x, y))
            new_color = tuple(int(factor * (c-128) + 128) for c in color)
            img.putpixel((x, y), new_color)

    return img


for filepath in glob.iglob('data/circle/*.jpg'):
    result = change_contrast(filepath, 100)
    result.save(filepath)

print('done')
