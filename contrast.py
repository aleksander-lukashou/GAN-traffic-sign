from PIL import Image
#from pathlib import Path
import glob

#pathlist = Path(directory_in_str).glob('data/circle/*.jpg')



def change_contrast(filepath, level):

    img = Image.open(filepath)
    img.load()

    factor = (259 * (level+255)) / (255 * (259-level))
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            color = img.getpixel((x, y))
            new_color = tuple(int(factor * (c-128) + 128) for c in color)
            img.putpixel((x, y), new_color)

    return img

#for path in pathlist:
#    path_in_str = str(path)
for filepath in glob.iglob('data/circle/*.jpg'):
    result = change_contrast(filepath, 100)
    result.save(filepath)

print('done')
