from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import numpy as np
import os


def Visualiser(json_path, IMG_DIR):
    example_coco = COCO(json_path)
    categories = example_coco.loadCats(example_coco.getCatIds())
    category_names = [category['name'] for category in categories]
    print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))
    category_names = set([category['supercategory'] for category in categories])
    print('Custom COCO supercategories: ', category_names)

    category_ids = example_coco.getCatIds(catNms=['Rail_road'])
    image_ids = example_coco.getImgIds(catIds=category_ids)
    image_data = example_coco.loadImgs(image_ids[np.random.randint(0, len(image_ids))])[0]
    print("Image Id ==>", image_ids)
    print("image_data==>", image_data)

    image = io.imread(os.path.join(IMG_DIR, image_data['file_name']))
    plt.imshow(image);
    plt.axis('off')
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)
    annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
    annotations = example_coco.loadAnns(annotation_ids)
    example_coco.showAnns(annotations)

#Visualiser("trail_data/test.json",IMG_DIR = "trail_data/Images")