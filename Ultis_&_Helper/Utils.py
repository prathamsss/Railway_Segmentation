from pycococreatortools import pycococreatortools
import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
import argparse


class Utils(object):

    def __init__(self, ROOT_DIR):
        '''
        ROOT_DIR: Directory Containing Imgs and Masks
        This Class helps to convert Annotations to COCO Formate.
        So, All required operations are been covered in this class.

        NOTE: Under ROOT_DIR there should folder named "Images"
        and "Masks" containing images and masks respectively.
        '''

        self.ROOT_DIR = ROOT_DIR
        self.IMAGE_DIR = os.path.join(self.ROOT_DIR, "Images")
        self.ANNOTATION_DIR = os.path.join(self.ROOT_DIR, "Masks")

        self.INFO = {
            "description": "Example Dataset",
            "url": "https://github.com/waspinator/pycococreator",
            "version": "0.1.0",
            "year": 2018,
            "contributor": "waspinator",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        }

        self.LICENSES = [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ]

        self.CATEGORIES = [
            {
                'id': 1,
                'name': 'Rail_road',
                'supercategory': 'shape',
            }
        ]

    def check_non_img_files(self, directory_path):
        file_types = ['.jpg', '.png', '.jpeg']
        for f in os.listdir(directory_path):
            if os.path.splitext(f)[1] not in file_types:
                return f

    def filter_for_jpeg(self, root, files):
        file_types = ['*.jpeg', '*.jpg']
        file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
        files = [os.path.join(root, f) for f in files]
        files = [f for f in files if re.match(file_types, f)]

        return files

    def filter_for_annotations(self, root, files, image_filename):
        file_types = ['*.png']
        file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
        basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
        file_name_prefix = basename_no_extension + '.*'
        files = [os.path.join(root, f) for f in files]
        files = [f for f in files if re.match(file_types, f)]
        files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

        return files

    def add_class_name_to_files_names(self):

        files = os.listdir(self.ANNOTATION_DIR)
        for index, file in enumerate(files):
            if '_Rail_road' in file:
                pass

            else:
                os.rename(os.path.join(self.ANNOTATION_DIR, file),
                          os.path.join(self.ANNOTATION_DIR, file.split('.')[0] + '_Rail_road' + '.png'))

            if (os.path.isdir(os.path.join(self.ANNOTATION_DIR, file))) == True:
                os.rmdir(os.path.join(self.ANNOTATION_DIR, file))

    def export_to_COCO(self, dataset_type='train'):

        coco_output = {
            "info": self.INFO,
            "licenses": self.LICENSES,
            "categories": self.CATEGORIES,
            "images": [],
            "annotations": []
        }

        image_id = 1
        segmentation_id = 1

        # filter for jpeg images
        for root, _, files in os.walk(self.IMAGE_DIR):
            image_files = self.filter_for_jpeg(root, files)

            # go through each image
            for image_filename in image_files:
                image = Image.open(image_filename)
                image_info = pycococreatortools.create_image_info(
                    image_id, os.path.basename(image_filename), image.size)
                coco_output["images"].append(image_info)

                # filter for associated png annotations
                for root, _, files in os.walk(self.ANNOTATION_DIR):
                    annotation_files = self.filter_for_annotations(root, files, image_filename)

                    # go through each associated annotation
                    for annotation_filename in annotation_files:

                        #                         print(annotation_filename)

                        class_id = [x['id'] for x in self.CATEGORIES if x['name'] in annotation_filename][0]

                        category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                        binary_mask = np.asarray(Image.open(annotation_filename)
                                                 .convert('1')).astype(np.uint8)

                        annotation_info = pycococreatortools.create_annotation_info(
                            segmentation_id, image_id, category_info, binary_mask,
                            image.size, tolerance=2)

                        if annotation_info is not None:
                            coco_output["annotations"].append(annotation_info)

                        segmentation_id = segmentation_id + 1

                image_id = image_id + 1

        with open(os.path.join(self.ROOT_DIR, dataset_type + '.json'), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)



