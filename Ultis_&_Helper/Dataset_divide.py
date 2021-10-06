import shutil
import os
import argparse


class Divide_dataset(object):
    def __init__(self, main_dir, destination, thress):
        '''
        Divide_dataset is for dividing dataset into Train, Test, Validation
        main_dir : Directory containing all files Eg: /Images OR /Masks
        destination: Directory path to move files.
        '''
        count = 0
        get_files = os.listdir(main_dir)
        get_files.sort()
        for i in get_files:

            #             shutil.move(os.path.join(main_dir,i),destination)

            if count == thress:
                break
            count = count + 1

    def check(self, path_to_set):
        '''
        Check whether divided set contains required no of imgs and masks accordingly.
        path_to_set : Path of directory containing Images and Masks folder.
        '''
        print("No of Images = >", len(os.listdir(path_to_set + '/Images')))
        print("No of Masks = >", len(os.listdir(path_to_set + '/Masks')))

        masks_dir = os.path.join(path_to_set, 'Masks')
        imgs_dir = os.path.join(path_to_set + '/Images')

        mask = [i.split('.')[0] for i in os.listdir(masks_dir)]
        mask = [j.split('_')[0] for j in mask]

        imgs = [i.split('.')[0] for i in os.listdir(imgs_dir)]

        if not list(set(imgs).difference(mask)):
            print("All files are successfully Moved!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Path cotaining Images files to Divide")
    parser.add_argument("--destination", help="Directory path to move files")
    parser.add_argument("--thress", help="No of files to move")

    args = parser.parse_args()
    d = Divide_dataset(args.dir, args.destination, args.thress)
    d.check(args.destination)