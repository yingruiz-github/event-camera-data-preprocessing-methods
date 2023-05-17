'''
make sure there are no other folders or files in the dir_to_read folder other than the data folders
make sure 
'''

import argparse
import numpy as np
import pandas as pd
import os
from utils import check_pixel_correctness, seperate_pos_neg
import shutil
from PIL import Image

# take arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dir_to_save", type=str, required=True, help="directory to save processed images")
parser.add_argument("--dir_to_read", type=str, required=True, help="directory to read the data")
args = parser.parse_args()

# constant 
PIXEL_SET = {(30, 30, 220), (200, 30, 30), (255, 255, 255)} # (30, 30, 220) -> neg, (200, 30, 30) -> pos, (255, 255, 255) -> empty

def main():
	# create folder structure to store, the folder structure is show below:
	'''
	seperated_imgs
	|
	|--<original folder name>
	| |--neg (all speperated negative imgs)
	| |--pos (all seperated postive imgs)
	'''
	# make top level seperated_imgs dir
	new_folder = os.path.join(args.dir_to_save, "seperated_imgs")
	if os.path.exists(new_folder):
		shutil.rmtree(new_folder)
	os.makedirs(new_folder)

	# make rest dirs within top level and process the imgs
	for original_folder_name in os.listdir(args.dir_to_read):
		original_folder_name_abs = os.path.join(new_folder, original_folder_name)
		current_abs_neg = os.path.join(original_folder_name_abs, "neg")
		current_abs_pos = os.path.join(original_folder_name_abs, "pos")
		os.makedirs(original_folder_name_abs)
		os.makrdirs(current_abs_neg)
		os.makrdirs(current_abs_pos)

		# process and save imgs 
		folder_abs = os.path.join(args.dir_to_read, original_folder_name) # get folder abs path
		event_imgs_abs = os.path.join(folder_abs, "event_imgs")
		for event_image_name in os.listdir(event_imgs_abs):
			event_image = cv.imread(event_image_name)
			check_pixel_correctness(event_image, PIXEL_SET) # all event images should only has 3 pre-defined pixel values, if not, print it to cmd
			pos_img, neg_img = seperate_pos_neg(event_image, neg = (30, 30, 220), pos = (200, 30, 30))
			# save imgs
			pos_img = Image.fromarray(pos_img)
			neg_img = Image.fromarray(neg_img)
			neg_img.save(os.path.join(current_abs_neg, "{}_neg.bmp".format(event_image_name)))
			pos_img.save(os.path.join(current_abs_pos, "{}_pos.bmp".format(event_image_name)))


if __name__ == "__main__":
	main()


'''
# process the original image
folders = os.listdir(args.dir_to_read)
for folder in folders:
	folder_abs = os.path.join(args.dir_to_read, folder) # get folder abs path
	event_imgs_abs = os.path.join(folder_abs, "event_imgs")
	for event_image_name in os.listdir(event_imgs_abs):
		event_image = cv.imread(event_image_name)
		check_pixel_correctness(event_image, PIXEL_SET) # all event images should only has 3 pre-defined pixel values, if not, print it to cmd
		pos_img, neg_img = seperate_pos_neg(event_image, neg = (30, 30, 220), pos = (200, 30, 30))
		# save imgs
		pos_img = Image.fromarray(temp_pos_img)
		im_pos_img.save(os.path.join(pos_directory, "pos_img_{}_{}.png".format(i, j)))
		'''