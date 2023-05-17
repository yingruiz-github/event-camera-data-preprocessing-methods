# import function
import numpy as np
import pandas as pd
from PIL import Image
import os


# functions

def extract_data(filename, sep = " "):
    """
    filename: path to text file to extract
    sep: characters used to seperate field

    extract timestamp, x location, y location, polarity from target text file. 
    Assume field order: timestamp, x, y, polarity

    return: extract info as array of lists 
    """
    infile = open(filename, 'r')
    ts, x, y, p = [], [], [], []
    for line in infile:
        words = line.split(sep)
        ts.append(float(words[0]))
        x.append(int(words[1]))
        y.append(int(words[2]))
        p.append(int(words[3]))
    infile.close()
    return pd.DataFrame({"timestamps":ts, "x":x, "y":y, "polarity":p})


def n_bins_two_frame_no_agg(timestamps, n_bins, df, resolution, pos_directory, neg_directory):
    """
    this transformation implementation is used in paper "Spiking Transformers for Event-based Single Object Tracking"
    
    some notes: 
        1. each bin interval is [start time, end time)
        2. assumption is no two consecutive timestamps are same -> if not will throw error
        3. each interval is defined by the time between two ground truth timestamps. This interval is further divide to n_bins numbers
           sub-intervals. The aggregation is happen in each such sub-interval
        4. The polarity is not aggregated. If a pixel shoot a pos/neg signal during anytime in a sub-interval. that pixel will be marked with
           pos value (1) or neg value （0）


    *input*
    timestamps: an array of timestamps (in float form)
    n_bins: number of bins to aggeragate to between two consuccsive timestamps
    df: a pd df with format {"timestamps": [float], "x"[float], "y":[float], polarity:[0 or 1]}
    resolution: The output image's resolution. resolution < any x,y in "data", an error will throw
    pos_directory: directory to save positive images
    neg_directory: directory to save negative images 
    """
    pos_images = [] # all processed postive imgs
    neg_images = [] # all processed negatie imgs
    pos_img_names = [] # saved postive img name
    neg_img_names = [] # saved negative img name
    bin_starts = [] # each bin start time
    bin_ends = [] # each bin end time 

    # loop through interval between each pair of timestamps
    for i in range(len(timestamps) - 1):
        bin_length = (timestamps[i + 1] - timestamps[i])/n_bins
        
        # aggegate the polarity in each image. 1 means a signal, 0 means no signal
        for j in range(n_bins):
            # create an empty image
            temp_pos_img = np.zeros(resolution)
            temp_neg_img = np.zeros(resolution)

            # create new
            bin_start_time = timestamps[i] + j * bin_length
            bin_end_time = timestamps[i] + (j+1) * bin_length

            # aggegation on specific bin
            temp_df = df[(df.timestamps >= bin_start_time) & (df.timestamps < bin_end_time)] 
            for k in range(len(temp_df)):
                row = temp_df.iloc[k]
                if row["polarity"]:
                    temp_pos_img[int(row["y"])][int(row["x"])] = 255 
                else:
                    temp_neg_img[int(row["y"])][int(row["x"])] = 255
            
            # append processed imgs
            pos_images.append(temp_pos_img)
            neg_images.append(temp_neg_img)

            # save the image to specific dir
            im_pos_img = Image.fromarray(temp_pos_img)
            im_pos_img = im_pos_img.convert("L") # convert to gray scale
            im_neg_img = Image.fromarray(temp_neg_img)
            im_neg_img = im_neg_img.convert("L") # convert to gray scale
            im_pos_img.save(os.path.join(pos_directory, "pos_img_{}_{}.png".format(i, j)))
            im_neg_img.save(os.path.join(neg_directory, "neg_img_{}_{}.png".format(i, j)))

            # record image name, bin start time, bin end time
            pos_img_names.append("pos_img_{}_{}.png".format(i, j))
            neg_img_names.append("neg_img_{}_{}.png".format(i, j))
            bin_starts.append(bin_start_time)
            bin_ends.append(bin_end_time)
    
    # save processed img name, img aggregation start time, img aggregation end time as csv in same folder as imgs 
    pd.DataFrame({"pos_img_name":pos_img_names, "bin_start":bin_starts, "bin_ends":bin_ends}).to_csv(os.path.join(pos_directory, "meta_pos.csv"))
    pd.DataFrame({"neg_img_name":neg_img_names, "bin_start":bin_starts, "bin_ends":bin_ends}).to_csv(os.path.join(neg_directory, "meta_neg.csv"))

    return pos_images, neg_images, bin_starts, bin_ends


# this section is used for seperate neg and pos event points in visevent event images. This section is only for visevent event images

def check_pixel_correctness(image, checkSet, image_id):
    '''
    *description*
    given a 3D np array. check if all its pixel value are in a pre-defined set. If find un-allowed pixel, print it to cmd.

    *input*
    image: 3D array. Represent an image
    checkSet: a set of tuples. store the allowed pixels

    *return*
    0 or 1. 0 -> normal, 1 -> not normal
    '''
    for row in image:
        for pixel in row:
            if not (tuple(pixel) in checkSet):
                print("find pixel value {} in image {}".format(pixel, image_id))
                return 1
    return 1

def seperate_pos_neg(image, neg, pos):
    '''
    *description*
    given a image input (np 3D arrau). seperate it to two seperate images based on polarity. 

    *input*
    image: 3D array, represent an visEvent event image
    neg: negative pixel value
    pos: postive pixel value

    *return*
    (pos_image, neg_image)
    '''
    resolution = (image.shape[0], image.shape[1]) 
    neg_image = np.zeros(image.shape, dtype=np.uint8 ) # init empty neg image
    pos_image = np.zeros(image.shape, dtype=np.uint8 ) # init empty pos image
    
    # seperate the image
    for i in range(resolution[0]):
        for j in range(resolution[1]):
            if tuple(image[i][j]) == neg:
                neg_image[i][j] = neg_image[i][j]+ 255
            elif tuple(image[i][j]) == pos:
                pos_image[i][j] = neg_image[i][j]+ 255

    return (pos_image, neg_image)


                
            
        





