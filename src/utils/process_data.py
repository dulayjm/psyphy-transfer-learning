import os
# from natsort import natsorted
import numpy as np
import matplotlib
from imageio import imread
import matplotlib.pyplot as plt
import glob

#if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)
desired_im_sz = (128, 160) #match kitti


# Create image datasets.
# Processes images and saves them in train, val, test splits.
def process_data():
    nt=10 #number of transformations per image
    numTransf = 10
    step = 1 # choose obj every 10 degrees of movement
    
    # so combine the steps of the da/scratch365/jhuang24/dataset_v1_3_partition/train_valid/known_known/00403/ztaloaders 
    root_obj = '/scratch365/jhuang24/dataset_v1_3_partition/train_valid/known_known/00403/'
    
    # but it's a fucking image folder 
    
    
#     json_data_base = '/afs/crc.nd.edu/user/j/jdulay'
#     train_known_known_with_rt_path = os.path.join(json_data_base, "train_known_known_with_rt.json")
    
#     with open(train_known_known_with_rt_path) as f:
#         data = json.load(f)
#         print("Json file loaded: %s" % json_path)
        

#     jDirs = objDirs[:8000]
    
    
    # so dump all the classes into a big pile
    
    
    stimuli = glob.glob(os.path.join(root_obj,'*.JPEG'))
    #testper = 1.0 #technically does nothing
#     stimuli=natsorted(stimuli)
    #test = objDirs[valend:]
    
#     with open(os.path.join(root, 'test.txt'), 'w') as f:
#         f.write('\n'.join(stimuli)+'\n')
        
#     print(stimuli)
    X_data = np.zeros((len(stimuli),) + (nt,) + desired_im_sz + (3,), np.uint8)
    for i, objID in enumerate(stimuli): #0-4000
        print(objID)
        for transID in range(0, numTransf, step): #starts at 0, up to not including nt
#                 print(os.path.join(root_of_objects,objID))
#                 image=imread(os.path.join(root_of_objects,objID))
            image=imread(os.path.join(root_obj,objID))
                
            print('the path is um, ', os.path.join(root_obj,objID))
            # we don't need the weird json loading stuff here, because of how
            # we set it up before ... 
            #item = data[str(transID)]
            # transID is just a num 
            #
            
            #image = imread(item["img_path"])
                
            print("checkpoint_1")
            #image = cv2.resize(image, (desired_im_sz[1], desired_im_sz[0]))
            print('pre image shape is,', image.shape)
            image = np.resize(image, (desired_im_sz[0], desired_im_sz[1], 3))
            print('pre image shape is,', image.shape)
            print("checkpoint_2")
            print(transID/step)

            print('shape is ', image.shape)
            # X_data[i, (transID/step)] = process_im(image, desired_im_sz)
            print(transID/step)
            print(i)
            print(X_data.shape)
            print('type of the image going in is', type(image))
            #print('eek', X_data[0,(transID/step)])
            #X_data[i, (transID/step)] = process_im(image, desired_im_sz[1])
            X_data[i, (transID//step)] = image
            print("checkpoint_3")
            1/0

    # from the other stuff, we want the batch nt chan h w
    X_data = np.transpose(X_data,(0,1,4,2,3)) #changing the position of numChannels
    X_data = (X_data.astype(np.float32))/255 #normalize the image

    #hkl.dump(X_data, os.path.join(root,'stimuli_test_data.hkl'))

    with h5py.File("mytestfile.hdf5", "w") as f:
        dset = f.create_dataset("mydataset", (100,), dtype='i')

# resize and crop image
def process_im(im, desired_sz):
    print('in proces')
    print('1im shape is', im.shape)
    target_ds = float(desired_sz[0])/im.shape[0]
    im = np.resize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    print('2im shape is', im.shape)
    d = (im.shape[1] - desired_sz[1]) / 2
    print('d shape is', d)
    im = im[:, d:d+desired_sz[1]]
    print('im shape is', im.shape)
    return im


if __name__ == '__main__':
	process_data()
