import cv2
import os
import numpy as np

def getdata():
    video_folder = 'vids/'
    X_data = []
    y_data = []
    list_of_videos = os.listdir(video_folder)

    for i,vidnam in enumerate(list_of_videos):
        print('processing ',i)
        #Video Path
        vid = str(video_folder + vidnam) #path to each video from list1 = os.listdir(path)
        #Reading the Video
        cap = cv2.VideoCapture(vid)
        #Reading Frames
        #fps = vcap.get(5)
        #To Store Frames
        
        offset=0
        for k in range(9):
            frames = []
            for j in range(7): #here we get 40 frames, for example
                ret, frame = cap.read()
                if ret == True:
                    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converting to gray
                    frame = cv2.resize(frame,(11,11),interpolation=cv2.INTER_AREA)
                    frames.append(frame)
                else:
                    print('Error!')
            
            X_data.append(frames) #appending each tensor of 40 frames resized for 30x30
            y_data.append(i) #appending a class label to the set of 40 frames
            
    X_data = np.array(X_data)
    y_data = np.array(y_data) #ready to split! :)
    return X_data, y_data


'''
def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        #import pdb;pdb.set_trace()
        #img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        #import pdb;pdb.set_trace()
        yield img
        
def trainGenerator(batch_size,train_path,image_folder,mask_folder='',aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        #import pdb;pdb.set_trace()
        yield (img,mask)



def train2Generator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        #import pdb;pdb.set_trace()
        #img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        #import pdb;pdb.set_trace()
        yield img
'''

