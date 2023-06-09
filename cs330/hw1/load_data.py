import numpy as np
import os
import random
import torch

def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [(i, os.path.join(path, image))
                     for i, path in zip(labels, paths)
                     for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    import imageio
    image = imageio.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image


class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(self, num_classes, num_samples_per_class, config={}, device = torch.device('cpu')):
        """
        Args:
            num_classes: int
                Number of classes for classification (N-way)
            
            num_samples_per_class: int
                Number of samples per class in the support set (K-shot).
                Will generate additional sample for the querry set.
                
            device: cuda.device: 
                Device to allocate tensors to.
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get('data_folder', './omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[: num_train]
        self.metaval_character_folders = character_folders[
            num_train:num_train + num_val]
        self.metatest_character_folders = character_folders[
            num_train + num_val:]
        self.device = device

    def sample_batch(self, batch_type, batch_size):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: str
                train/val/test set to sample from
                
            batch_size: int:
                Size of batch of tasks to sample
                
        Returns:
            images: tensor
                A tensor of images of size [B, K+1, N, 784]
                where B is batch size, K is number of samples per class, 
                N is number of classes
                
            labels: tensor
                A tensor of images of size [B, K+1, N, N] 
                where B is batch size, K is number of samples per class, 
                N is number of classes
        """
        if batch_type == "train":
            folders = self.metatrain_character_folders
        elif batch_type == "val":
            folders = self.metaval_character_folders
        else:
            folders = self.metatest_character_folders

        #############################
        #### YOUR CODE GOES HERE ####
        #############################
       
    
        N=self.num_classes
        K=self.num_samples_per_class
        B=batch_size

        images=torch.zeros(B,N,K+1,784,dtype=torch.float32)
        labels=torch.zeros(B,N,K+1,N,dtype=torch.float32)
        
        # random.seed(5533)
        for b in range(B):
            task_folders=[folders[random.randint(0, len(folders)-1)] for i in range(N)]

            task_images=torch.zeros(N,K+1,784,dtype=torch.float32)
            task_labels=torch.zeros(N,K+1,N,dtype=torch.float32)
            for class_index,class_path in enumerate(task_folders):
                imgs_paths=os.listdir(class_path)
                # 从imgs_paths 选择K+1个路径 ，这里考虑不放回
                K_imgs_paths=[os.path.join(class_path, imgs_paths[r]) for r in random.sample(list(range(len(imgs_paths))),K+1)]
                #归一化处理？？
                task_images[class_index]=torch.stack([torch.tensor(image_file_to_array(p,28*28)) for p in K_imgs_paths])
                task_labels[class_index]= torch.nn.functional.one_hot(torch.tensor(class_index), num_classes=N)

            idx=list(range(N))
            sh_idx=idx[:]
            random.shuffle(sh_idx)

            task_images[idx,K]=task_images[sh_idx,K]
            task_labels[idx,K]=task_labels[sh_idx,K]
            images[b]=task_images
            labels[b]=task_labels
        images.transpose_(1, 2)
        labels.transpose_(1, 2)
        return images.to(self.device),labels.to(self.device)
        # SOLUTION:
