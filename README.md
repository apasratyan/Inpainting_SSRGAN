# Inpainting_SSRGAN

A realization of an inpainting SSRGAN model from [__Image inpainting for corrupted images by using the semi-super resolution GAN__](https://arxiv.org/abs/2409.12636) alongside a few modifications.

Find datasets used [here.](https://drive.google.com/file/d/12D4DpgXUz6b7PRP61MVfay990zLuRgZh/view?usp=sharing)

Find trained model weights here. (placeholder dw)


How to use:

  -Open __Inpainting_SSRGAN.ipynb__
  
  -Specify __ssrgan_pipeline__ function args (breakdown below)
  
  -Run cell. Make sure the args are specified correctly - the training pipeline may fail or, worse, continue running with bad args.


__ssrgan_pipeline__ args breakdown:

  __dataset_name: str__
  
    Dataset name. Can be set to any string safely, will only impact the name of the model when it's saved.
    
  __dataset_path: str__
  
    Path to dataset. Must be a path to a .jpg image folder.
    
  __test_size: float(0, 1) = 0.5__
  
    Test size of dataset. ex. 0.2 means 20% of dataset is test data and 80% is training data. Seed is fixed, same dataset with same test size will give the same breakdown.
    
  __ssrgan_type: ["vanilla", "modified"] = "vanilla"__
  
    Which SSRGAN is being trained: vanilla or modified.
    
  __mask: ["random", "cutout"] = "random"__
  
    Random mask samples random pixels to zero. Cutout mask samples random rectangular patches to zero.
    
  __p: [float(0, 1), None] = 0.5__
  
    Probability for each pixel to be zeroed with "random" mask. Ignored if the mask is "cutout". If None, probability will be sampled from U[0, 1] for each image during training.
    
  __n_patches: int = 3__
  
    Number of patches to zero with "cutout" mask. Ignored if the mask is "random".
    
  __model_size: int = 64__
  
    SSRGAN generator model size. Affects channel counts for both SSRGAN types, affects probability encoding sizes for "modified" ssrgan_type.
    
  __loss_function: ["mse", "vgg"] = "mse"__
  
    Loss function to use to compare an image with its reconstruction during training. "mse" computes MSE loss. "vgg" computes VGG16 loss.
    
  __n_epochs: int = 100__
  
    Number of epochs to use to train the model.
    
  __optimizer: ["sgd", "adam"] = "adam"__
  
    Optimizer to use to train the model. "adam" uses Adam, "sgd" uses SGD with momentum equal 0.9.
    
  __initial_lr: float = 2e-4__
  
    Initial learning rate for the optimizer.
    
  __scheduler_step_size: int = 25__
  
    Learning rate scheduler step size. 25 means the learning rate will change each 25 epochs
    .
  __scheduler_gamma: float = 0.5__
  
    Number to multiply the learning rate by every scheduler_step_size epochs. Values between 0 and 1 recommended.
    
  __image_size: int = 128__
  
    Input image size. Images will be resized to that size.
    
  __batch_size: int = 8__
  
    Batch size for training and testing dataloaders. Large numbers may or may not work faster, but will use significantly more memory.
