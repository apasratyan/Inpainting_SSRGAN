# Inpainting_SSRGAN

A realization of an inpainting SSRGAN model from [__Image inpainting for corrupted images by using the semi-super resolution GAN__](https://arxiv.org/abs/2409.12636) alongside a few modifications.

Find datasets used [here.](https://drive.google.com/file/d/12D4DpgXUz6b7PRP61MVfay990zLuRgZh/view?usp=sharing)

Find trained model weights here. (placeholder dw)


How to use:

  -Open __Inpainting_SSRGAN.ipynb__
  
  -Specify __ssrgan_pipeline__ function args (breakdown below)
  
  -Run cell. Make sure the args are specified correctly - the training pipeline may fail or, worse, continue running with bad args.


__ssrgan_pipeline__ args breakdown:

    dataset_name: str
  
  Dataset name. Can be set to any string safely, will only impact the name of the model when it's saved.
    
    dataset_path: str
  
  Path to dataset. Must be a path to a .jpg image folder.

    save_path : str

  Directory for saving the trained model and NMSE graph.
    
    test_size: float(0, 1) = 0.5
  
  Test size of dataset. ex. 0.2 means 20% of dataset is test data and 80% is training data. Seed is fixed, same dataset with same test size will give the same breakdown.
    
    ssrgan_type: ["vanilla", "modified"] = "vanilla"
  
  Which SSRGAN is being trained: vanilla or modified.
    
    mask: ["random", "cutout"] = "random"
  
  Random mask samples random pixels to zero. Cutout mask samples random rectangular patches to zero.
    
    p: [float(0, 1), None] = 0.5
  
  Probability for each pixel to be zeroed with "random" mask. Ignored if the mask is "cutout". If None, probability will be sampled from U[0, 1] for each image during training.
    
    n_patches: int = 3
  
  Number of patches to zero with "cutout" mask. Ignored if the mask is "random".
    
    model_size: int = 64
  
  SSRGAN generator model size. Affects channel counts for both SSRGAN types, affects probability encoding sizes for "modified" ssrgan_type.
    
    loss_function: ["mse", "vgg"] = "mse"
  
  Loss function to use to compare an image with its reconstruction during training. "mse" computes MSE loss. "vgg" computes VGG16 loss.
    
    n_epochs: int = 100
  
  Number of epochs to use to train the model.
    
    optimizer: ["sgd", "adam"] = "adam"
  
  Optimizer to use to train the model. "adam" uses Adam, "sgd" uses SGD with momentum equal 0.9.
    
    initial_lr: float = 2e-4

  Initial learning rate for the optimizer.
    
    scheduler_step_size: int = 25
  
  Learning rate scheduler step size. 25 means the learning rate will change each 25 epochs.
    
    scheduler_gamma: float = 0.5
  
  Number to multiply the learning rate by every scheduler_step_size epochs. Values between 0 and 1 recommended.
    
    image_size: int = 128
  
  Input image size. Images will be resized to that size.
    
    batch_size: int = 8
  
  Batch size for training and testing dataloaders. Large numbers may or may not work faster, but will use significantly more memory.
