from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def denormalize(batch_tensor, mean, std):
    '''
    De-normalize the image using the mean and std passed for making it
    visually meaningfull.
    '''
    return batch_tensor * std + mean


def show_image(tensor, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
    '''
    Plot an input 4-D tensor image (OR) image_grid from torhcvision.utils.make_grid
    '''
    tensor = denormalize(tensor, mean, std)
    if tensor.size()[0] <= 5:
        nrow=tensor.size()[0]
    else:
        nrow=5
    img_grid = make_grid(tensor, nrow=nrow)
    img_grid = img_grid.permute(1, 2, 0)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(tensor)

def show_batch(batch_tensor, nmax=10, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
    batch_tensor = batch_tensor.detach()[:nmax]
    show_image(batch_tensor, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    
