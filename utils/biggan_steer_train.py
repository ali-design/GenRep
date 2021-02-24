import torch
from pytorch_pretrained_biggan import (
    BigGAN, 
    one_hot_from_int, 
    truncated_noise_sample)
import time
import torchvision.utils as vutils
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import numpy as np
import IPython.display
import cv2
import pixel_transformations as pixt
G = BigGAN.from_pretrained('biggan-deep-256')
if torch.cuda.device_count() > 1:
    print('Using {} gpus for G'.format(torch.cuda.device_count()))
    G = torch.nn.DataParallel(G)
G.to('cuda')
        
# training parameters
dim_z = truncated_noise_sample(truncation=0.1, batch_size=1).shape[1]
learning_rate = 0.001 # you can also try changing this as well

w_rot3d = torch.tensor(np.random.normal(0.0, 0.1, [1, dim_z]),  
                 device='cuda', dtype=torch.float32, requires_grad=True)
w_rot2d = torch.tensor(np.random.normal(0.0, 0.1, [1, dim_z]),  
                 device='cuda', dtype=torch.float32, requires_grad=True)
w_zoom = torch.tensor(np.random.normal(0.0, 0.1, [1, dim_z]),  
                 device='cuda', dtype=torch.float32, requires_grad=True)
w_shiftx = torch.tensor(np.random.normal(0.0, 0.1, [1, dim_z]),  
                 device='cuda', dtype=torch.float32, requires_grad=True)
w_shifty = torch.tensor(np.random.normal(0.0, 0.1, [1, dim_z]),  
                 device='cuda', dtype=torch.float32, requires_grad=True)
w_color = torch.tensor(np.random.normal(0.0, 0.1, [1, dim_z, 3]),  
                 device='cuda', dtype=torch.float32, requires_grad=True)
# loss function and optimizer
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam([w_rot3d, w_rot2d, w_zoom, w_shiftx, w_shifty, w_color], lr=learning_rate)


# training
num_samples = 160000
truncation = 1.0
loss_sum = 0
optim_iter = 0
batch_size = 32
loss_values = []
save_freq = 1000 # frequency to save weights

# train loop
for batch_num in range(num_samples // batch_size):
    start_time = time.time()
  
    # 1. sample z and y randomly
    # latents
    zs = truncated_noise_sample(truncation=truncation, batch_size=batch_size, seed=None)
    zs = torch.from_numpy(zs).to('cuda')
    # labels
    ys = one_hot_from_int(np.random.choice(1000, batch_size), batch_size)
    ys = torch.from_numpy(ys).to('cuda')
    # transforms
    # 3D
    rot3d_transform = pixt.Rot3dTransform()
    alphas_rot3d_target, alphas_rot3d_graph = rot3d_transform.get_alphas(batch_size)
    alphas_rot3d_graph = torch.tensor(alphas_rot3d_graph, device='cuda', dtype=torch.float32)
    # 2D
    rot2d_transform = pixt.Rot2dTransform()
    alphas_rot2d_target, alphas_rot2d_graph = rot2d_transform.get_alphas(batch_size)
    alphas_rot2d_graph = torch.tensor(alphas_rot2d_graph, device='cuda', dtype=torch.float32)
    # Zoom, shiftx, shifty
    zxy_transform = pixt.ZoomShiftXYTransform()
    alphas_zxy = zxy_transform.get_alphas(batch_size)
    alphas_zoom_graph = torch.tensor(alphas_zxy[1], device='cuda', dtype=torch.float32)
    alphas_shiftx_graph = torch.tensor(alphas_zxy[3], device='cuda', dtype=torch.float32)
    alphas_shifty_graph = torch.tensor(alphas_zxy[5], device='cuda', dtype=torch.float32)
    # Color
    color_transform = pixt.ColorTransform()
    alphas_color_target, alphas_color_graph = color_transform.get_alphas(batch_size)
    alphas_color_graph = torch.tensor(alphas_color_graph, device='cuda', dtype=torch.float32)

    with torch.no_grad():
        out_im = G(zs, ys, truncation)

    # get composed target
    color_target, color_mask = color_transform.get_target(out_im.cpu().numpy(), alphas_color_target)
    rot3d_target, rot3d_mask = rot3d_transform.get_target(color_target, alphas_rot3d_target)
    rot2d_target, rot2d_mask = rot2d_transform.get_target(rot3d_target, rot3d_mask, alphas_rot2d_target)
    targets, masks = zxy_transform.get_target(rot2d_target, rot2d_mask, alphas_zxy[0], alphas_zxy[2], alphas_zxy[4])
    numel_masks = np.count_nonzero(masks)

    targets_tensor = torch.tensor(targets, device='cuda', dtype=torch.float32)
    masks_tensor = torch.tensor(masks, device='cuda', dtype=torch.float32)
    numel_masks_tensor = torch.tensor(numel_masks, device='cuda', dtype=torch.float32)
  
    # forward pass
    optimizer.zero_grad()
    z_new = zs + alphas_rot3d_graph * w_rot3d + alphas_rot2d_graph * w_rot2d + \
            alphas_zoom_graph * w_zoom + alphas_shiftx_graph * w_shiftx + alphas_shifty_graph * w_shifty
    for i in range(color_transform.num_channels):
        z_new = z_new + alphas_color_graph[:,i].unsqueeze(1) * w_color[:,:,i]
        
    out_im = G(z_new, ys, truncation)
    # mean over unmasked regions in the target
    loss = criterion(out_im * masks_tensor, targets_tensor * masks_tensor) / numel_masks_tensor
  
    # 4. optimize loss 
    loss.backward()
    optimizer.step()
  
    loss_values.append(loss.detach().cpu().numpy())
    loss_sum += loss.detach().cpu().numpy()
    elapsed_time = time.time() - start_time
  
    print('Time:{}, batch_start:{}, loss:{}'.format(elapsed_time, batch_num * batch_size, loss))
 
    # save intermediate walk
    if (optim_iter % save_freq == 0) and (optim_iter > 0):
#         print('Saving intermediate ckpt at optim_iter x batch_size {}'.format(optim_iter * batch_size))
        torch.save({'walk_color': w_color, 'walk_rot3d': w_rot3d, 'walk_rot2d': w_rot2d, 
                    'walk_zoom': w_zoom, 'walk_shiftx': w_shiftx, 'walk_shifty': w_shifty}, 
                   'walk_weights_biggan_deep/w_composed_{}.pth'.format(optim_iter * batch_size))
  
    optim_iter += 1
  
print('average loss with this metric: ', loss_sum/(optim_iter*batch_size))
torch.save({'walk_color': w_color, 'walk_rot3d': w_rot3d, 'walk_rot2d': w_rot2d, 
            'walk_zoom': w_zoom, 'walk_shiftx': w_shiftx, 'walk_shifty': w_shifty}, 
           'walk_weights_biggan_deep/w_composed_final.pth')

np.save('walk_weights_biggan_deep/loss_values.npy',loss_values)