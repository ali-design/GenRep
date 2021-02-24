from PIL import Image
import numpy as np
import IPython.display
import cv2
import io
import PIL
import sys
sys.path.append('./rotate_3d')
from image_transformer import ImageTransformer
    
def imshow(a, format='png', jpeg_fallback=True, filename=None):
  a = np.asarray(a, dtype=np.uint8)
  str_file = io.BytesIO()
  PIL.Image.fromarray(a).save(str_file, format)
  im_data = str_file.getvalue()
  try:
    disp = IPython.display.display(IPython.display.Image(im_data))
    if filename:
        size = (a.shape[1]//2, a.shape[0]//2)
        im = PIL.Image.fromarray(a)
        im.thumbnail(size,PIL.Image.ANTIALIAS)
        im.save('{}.{}'.format(filename, format))
        
  except IOError:
    if jpeg_fallback and format != 'jpeg':
      print ('Warning: image was too large to display in format "{}"; '
             'trying jpeg instead.').format(format)
      return imshow(a, format='jpeg')
    else:
      raise
  return disp

def imshow_unscaled(target, return_img=False):
  np_target = target
  np_target_scaled = np.clip(((np_target + 1) / 2.0) * 256, 0, 255)
  im = np.concatenate(np_target_scaled, axis=0)
  imshow(np.uint8(im))
  if return_img:
      return im

    
## Define our edit fns
class Rot3dTransform:    
    def __init__(self, alpha_max=60):
        self.alpha_max = alpha_max
            
    def get_alphas(self, batch_size):
        ''' get an alpha for training, return in format
        alpha_val_for_graph, alpha_val_for get_target_np'''
        alphas = np.expand_dims(np.random.choice(np.arange(-self.alpha_max, self.alpha_max+10, 10), 
                                                    batch_size), axis=1)
        # graph and target use the same slider value
        alphas_target = alphas
        alphas_graph = alphas/self.alpha_max
        return alphas_target, alphas_graph
    
    def get_target(self, outputs_zs, alpha_batch, show_img=False, show_mask=False):
        assert alpha_batch.shape[1] == 1
        assert outputs_zs.shape[0] == alpha_batch.shape[0]
        assert isinstance(outputs_zs, np.ndarray)
        if not np.any(alpha_batch):
            mask_fn = np.ones(outputs_zs.shape)
            return outputs_zs, mask_fn

        outputs_zs = outputs_zs.transpose(0, 2, 3, 1)
        mask_fn = np.ones(outputs_zs.shape)

        target_fn = np.zeros(outputs_zs.shape)
        mask_out = np.zeros(outputs_zs.shape)
        for i in range(outputs_zs.shape[0]):
            alpha = alpha_batch[i][0]
            it = ImageTransformer(outputs_zs[i,:,:,:], shape=None)
            target_fn[i,:,:,:] = it.rotate_along_axis(phi = alpha, dx = -alpha/5)
            it = ImageTransformer(mask_fn[i,:,:,:], shape=None)
            mask_out[i,:,:,:] = it.rotate_along_axis(phi = alpha, dx = -alpha/5)

        if show_img:
            print('Target image:')
            imshow_unscaled(target_fn)
        if show_mask:
            print('Target mask:')
            imshow_unscaled(mask_out)

        target_fn = target_fn.transpose(0, 3, 1, 2)
        mask_out = mask_out.transpose(0, 3, 1, 2)
        return target_fn, mask_out

    
class Rot2dTransform:    
    def __init__(self, alpha_max=30):
        self.alpha_max = alpha_max
            
    def get_alphas(self, batch_size):
        ''' get an alpha for training, return in format
        alpha_val_for_graph, alpha_val_for get_target_np'''
        alphas = np.expand_dims(np.random.choice(np.arange(-self.alpha_max, self.alpha_max+5, 5), 
                                                    batch_size), axis=1)
        # graph and target use the same slider value
        alphas_target = alphas
        alphas_graph = alphas/self.alpha_max
        return alphas_target, alphas_graph

    def get_target(self, outputs_zs, mask_fn, alpha_batch, show_img=False, show_mask=False):  
        assert alpha_batch.shape[1] == 1
        assert outputs_zs.shape[0] == mask_fn.shape[0] == alpha_batch.shape[0]
        assert isinstance(outputs_zs, np.ndarray)
        if not np.any(alpha_batch):
            return outputs_zs, mask_fn

        outputs_zs = outputs_zs.transpose(0, 2, 3, 1)
        mask_fn = np.ones(outputs_zs.shape)

        target_fn = np.zeros(outputs_zs.shape)
        mask_out = np.zeros(outputs_zs.shape)
        img_size = target_fn.shape[1]
        for i in range(outputs_zs.shape[0]):
            degree = alpha_batch[i][0] 
            M = cv2.getRotationMatrix2D((img_size//2, img_size//2), degree, 1)
            target_fn[i,:,:,:] = cv2.warpAffine(outputs_zs[i,:,:,:], M, (img_size, img_size))
            mask_out[i,:,:,:] = cv2.warpAffine(mask_fn[i,:,:,:], M, (img_size, img_size))

        mask_out[np.nonzero(mask_out)] = 1.
        assert(np.setdiff1d(mask_out, [0., 1.]).size == 0)

        if show_img:
            print('Target image:')
            imshow_unscaled(target_fn)
        if show_mask:
            print('Target mask:')
            imshow_unscaled(mask_out)
            
        target_fn = target_fn.transpose(0, 3, 1, 2)
        mask_out = mask_out.transpose(0, 3, 1, 2)
        return target_fn, mask_out

    
class ZoomShiftXYTransform:    
    def __init__(self, alpha_max_zoom=4.0, alpha_max_shift=100):
        self.alpha_max_zoom = alpha_max_zoom
        self.alpha_max_shift = alpha_max_shift

    def get_alphas(self, batch_size):
        ''' get an alpha for training, return in format
        alpha_val_for_graph, alpha_val_for get_target_np'''
        
        alphas_shiftx_target = np.expand_dims(np.random.choice(np.arange(-self.alpha_max_shift, self.alpha_max_shift+10, 10), 
                                                           batch_size), axis=1)
        alphas_shiftx_graph = alphas_shiftx_target / self.alpha_max_shift
        
        alphas_shifty_target = np.expand_dims(np.random.choice(np.arange(-self.alpha_max_shift, self.alpha_max_shift+10, 10), 
                                                           batch_size), axis=1)
        alphas_shifty_graph = alphas_shifty_target / self.alpha_max_shift
        
        
        alphas_zoom_target = np.ones((batch_size, 1))
        alphas_zoom_graph = np.ones((batch_size, 1))
        for b in range(batch_size):            
            coin = np.random.uniform(0, 1)
            if coin <= 0.5:
                alpha_val = np.random.uniform(0.25, 1.)
            else:
                alpha_val = np.random.uniform(1., 4.)
            alphas_zoom_target[b,0] = alpha_val
            alphas_zoom_graph[b,0] = np.log(alpha_val)
        
        return [alphas_zoom_target, alphas_zoom_graph, alphas_shiftx_target, alphas_shiftx_graph, alphas_shifty_target, alphas_shifty_graph]
            
    def get_target(self, outputs_zs, mask_fn, alpha_batch_zoom, alpha_batch_shift_x, alpha_batch_shift_y, 
                   show_img=False, show_mask=False):
        assert alpha_batch_zoom.shape[1] == 1
        assert outputs_zs.shape[0] == mask_fn.shape[0] == alpha_batch_zoom.shape[0]
        assert(alpha_batch_zoom.shape == alpha_batch_shift_x.shape == alpha_batch_shift_y.shape)
        assert isinstance(outputs_zs, np.ndarray)

        outputs_zs = outputs_zs.transpose(0, 2, 3, 1)
        mask_fn = np.ones(outputs_zs.shape)

        target_fn = np.zeros(outputs_zs.shape)
        mask_out = np.zeros(outputs_zs.shape)
        img_size = target_fn.shape[1]

        for i in range(outputs_zs.shape[0]):
            alpha_zoom = alpha_batch_zoom[i][0]
            new_size = int(alpha_zoom*img_size)
            output_cropped = np.zeros((new_size, new_size, outputs_zs.shape[3]))
            mask_cropped = np.zeros((new_size, new_size, outputs_zs.shape[3]))
            ## crop
            if alpha_zoom < 1:
                output_cropped = outputs_zs[i,img_size//2-new_size//2:img_size//2+new_size//2, img_size//2-new_size//2:img_size//2+new_size//2,:]
                mask_cropped = mask_fn[i,img_size//2-new_size//2:img_size//2+new_size//2, img_size//2-new_size//2:img_size//2+new_size//2,:]
            ## padding
            else:
                output_cropped[new_size//2-img_size//2:new_size//2+img_size//2, new_size//2-img_size//2:new_size//2+img_size//2,:] = outputs_zs[i] 
                mask_cropped[new_size//2-img_size//2:new_size//2+img_size//2, new_size//2-img_size//2:new_size//2+img_size//2,:] = mask_fn[i]

            ## Resize
            target_fn[i,:,:,:] = cv2.resize(output_cropped, (img_size, img_size), interpolation = cv2.INTER_LINEAR)
            mask_out[i,:,:,:] = cv2.resize(mask_cropped, (img_size, img_size), interpolation = cv2.INTER_LINEAR)

            # shift x and y:
            M = np.float32([[1,0,alpha_batch_shift_x[i][0]],[0,1,alpha_batch_shift_y[i][0]]])
            target_fn[i,:,:,:] = cv2.warpAffine(target_fn[i,:,:,:], M, (img_size, img_size))
            mask_out[i,:,:,:] = cv2.warpAffine(mask_out[i,:,:,:], M, (img_size, img_size))

        mask_out[np.nonzero(mask_out)] = 1.
        assert(np.setdiff1d(mask_out, [0., 1.]).size == 0)

        if show_img:
            print('Target image:')
            imshow_unscaled(target_fn)
        if show_mask:
            print('Target mask:')
            imshow_unscaled(mask_out)
            
        target_fn = target_fn.transpose(0, 3, 1, 2)
        mask_out = mask_out.transpose(0, 3, 1, 2)
        return target_fn, mask_out
    

class ColorTransform:
    def __init__(self, alpha_max=0.5, num_channels=3):
        self.alpha_max = alpha_max
        self.num_channels = num_channels
        
    def get_alphas(self, batch_size):
        ''' get an alpha for training, return in format
        alpha_val_for_graph, alpha_val_for get_target_np'''
        alphas = np.random.random(size=(batch_size, self.num_channels))-self.alpha_max
        # graph and target use the same slider value
        alphas_target = alphas
        alphas_graph = alphas
        return alphas_target, alphas_graph

    def get_target(self, outputs_zs, alpha_batch, show_img=False, show_mask=False):
        assert alpha_batch.shape[1] == 3
        assert(outputs_zs.shape[0] == alpha_batch.shape[0])
        assert isinstance(outputs_zs, np.ndarray)
        if not np.any(alpha_batch): # alpha is all zeros
            return outputs_zs, np.ones(outputs_zs.shape)

        outputs_zs = outputs_zs.transpose(0, 2, 3, 1)

        target_fn = np.copy(outputs_zs)
        for b in range(outputs_zs.shape[0]):
            for i in range(self.num_channels):
                target_fn[b,:,:,i] = target_fn[b,:,:,i]+alpha_batch[b,i]

        mask_out = np.ones(outputs_zs.shape)

        if show_img:
            print('Target image:')
            imshow_unscaled(target_fn)
        if show_mask:
            print('Target mask:')
            imshow_unscaled(mask_out)
            
        target_fn = target_fn.transpose(0, 3, 1, 2)
        mask_out = mask_out.transpose(0, 3, 1, 2)
        return target_fn, mask_out
        