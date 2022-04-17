#Addapted from https://github.com/sagartesla/flops-cnn/blob/master/flops_calculation.py
#From sagartesla 


import os
import argparse
import sys
import importlib
import pickle
from keras import backend as K
from keras import layers

#Local imports
from import_module import import_parents

if __name__ == '__main__' and __package__ is None:
    import_parents(level=1)

from Utils.CacheManager import CacheManager

def alsm_layer_flops(output_shape,conv_filter):
    """
    input shape: (8, 8, 512) (rows,cols,channels)
    output_shape:(8, 8, 2048) (rows, cols, channels)
    filter shape:(1, 1, 512, 2048) (rows, cols, channels, filters)
    """
    kernel_shape = conv_filter[0]*conv_filter[1]*conv_filter[2]
    out = output_shape[0]*output_shape[1]
    
    flops = 2*conv_filter[3]*kernel_shape*out

    if flops / 1e9 > 1:   # for Giga Flops
        print(flops/ 1e9 ,'{}'.format('GFlops'))
    else:
        print(flops / 1e6 ,'{}'.format('MFlops'))

    return flops
    
def layer_flops(input_shape,conv_filter,stride,padding,activation):
    """
    Input:

    conv_filter = (64 ,3 ,3 ,3)  # Format: (num_filters, channels, rows, cols)
    input_shape = (64, 240 240) # Format: (channels, rows, cols)
    """
    if K.image_data_format() == 'channels_last':
        input_shape = (input_shape[2], input_shape[0], input_shape[1])
        conv_filter = (conv_filter[3],conv_filter[2],conv_filter[0],conv_filter[1])
        print("  Converted:\n - input shape {};\n - conv_filter {}".format(input_shape,conv_filter))
        
    if conv_filter[1] == 0:
        kn_shape = conv_filter[2] * conv_filter[3] # vector_length
    else:
        kn_shape = conv_filter[1] * conv_filter[2] * conv_filter[3]  # kernel shape

    flops_per_instance = kn_shape + ( kn_shape -1)    # general defination for number of flops (n: multiplications and n-1: additions)

    #Output shape rows x cols
    num_instances_per_filter = (( input_shape[1] - conv_filter[2] + 2 * padding) / stride) + 1  # for rows
    num_instances_per_filter *= ((input_shape[2] - conv_filter[3] + 2 * padding) / stride) + 1  # multiplying with cols

    flops_per_filter = num_instances_per_filter * flops_per_instance
    total_flops_per_layer = flops_per_filter * conv_filter[0]  # multiply with number of filters

    if activation == 'relu':
        # Here one can add number of flops required
        # Relu takes 1 comparison and 1 multiplication
        # Assuming for Relu: number of flops equal to length of input vector
        total_flops_per_layer += conv_filter[0] * input_shape[1] * input_shape[2]

    if total_flops_per_layer / 1e9 > 1:   # for Giga Flops
        print(total_flops_per_layer/ 1e9 ,'{}'.format('GFlops'))
    else:
        print(total_flops_per_layer / 1e6 ,'{}'.format('MFlops'))

    return total_flops_per_layer

def get_padding_needed(input_spatial_shape,filter_shape,strides):
  num_spatial_dim=len(input_spatial_shape)
  padding_needed=[0]*num_spatial_dim

  for i in range(num_spatial_dim):
    if input_spatial_shape[i] % strides[i] == 0:
      padding_needed[i] = max(filter_shape[i]-strides[i],0)
    else:
      padding_needed[i] = max(filter_shape[i]-(input_spatial_shape[i]%strides[i]),0)

  print("Padding by dimension: {}".format(padding_needed))

  return padding_needed

def calc_flops(model,deps=None):

    total_flops = 0.0
    conv_count = 0
    for l in model.layers:
        if isinstance(l,layers.Conv2D):
            input_shape = l.input_shape[1:]
            output_shape = l.output_shape[1:]
            conv_filter = l.get_weights()[0]
            conv_shape = conv_filter.shape
            if not deps is None:
                conv_shape = (conv_shape[0],conv_shape[1],conv_shape[2],deps[conv_count])
            padding = 0 #if l.padding == 'valid' else get_padding_needed(input_shape[:2],conv_shape,l.strides)[0]
            print("Layer {}:\n input shape: {}\n output_shape:{}\n filter shape:{}\n conv padding:{}\n conv stride:{}".format(l.name,
                                                                                                                                  input_shape,
                                                                                                                                  output_shape,
                                                                                                                                  conv_shape,
                                                                                                                                  padding,
                                                                                                                                  l.strides))
            #total_flops += alsm_layer_flops(output_shape,conv_shape)
            total_flops += layer_flops(input_shape,conv_shape,l.strides[0],padding,activation='')
            conv_count +=1

        if isinstance(l,layers.Dense):
            input_shape = l.input_shape
            output_shape = l.output_shape
            print("Layer {}:\n input shape: {}\n output_shape:{}".format(l.name,
                                                                             input_shape,
                                                                             output_shape))
            total_flops += 2*input_shape[1]*output_shape[1]
            
    if total_flops / 1e9 > 1:   # for Giga Flops
        print('Network consumes {:.3f} GFlops'.format(total_flops/ 1e9))
    else:
        print('Network consumes {:.3f} MFlops'.format(total_flops / 1e6))

    print("Total of convolutional layers: {}".format(conv_count))
        
if __name__ == "__main__":

    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Convolunional Neural \
        Network for Active Learning.')

    parser.add_argument('-tdim', dest='tdim', nargs='+', type=int, 
        help='Tile width and heigth, optionally inform the number of channels (Use: 200 200 for SVS 50 um).', 
        default=(240,240,3), metavar=('Width', 'Height'))

    parser.add_argument('-net',dest='network',type=str,default='',help='Network name which should be trained.\n \
    Check documentation for available models.')
    parser.add_argument('-phi', dest='phi', type=int, 
        help='Phi defines network architecture reduction. Values bigger than 1 reduce nets by 1/phi. Default = 1 (use original sizes).',default=0)
    parser.add_argument('-strategy',dest='strategy',type=str,
       help='Which strategy to use: ALTrainer, EnsembleTrainer, etc.',default='ALTrainer')
    parser.add_argument('-gpu', dest='gpu_count', type=int, 
        help='Number of GPUs available (Default: 0).', default=0)
    parser.add_argument('-cpu', dest='cpu_count', type=int, 
        help='Number of CPU cores available (Default: 1).', default=1)
    parser.add_argument('-lr', dest='learn_r', type=float, 
        help='Learning rate (Default: 0.00005).', default=0.00005)
    parser.add_argument('-tn', action='store_true', dest='new_net',
        help='Do not use older weights file.',default=False)
    parser.add_argument('-wpath', dest='weights_path',
        help='Use weights file contained in path - usefull for sequential training (Default: None).',
        default='ModelWeights')
    parser.add_argument('-model_dir', dest='model_path',
        help='Save trained models in dir (Default: TrainedModels).',
        default='TrainedModels')
    parser.add_argument('-deps', dest='deps',type=str,
        help='Use deps from file.',default=None)

    parser.add_argument('-cache', dest='cache', type=str,default='cache', 
        help='Keeps caches in this directory',required=False)
    parser.add_argument('-v', action='count', default=0, dest='verbose',
        help='Amount of verbosity (more \'v\'s means more verbose).')
    parser.add_argument('-i', action='store_true', dest='info', default=False, 
        help='Return general info about data input, the CNN, etc.')
    parser.add_argument('-logdir', dest='logdir', type=str,default='logs', 
        help='Keep logs of current execution instance in dir.')
    parser.add_argument('-show', action='store_true', dest='show', default=False, 
        help='Display and save model plot. Image will be saved to -model_dir.')    
    
    config, unparsed = parser.parse_known_args()

    cache = CacheManager(locations={})
    
    if len(config.tdim) == 2:
        config.tdim += (3,)
        
    net_module = importlib.import_module('Models',config.network)
    net_model = getattr(net_module,config.network)(config,None)

    model,_ = net_model.build(data_size=4300,allocated_gpus=config.gpu_count,preload_w=False,layer_freeze=False)

    deps = None
    if not config.deps is None:
        with open(config.deps,'rb') as fd:
            deps = pickle.load(fd)

    if config.show:
        from keras.utils.vis_utils import plot_model
        if not os.path.isdir(config.model_path):
            os.mkdir(config.model_path)
        plot_model(model,to_file=os.path.join(config.model_path,'{}_plot.png'.format(config.network)),show_shapes=True, show_layer_names=False)
        
    calc_flops(model,deps)
    
