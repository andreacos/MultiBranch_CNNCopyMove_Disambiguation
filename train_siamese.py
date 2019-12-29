'''
Copyright is preserved to Quoc-Tin Phan (dimmoon2511[at]gmail.com)
'''

from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import siamese_single_net
import single_net_solver
from queue_runner import CustomRunner
from data import Data_Train_Val_Single


if __name__ == '__main__':
    tf.reset_default_graph()
    # TO CONFIGURE: pos_lst.txt is the file whose each line is the absolute path to a training image
    data = Data_Train_Val_Single({'pos': 'pos_lst.txt'}, max_items=880000)
    
    data_queue = CustomRunner(data.data_fn, n_processes=3, max_size=2000, n_threads=10)
    net = siamese_single_net.initialize({
        'use_tf_threading'  :   True, 
        'batch_size'        :   128, 
        'im_size'           :   64, 
        'use_gpu'           :   [0,1,2,3], # the GPU's ID (here we use four GPUs)
        'is_training'       :   True,
        'train_runner'      :   data_queue
    })
    net.model()
    solver = single_net_solver.initialize({
        'working_dir'           :   '../pretrained_siamese', 
        'batch_size'            :   128,
        'max_iter'              :   375000, # 60 epochs
        'val_iter'              :   6250, 
        'save_iter'             :   6250, 
        'log_iter'              :   2*100,
        'lr_start_decay'        :   250000, # decay after 40 epochs
        'lr_decay_every'        :   62500,  # decay every 10 epochs
        'learning_rate'         :   0.0001, 
        'pretrained_resnet'     :   '../pretrained/resnet_v2_50_2017_04_14/resnet_v2_50.ckpt'
    })
    solver.setup_net(net)
    solver.setup_data(data)
    solver.setup_summary()
    solver.train()
