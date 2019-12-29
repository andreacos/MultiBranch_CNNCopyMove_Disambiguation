'''
Copyright is preserved to Quoc-Tin Phan (dimmoon2511[at]gmail.com)
'''

from __future__ import print_function
import tensorflow as tf 
import ops
import copy, numpy as np
import tensorflow.contrib.layers as layers
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
slim = tf.contrib.slim

class SiameseNetCouple:
    def __init__(self, use_tf_threading=False, batch_size=64, im_size=64, \
                 use_gpu=0, is_training=True, train_runner=None, \
                 batch_norm=True):
        self.use_tf_threading = use_tf_threading
        self.batch_size = batch_size
        self.im_size = im_size
        self.use_gpu = use_gpu if type(use_gpu) is list else [use_gpu]
        self.train_runner = train_runner
        self.batch_norm = batch_norm

        if self.use_tf_threading:
            assert self.batch_size%len(self.use_gpu) == 0, 'batch size should be multiple of \
                   the number of GPUs'
            im_a, im_b, label1, im_c, im_d, label2 = self.train_runner.get_inputs(self.batch_size)
            self.im_a   = tf.placeholder_with_default(im_a, [None, self.im_size, self.im_size, 3])
            self.im_b   = tf.placeholder_with_default(im_b, [None, self.im_size, self.im_size, 3])
            self.im_c   = tf.placeholder_with_default(im_c, [None, self.im_size, self.im_size, 3])
            self.im_d   = tf.placeholder_with_default(im_d, [None, self.im_size, self.im_size, 3])
            self.label1  = tf.placeholder_with_default(label1, [None, 1])
            self.label2  = tf.placeholder_with_default(label2, [None, 1])
        else:
            self.im_a   = tf.placeholder(tf.float32, [None, self.im_size, self.im_size, 3])
            self.im_b   = tf.placeholder(tf.float32, [None, self.im_size, self.im_size, 3])
            self.im_c   = tf.placeholder(tf.float32, [None, self.im_size, self.im_size, 3])
            self.im_d   = tf.placeholder(tf.float32, [None, self.im_size, self.im_size, 3])
            self.label1  = tf.placeholder(tf.float32, [None, 1])
            self.label2  = tf.placeholder(tf.float32, [None, 1])
        self.labels = tf.concat([self.label1, self.label2], axis=-1)
        self.learning_rate = tf.placeholder(tf.float32, shape=())
        self.is_training = tf.placeholder_with_default(is_training, None)
        self.global_iter = tf.Variable(0, trainable=False)

    def model(self, preemptive_reuse=False, graph=None):
        """
        Initialize the model to train
        Support multiple GPUs
        """
        self.graph = graph
        with tf.variable_scope(tf.get_variable_scope()):
            # split data into n equal batches and distribute them onto multiple GPUs
            im_a_list   = tf.split(self.im_a, len(self.use_gpu))
            im_b_list   = tf.split(self.im_b, len(self.use_gpu))
            im_c_list   = tf.split(self.im_c, len(self.use_gpu))
            im_d_list   = tf.split(self.im_d, len(self.use_gpu))
            labels_list = tf.split(self.labels, len(self.use_gpu))
            # initialize the optimizer
            self._opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            # Used to average
            all_grads              = []
            all_out                = []
            all_loss               = []
            batchnorm_updates      = []
            for i, gpu_id in enumerate(self.use_gpu):
                print('Initializing graph on gpu(cpu) %i' % gpu_id)
                with tf.device('/gpu:%d' % gpu_id):
                    with tf.name_scope('tower_%d' % gpu_id):
                        if preemptive_reuse:
                            tf.get_variable_scope().reuse_variables()
                        
                        im_a, im_b = im_a_list[i], im_b_list[i]
                        im_c, im_d = im_c_list[i], im_d_list[i]
                        labels = labels_list[i]
                        with tf.name_scope('extract_feature_a') as scope:
                            im_a_feat = self.extract_features_resnet50(im_a, scope_name='feature_cnn')
                            self.im_a_feat = im_a_feat
                            # we should retain update ops of batch norm on one branch of the 1st tower
                            # because the var,mean,offset,scale are shared across branches, towers
                            if i == 0:
                                batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)

                        with tf.name_scope('extract_feature_b'):
                            im_b_feat = self.extract_features_resnet50(im_b, scope_name='feature_cnn', reuse=True)
                            self.im_b_feat = im_b_feat

                        with tf.name_scope('extract_feature_c'):
                            im_c_feat = self.extract_features_resnet50(im_c, scope_name='feature_cnn', reuse=True)
                            self.im_c_feat = im_c_feat
                        
                        with tf.name_scope('extract_feature_d'):
                            im_d_feat = self.extract_features_resnet50(im_d, scope_name='feature_cnn', reuse=True)
                            self.im_d_feat = im_d_feat

                        with tf.name_scope('predict_same'):
                            feat_ab = tf.concat([self.im_a_feat, self.im_b_feat], axis=-1)
                            feat_cd = tf.concat([self.im_c_feat, self.im_d_feat], axis=-1)
                            out_ab = self.predict(feat_ab, name='predict')
                            out_cd = self.predict(feat_cd, name='predict', reuse=True)
                            out = tf.concat([out_ab, out_cd], axis=-1)
                            all_out.append(out)
                        
                        with tf.name_scope('classification_loss'):
                            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=out)
                            all_loss.append(loss)

                    # once calling this, all variables are reused. A setting reuse=False is no more effective
                    tf.get_variable_scope().reuse_variables()
                    grad = self._opt.compute_gradients(loss, var_list=self.get_variables())
                    all_grads.append(grad) # List of lists of (gradient, variable) tuples

        # Average the gradient and apply
        avg_grads       = ops.average_gradients(all_grads)
        self.all_loss   = all_loss
        self.avg_grads  = avg_grads
        self.loss       = tf.reduce_mean(all_loss)
        
        # Trains all variables for now
        apply_grad_op  = self._opt.apply_gradients(self.avg_grads, global_step=self.global_iter)
        if len(batchnorm_updates)  != 0:
            batchnorm_updates_op    = tf.group(*batchnorm_updates)
            self.opt                = tf.group(apply_grad_op, batchnorm_updates_op)
        else:
            self.opt                = apply_grad_op
            
        # For logging results
        self.logits     = tf.concat(all_out, axis=0)
        self.pred       = tf.nn.softmax(self.logits)
        self.cls        = tf.round(self.pred)

    def get_variables(self):
        """
        Returns only variables that are needed.
        """
        var_list = tf.trainable_variables()
        assert len(var_list) > 0, 'No variables are linked to the optimizer'
        return var_list
    
    def exclude_finetune_scopes(self):
        return ['predict', 'resnet_v2_50/logits']

    def extract_features_resnet50(self, im, scope_name, reuse=False):
        use_global_pool = True
        num_classes = 512
        with tf.name_scope(scope_name):
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                out, _ = resnet_v2.resnet_v2_50(inputs=im,
                                                num_classes=num_classes,
                                                global_pool=use_global_pool,
                                                is_training=self.is_training,
                                                scope='resnet_v2_50',
                                                reuse=reuse)
        print('\nShape after Resnet_50\n')
        print(out.get_shape())
        out = layers.flatten(out)
        return out

    def predict(self, feat_ab, name, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            out = layers.stack(feat_ab, layers.fully_connected, [1024,256], scope='fc', reuse=reuse)
            out = layers.fully_connected(out, 1, activation_fn=None, scope='fc_out', reuse=reuse)
        return out
    
def initialize(args):
    return SiameseNetCouple(**args)