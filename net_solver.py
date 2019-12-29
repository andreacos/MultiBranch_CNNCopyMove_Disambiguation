'''
Copyright is preserved to Quoc-Tin Phan (dimmoon2511[at]gmail.com)
'''

import tensorflow as tf 
import numpy as np 
import ops
import os
from tqdm import tqdm
import inspect
import shutil
import time
import pickle
slim = tf.contrib.slim

old_print = print
def new_print(*args, **kwargs):
    # if tqdm.tqdm.write raises error, use builtin print
    try:
        tqdm.tqdm.write(*args, **kwargs)
    except:
        old_print(*args, ** kwargs)
inspect.builtins.print = new_print

class NetSolver:

    def __init__(self, working_dir, batch_size=64, max_iter=10e5, val_iter=10e3, save_iter=10e3, log_iter=100, \
                 learning_rate=0.0001, lr_start_decay=None, lr_decay_every=None, pretrained_resnet=None):
        self.working_dir = working_dir
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.val_iter = val_iter
        self.save_iter = save_iter
        self.log_iter = log_iter
        self.pretrained_resnet = pretrained_resnet
        self.learning_rate = learning_rate
        self.lr_start_decay = lr_start_decay
        self.lr_decay_every = lr_decay_every

    def save_log(self):
        path = os.path.join(self.working_dir,'model','log.pkl')
        with open(path,'wb') as f:
            pickle.dump(self.log, f)
    
    def load_log(self):
        path = os.path.join(self.working_dir,'model','log.pkl')
        if os.path.exists(path):
            with open(path,'rb') as f:
                data = pickle.load(f)
                self.log['costs'] = data['costs']
                self.log['val_err'] = data['val_err']

    def load_model(self, ckpt_id=None):
        saver = tf.train.Saver()
        path = os.path.join(self.working_dir, 'model')
        if ckpt_id:
            ckpt =  os.path.join(path, 'saved-model-' + str(ckpt_id))
            saver.restore(self.sess, ckpt)
            print('\nLoaded %s\n'%ckpt)
        else:
            ckpt = tf.train.latest_checkpoint(path)
            print('\nFound latest model: %s\n'%ckpt)
            if ckpt:
                saver.restore(self.sess, ckpt)
                print('\nLoaded %s\n'%ckpt)
    
    def load_resnet(self):
        saver = tf.train.Saver()
        all_variables = slim.get_model_variables()
        vars_to_restore = []
        exclusions = self.net.exclude_finetune_scopes()
        for var in all_variables:
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    break
            else:
                vars_to_restore.append(var)
        init_fn = slim.assign_from_checkpoint_fn(self.pretrained_resnet, vars_to_restore,
            ignore_missing_vars=False)
        init_fn(self.sess)
        print('\nLoaded variables from %s\n'%self.pretrained_resnet)

    def save_model(self):
        saver = tf.train.Saver()
        if not os.path.isdir(self.working_dir):
            os.makedirs(self.working_dir, exist_ok=True)
        if not os.path.isdir(os.path.join(self.working_dir, 'figure')):
            os.makedirs(os.path.join(self.working_dir, 'figure'), exist_ok=True)
        if not os.path.isdir(os.path.join(self.working_dir, 'model')):
            os.makedirs(os.path.join(self.working_dir, 'model'), exist_ok=True)
        path = os.path.join(self.working_dir, 'model','saved-model')
        save_path = saver.save(self.sess, path, global_step=self.net.global_iter.eval(session=self.sess))
        print('\nSave dir %s\n' % save_path)

    def setup_net(self, net, create_summary=True, ckpt_id=None):
        self.net = net
        self.sess = tf.Session(config=ops.config(self.net.use_gpu))
        self.log = {'costs':[], 'val_err':[]}
        if ckpt_id:
            self.load_model(ckpt_id=ckpt_id)
            self.load_log()
            self.i = self.net.global_iter.eval(session=self.sess)
        else:
            self.sess.run(tf.global_variables_initializer())
            print('Initializing from scratch')
            self.i = 0
            if self.pretrained_resnet is not None:
                assert os.path.exists(self.pretrained_resnet), 'Resnet checkpoint not found'
                self.load_resnet()
        
        if create_summary:
            summary_dir = os.path.join(self.working_dir, 'summaries', 'train_it_%d' % self.i)
            if os.path.isdir(summary_dir):
                shutil.rmtree(summary_dir)
            os.makedirs(summary_dir, exist_ok=True)
            self.train_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        self.start_i = self.i

        if self.net.use_tf_threading:
            self.coord = tf.train.Coordinator()
            self.net.train_runner.start_p_threads(self.sess)
            tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
            self.queue_size = self.net.train_runner.tf_queue.size()
    
    def setup_data(self, data):
        self.data = data
        self.data_fn = data.data_fn # a function that reads data for TRAINING only
    
    def setup_summary(self):
        max_outputs = 2
        self.summary = [
            tf.summary.image('input_a', self.net.im_a, max_outputs=max_outputs),
            tf.summary.image('input_b', self.net.im_b, max_outputs=max_outputs),
            tf.summary.scalar('total_loss', self.net.loss),
            tf.summary.scalar('learning_rate', self.net._opt._lr)
        ]
        for grad, var in self.net.avg_grads:
            self.summary.append(tf.summary.histogram(var.name + '/gradient', grad))
        self.merged_summary = tf.summary.merge_all()
    
    def add_summary(self):
        self.waiting_for_data(self.batch_size)
        self.train_writer.add_summary(self.sess.run(self.merged_summary, \
            feed_dict={self.net.learning_rate:self.learning_rate}), global_step=self.i)

    def validate_couple(self):
        n_gpus = len(self.net.use_gpu)
        batch_size = 32*n_gpus
        n_val_batches = (self.data.n_val // batch_size)
                        #+ int(self.data.n_val % batch_size != 0)
        #n_val_batches = 100
        print('# validation batches: %d\n' % n_val_batches)
        offset = 0
        labels = None
        predictions = None
        for i in range(n_val_batches):
            if i != n_val_batches-1:
                data1, data2 = self.data.data_val(offset, batch_size)
                offset += batch_size
            else:
                if self.data.n_val > offset + batch_size:
                    data1,data2 = self.data.data_val(offset, batch_size)
                else:
                    data1,data2 = self.data.data_val(offset, self.data.n_val-offset)
            _,pred_cls = self.predict_couple(data1[0], data1[1], data2[0], data2[1])
            print('Labels [%d, %d]' % (data1[2][0], data2[2][0]))
            print('Predictions [%d, %d]' % (pred_cls[0,0], pred_cls[0,1]))
            if i == 0:
                labels = np.concatenate((data1[2], data2[2]), axis=1)
                predictions = pred_cls
            else:
                labels = np.concatenate((labels, np.concatenate((data1[2], data2[2]), axis=1)))
                predictions = np.concatenate((predictions, pred_cls))
            
        labels = labels.astype(int)
        predictions = predictions.astype(int)
        acc = np.sum(predictions == labels, dtype=np.float32) / (2*labels.shape[0])
        print('Acc %f' % acc)
        return 1 - acc

    def predict_couple(self, im_a, im_b, im_c, im_d):
        if im_a.ndim < 4:
            im_a, im_b, im_c, im_d = (im_a[np.newaxis,...], im_b[np.newaxis,...], \
                                      im_c[np.newaxis,...], im_d[np.newaxis,...])
        return self.sess.run((self.net.pred,self.net.cls), feed_dict={
                self.net.im_a: im_a,
                self.net.im_b: im_b,
                self.net.im_c: im_c,
                self.net.im_d: im_d,
                self.net.is_training:False
        })
    
    def _train_couple(self):
        if self.net.use_tf_threading:
            [_, loss] = self.sess.run([self.net.opt, self.net.loss], feed_dict={
                self.net.is_training:True,
                self.net.learning_rate:self.learning_rate
            })
        else:
            data1,data2 = self.data.data_train(self.batch_size)
            [_, loss] = self.sess.run([self.net.opt, self.net.loss], feed_dict={
                self.net.im_a:data1[0],
                self.net.im_b:data1[1],
                self.net.im_c:data2[0],
                self.net.im_d:data2[1],
                self.net.label1:data1[2],
                self.net.label2:data2[2],
                self.net.is_training:True,
                self.net.learning_rate:self.learning_rate
            })
        return loss

    def waiting_for_data(self, size):
        while True:
            tf_queue_size = self.sess.run(self.queue_size)
            if tf_queue_size >= size:
                return
            else:
                print('Queue size %d \n' % tf_queue_size)
                time.sleep(0.5)

    def train(self):
        n_iters = int(self.max_iter - self.start_i)
        print('Train for %d iterations' % n_iters)
        t_obj = tqdm(range(n_iters))
        time.sleep(5)
        for t in t_obj:
            self.waiting_for_data(self.net.batch_size)
            loss = self._train_couple()
            print('Loss: %f\n' % loss)
            self.i += 1
            if self.i % self.save_iter == 0:
                self.save_model()
                self.save_log()
            if self.i % self.log_iter == 0:
                self.log['costs'].append(loss)
                self.add_summary()
            if self.i % self.val_iter == 0:
                val_err = self.validate_couple()
                print('Val error %f' % val_err)
                self.log['val_err'].append(val_err)
            if self.lr_start_decay is not None and \
                    self.i >= self.lr_start_decay and \
                    (self.i-self.lr_start_decay)%self.lr_decay_every == 0:
                self.learning_rate /= 2

def initialize(args):
    return NetSolver(**args)