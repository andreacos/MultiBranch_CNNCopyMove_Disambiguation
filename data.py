'''
Copyright is preserved to Quoc-Tin Phan (dimmoon2511[at]gmail.com)
'''

import numpy as np 
import random
import h5py

class Data_Single():
    def __init__(self, file_lst):
        self.lst = get_data_lst(file_lst['pos'])
        n = len(self.pos_lst)
        split_ratio = [0.8,0.1,0.1]
        # training set
        s = 0
        self.train_lst = self.lst[s:s+int(n*split_ratio[0])]
        self.n_train = len(self.train_lst)
        # validation
        s = s+int(n*split_ratio[0])
        self.val_lst = self.lst[s:s+int(n*split_ratio[1])]
        self.n_val = len(self.val_lst)
        # test
        s = s+int(n*split_ratio[1])
        self.test_lst = self.lst[s:]
        self.n_test = len(self.test_lst)
    
    def data_fn(self):
        """
        read training data, no argument required
        read a pair and return a list [x1, x2, y]
        """
        
        data_lst = self.train_lst
        # select a random index
        idx = np.random.random_integers(0, self.n_train-1, 1)[0]
        f = h5py.File(data_lst[idx], 'r')
        x1 = np.asarray(f['x1'], dtype=np.float32).T / 255
        x2 = np.asarray(f['x2'], dtype=np.float32).T / 255
        f.close()
        # alternate the label, permutate the order of x1,x2
        b = np.random.random(1) >= 0.5
        y = np.array([1.0], dtype=np.float32) if b else np.array([0.0], dtype=np.float32)
        return [x1[np.newaxis,:], x2[np.newaxis,:], y[:,np.newaxis]] if b else \
               [x2[np.newaxis,:], x1[np.newaxis,:], y[:,np.newaxis]]
    
    def data_val(self, offset, batch_size=1):
        assert offset + batch_size <= self.n_val, 'not enough validation data'
        label = []
        for i in range(offset, offset+batch_size):
            f = h5py.File(self.val_lst[i], 'r')
            x1 = np.asarray(f['x1'], dtype=np.float32).T / 255
            x1 = x1[np.newaxis,:]
            x2 = np.asarray(f['x2'], dtype=np.float32).T / 255
            x2 = x2[np.newaxis,:]
            label.append(1.0)
            if i == offset:
                im_a = x1
                im_b = x2
            else:
                im_a = np.concatenate((im_a, x1), axis=0)
                im_b = np.concatenate((im_b, x2), axis=0)
            f.close()
        label = np.asarray(label, dtype=np.float32)
        return [im_a, im_b, label[:, np.newaxis]]
    
    def data_test(self, offset, batch_size=1):
        assert offset + batch_size <= self.n_test, 'not enough test data'
        label = []
        for i in range(offset, offset+batch_size):
            f = h5py.File(self.test_lst[i], 'r')
            x1 = np.asarray(f['x1'], dtype=np.float32).T / 255
            x1 = x1[np.newaxis,:]
            x2 = np.asarray(f['x2'], dtype=np.float32).T / 255
            x2 = x2[np.newaxis,:]
            label.append(1.0)
            if i == offset:
                im_a = x1
                im_b = x2
            else:
                im_a = np.concatenate((im_a, x1), axis=0)
                im_b = np.concatenate((im_b, x2), axis=0)
            f.close()
        label = np.asarray(label, dtype=np.float32)
        return [im_a, im_b, label[:, np.newaxis]]
    
    def data_train(self, batch_size=32):
        label = []
        perm = np.random.permutation(self.n_train)
        for i in range(batch_size):
            f = h5py.File(self.train_lst(perm[i]), 'r')
            x1 = np.asarray(f['x1'], dtype=np.float32).T / 255
            x1 = x1[np.newaxis,:]
            x2 = np.asarray(f['x2'], dtype=np.float32).T / 255
            x2 = x2[np.newaxis,:]
            b = np.random.random(1) >= 0.5
            label.append(b[0]*1.0) 
            if i == offset:
                im_a = x1 if b else x2
                im_b = x2 if b else x1
            else:
                im_a = np.concatenate((im_a, x1), axis=0) if b else np.concatenate((im_a, x2), axis=0)
                im_b = np.concatenate((im_b, x2), axis=0) if b else np.concatenate((im_b, x1), axis=0)
            f.close()
        label = np.array(label, dtype=np.float32)
        return [im_a, im_b, label[:, np.newaxis]]

class Data_Train_Val_Single(Data_Single):
    def __init__(self, file_lst, max_items=500000):
        self.lst = get_data_lst(file_lst['pos'])
        random.shuffle(self.lst)
        n = min(len(self.lst), max_items)
        self.lst = self.lst[:n]

        split_ratio = [0.9, 0.1]

        # training set
        s = 0
        self.train_lst = self.lst[:int(n*split_ratio[0])]
        self.n_train = len(self.train_lst)
        print('number of train samples %d' % self.n_train)

        # validation set
        s += int(n*split_ratio[0])
        self.val_lst = self.lst[s:s+int(n*split_ratio[1])]
        self.n_val = len(self.val_lst)
        print('number of val samples %d' % self.n_val)

    def data_test(self, offset, batch_size=1):
        assert False, 'Test data not available'



# global functions
def get_data_lst(file_):
    # read file paths
    f = open(file_, 'r')
    file_lst = []
    line = f.readline().strip('\n ')
    if line is not '':
        file_lst.append(line)
    while line:
        line = f.readline().strip('\n ')
        if line is not '':
            file_lst.append(line)
    f.close()
    return file_lst


class Data_Couple():
    def __init__(self, file_lst):
        self.lst = get_data_lst(file_lst['pos_neg'])
        n = len(self.lst)
        split_ratio = [0.8, 0.1, 0.1]

        # training set
        s = 0
        self.train_lst = self.lst[s:s+int(n*split_ratio[0])]
        self.n_train = len(self.train_lst)
        self.training_label = [(1.0, 0.0)]*self.n_train
        print('number of train samples %d' % self.n_train)

        # validation set
        s += int(n*split_ratio[0])
        self.val_lst = self.lst[s:s+int(n*split_ratio[1])]
        self.n_val = len(self.val_lst)
        self.val_label = [(1.0, 0.0)]*self.n_val
        c = list(zip(self.val_lst,self.val_label))
        random.shuffle(c) # shuffle validation samples
        self.val_lst[:], self.val_label[:] = zip(*c)
        print('number of val samples %d' % self.n_val)

        # test set
        s += int(n*split_ratio[1])
        self.test_lst = self.lst[s:s+int(n*split_ratio[2])]
        self.n_test = len(self.test_lst)
        self.test_label = [(1.0, 0.0)]*self.n_test
        c = list(zip(self.test_lst,self.test_label))
        random.shuffle(c) # shuffle validation samples
        self.test_lst[:], self.test_label[:] = zip(*c)
        print('number of test samples %d' % self.n_test)


    def data_fn(self): 
        """ 
        read training data, no argument required
        read two pair and return a list [x1, x2, x3, x4, y]
        """
        # select a random index
        idx = np.random.random_integers(0, self.n_train - 1, 1)[0]
        f = h5py.File(self.train_lst[idx], 'r')
        x1 = np.asarray(f['x1'], dtype=np.float32).T / 255
        x2 = np.asarray(f['x2'], dtype=np.float32).T / 255
        x3 = np.asarray(f['x3'], dtype=np.float32).T / 255
        x4 = np.asarray(f['x4'], dtype=np.float32).T / 255
        y = np.asarray([1.0], dtype=np.float32)
        b = [np.random.random(1) >= 0.5, np.random.random(1) >= 0.5, np.random.random(1) >= 0.5]
        pair1 = [x1[np.newaxis,:], x2[np.newaxis,:], y[:,np.newaxis]] if b[0] else \
                [x2[np.newaxis,:], x1[np.newaxis,:], y[:,np.newaxis]] # pos pair
        pair2 = [x3[np.newaxis,:], x4[np.newaxis,:], 1.0-y[:,np.newaxis]] if b[1] else \
                [x4[np.newaxis,:], x3[np.newaxis,:], 1.0-y[:,np.newaxis]]
        
        # permutate the order pairs to faciliate batch norm
        return [pair1[0], pair1[1], pair1[2], pair2[0], pair2[1], pair2[2]] if b[2] else \
               [pair2[0], pair2[1], pair2[2], pair1[0], pair1[1], pair1[2]]

    def data_val(self, offset, batch_size=1):
        assert offset + batch_size <= self.n_val, 'not enough validation data'
        label_pos = []
        label_neg = []

        for i in range(offset, offset+batch_size):
            f = h5py.File(self.val_lst[i], 'r')
            #keys = list(f.keys())
            x1 = np.asarray(f['x1'], dtype=np.float32).T / 255
            x1 = x1[np.newaxis,:]
            x2 = np.asarray(f['x2'], dtype=np.float32).T / 255
            x2 = x2[np.newaxis,:]
            x3 = np.asarray(f['x3'], dtype=np.float32).T / 255
            x3 = x3[np.newaxis,:]
            x4 = np.asarray(f['x4'], dtype=np.float32).T / 255
            x4 = x4[np.newaxis,:]
            label_pos.append(self.val_label[i][0])
            label_neg.append(self.val_label[i][1])
            if i == offset:
                im_a = x1
                im_b = x2
                im_c = x3
                im_d = x4
            else:
                im_a = np.concatenate((im_a, x1), axis=0)
                im_b = np.concatenate((im_b, x2), axis=0)
                im_c = np.concatenate((im_c, x3), axis=0)
                im_d = np.concatenate((im_d, x4), axis=0)
            f.close()
        label_pos = np.asarray(label_pos, dtype=np.float32)
        label_neg = np.asarray(label_neg, dtype=np.float32)
        data1 = [im_a, im_b, label_pos[:,np.newaxis]]
        data2 = [im_c, im_d, label_neg[:,np.newaxis]]
        return (data1, data2)
    
    def data_train(self, batch_size=32):
        label_pos = []
        label_neg = []
        perm = np.random.permutation(self.n_train)
        for i in range(batch_size):
            f = h5py.File(self.train_lst[perm[i]], 'r')
            label_pos.append(self.train_label[i][0])
            label_neg.append(self.train_label[i][1])
            #keys = list(f.keys())
            x1 = np.asarray(f['x1'], dtype=np.float32).T / 255
            x1 = x1[np.newaxis,:]
            x2 = np.asarray(f['x2'], dtype=np.float32).T / 255
            x2 = x2[np.newaxis,:]
            x3 = np.asarray(f['x3'], dtype=np.float32).T / 255
            x3 = x3[np.newaxis,:]
            x4 = np.asarray(f['x4'], dtype=np.float32).T / 255
            x4 = x4[np.newaxis,:]
            
            if i == 0:
                im_a = x1
                im_b = x2
                im_c = x3 
                im_d = x4
            else:
                im_a = np.concatenate((im_a, x1), axis=0)
                im_b = np.concatenate((im_b, x2), axis=0)
                im_c = np.concatenate((im_c, x3), axis=0)
                im_d = np.concatenate((im_d, x4), axis=0)
            f.close()
        b = [np.random.random(1) >= 0.5, np.random.random(1) >= 0.5, np.random.random(1) >= 0.5]
        label_pos = np.asarray(label_pos, dtype=np.float32)
        label_neg = np.asarray(label_neg, dtype=np.float32)
        data1 = [im_a, im_b, label_pos[:,np.newaxis]] if b[0] else \
                [im_b, im_a, label_pos[:,np.newaxis]]
        data2 = [im_c, im_d, label_neg[:,np.newaxis]] if b[1] else \
                [im_d, im_c, label_neg[:,np.newaxis]]
        # permutate the order pairs to faciliate batch norm
        return (data1, data2) if b[2] else (data2, data1)
        
    def data_test(self, offset, batch_size=1):
        assert offset + batch_size <= self.n_test, 'not enough test data'
        label_pos = []
        label_neg = []
        for i in range(offset, offset+batch_size):
            # positive samples: make sure neg pair and pos pair have the same index
            f = h5py.File(self.test_lst[i], 'r')
            x1 = np.asarray(f['x1'], dtype=np.float32).T / 255
            x1 = x1[np.newaxis,:]
            x2 = np.asarray(f['x2'], dtype=np.float32).T / 255
            x2 = x2[np.newaxis,:]
            x3 = np.asarray(f['x3'], dtype=np.float32).T / 255
            x3 = x3[np.newaxis,:]
            x4 = np.asarray(f['x4'], dtype=np.float32).T / 255
            x4 = x4[np.newaxis,:]
            label_pos.append(self.test_label[i][0])
            label_neg.append(self.test_label[i][1])
            f.close()

            if i == offset:
                im_a = x1
                im_b = x2
                im_c = x3
                im_d = x4
            else:
                im_a = np.concatenate((im_a,x1), axis=0)
                im_b = np.concatenate((im_b,x2), axis=0)
                im_c = np.concatenate((im_c,x3), axis=0)
                im_d = np.concatenate((im_d,x4), axis=0)

        label_pos = np.asarray(label_pos, dtype=np.float32)
        label_neg = np.asarray(label_neg, dtype=np.float32)
        data1 = [im_a, im_b, label_pos[:,np.newaxis]]
        data2 = [im_c, im_d, label_neg[:,np.newaxis]]
        return (data1, data2)

class Data_Train_Val_Couple(Data_Couple):
    def __init__(self, file_lst, max_items=500000):
        self.lst = get_data_lst(file_lst['pos_neg'])
        random.shuffle(self.lst)
        n = min(len(self.lst), max_items)
        self.lst = self.lst[:n]

        split_ratio = [0.9, 0.1]

        # training set
        s = 0
        self.train_lst = self.lst[:int(n*split_ratio[0])]
        self.n_train = len(self.train_lst)
        self.training_label = [(1.0, 0.0)]*self.n_train
        print('number of train samples %d' % self.n_train)

        # validation set
        s += int(n*split_ratio[0])
        self.val_lst = self.lst[s:s+int(n*split_ratio[1])]
        self.n_val = len(self.val_lst)
        self.val_label = [(1.0, 0.0)]*self.n_val
        c = list(zip(self.val_lst,self.val_label))
        random.shuffle(c) # shuffle validation samples
        self.val_lst[:], self.val_label[:] = zip(*c)
        print('number of val samples %d' % self.n_val)

    def data_test(self, offset, batch_size=1):
        assert False, 'Test data not available'
    

class Data_Test_Couple(Data_Couple):
    def __init__(self, file_lst, shuffle=False):
        self.test_lst = get_data_lst(file_lst['pos_neg'])
        # test set
        self.n_test = len(self.test_lst)
        print('number of test samples %d' % self.n_test)
        self.test_label = [(1.0, 0.0)]*len(self.test_lst)
        if shuffle:
            c = list(zip(self.test_lst, self.test_label)) 
            random.shuffle(c) # shuffle test samples
            self.test_lst[:], self.test_label[:] = zip(*c)

    def data_train(self, batch_size=32):
        assert False, 'Training data not available'

    def data_val(self, offset, batch_size=1):
        assert False, 'Validation data not available'

