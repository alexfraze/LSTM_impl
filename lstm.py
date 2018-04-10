"""
Long Short Term Memory for character level entity classification
"""
import sys
import argparse
import numpy as np
from xman import *
from utils import *
from autograd import *

np.random.seed(0)

def glorot(m,n):
    # return scale for glorot initialization
    return np.sqrt(6./(m+n))

class LSTM(object):
    """
    Long Short Term Memory + Feedforward layer
    Accepts maximum length of sequence, input size, number of hidden units and output size
    """
    def __init__(self, max_len, in_size, num_hid, out_size):
        self.my_xman = self._build(max_len, in_size, num_hid, out_size) #DO NOT REMOVE THIS LINE. Store the output of xman.setup() in this variable
        print self.my_xman.operationSequence(self.my_xman.loss)

    def iterate_through_sequence(self, max_len):
        pass

    def _build(self, max_len, in_size, num_hid, out_size):
        x = XMan()
        print "INITIAZLIZING with layer amount:", num_hid
        self.params = {}
        sc = glorot(in_size, out_size)
        
        #cell_state
        self.params['cell_state'] = f.param(name='cell_state',
                default=sc*np.random.uniform(low=-1.,high=1., size=(num_hid,)))
        
        #Weights    
        self.params['W_input'] = f.param(name='W_input',
                default=sc*np.random.uniform(low=-1.,high=1., size=(in_size, num_hid)))
        self.params['W_forget'] = f.param(name='W_forget',
                default=sc*np.random.uniform(low=-1.,high=1., size=(in_size, num_hid)))
        self.params['W_output'] = f.param(name='W_output',
                default=sc*np.random.uniform(low=-1.,high=1., size=(in_size, num_hid)))
        self.params['W_candidate'] = f.param(name='W_candidate',
                default=sc*np.random.uniform(low=-1.,high=1., size=(in_size, num_hid)))

        #Peep Weights?
        self.params['U_input'] = f.param(name='U_input',
                default=sc*np.random.uniform(low=-1.,high=1., size=(num_hid, num_hid)))
        self.params['U_forget'] = f.param(name='U_forget',
                default=sc*np.random.uniform(low=-1.,high=1., size=(num_hid, num_hid)))
        self.params['U_output'] = f.param(name='U_output',
                default=sc*np.random.uniform(low=-1.,high=1., size=(num_hid, num_hid)))
        self.params['U_candidate'] = f.param(name='U_candidate',
                default=sc*np.random.uniform(low=-1.,high=1., size=(num_hid, num_hid)))

        #Biases
        self.params['B_input'] = f.param(name='B_input',
                default=sc*np.random.uniform(low=-1.,high=1., size=(num_hid,)))
        self.params['B_forget'] = f.param(name='B_forget',
                default=sc*np.random.uniform(low=-1.,high=1., size=(num_hid,)))
        self.params['B_output'] = f.param(name='B_output',
                default=sc*np.random.uniform(low=-1.,high=1., size=(num_hid,)))
        self.params['B_candidate'] = f.param(name='B_candidate',
                default=sc*np.random.uniform(low=-1.,high=1., size=(num_hid,)))

        self.inputs = {}
        self.inputs['X'] = f.input(name='X', default=np.random.rand(1,in_size))
        self.inputs['y'] = f.input(name='y', default=np.random.rand(1,out_size))

        x = XMan()
        self.iterate_through_sequence()
        inp = self.inputs["X"]
        outp = self.inputs["Y"]
        gates = []
        for gate in ["input", "forget", "output"]:
            oo = f.mul(inp, self.params["W_" + gate]) + f.mul(outp, self.params["U_" + gate]) + self.params["B_" + gate]
            gate = f.relu(oo)
            gates.append(gate)
        #candidate
        oo = f.mul(inp, self.params["W_canidate"]) + f.mul(outp, self.params["U_candidate"]) + self.params["B_candidate"]        
        gates.append(f.tanh(oo))
        self.params['cell_state'] = f.hadamard(gates[1],self.params['cell_state']) + f.hadamard(gates[0], gates[3])
        pre_output = f.hadamard(gates[2], f.tanh(self.params["cell_state"]))    

        return x.setup()

def main(params):
    epochs = params['epochs']
    max_len = params['max_len']
    num_hid = params['num_hid']
    batch_size = params['batch_size']
    dataset = params['dataset']
    init_lr = params['init_lr']
    output_file = params['output_file']
    train_loss_file = params['train_loss_file']

    # load data and preprocess
    dp = DataPreprocessor()
    data = dp.preprocess('%s.train'%dataset, '%s.valid'%dataset, '%s.test'%dataset)
    # minibatches
    mb_train = MinibatchLoader(data.training, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    mb_valid = MinibatchLoader(data.validation, len(data.validation), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)
    mb_test = MinibatchLoader(data.test, len(data.test), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)

    # build
    print "building lstm..."
    lstm = LSTM(max_len,mb_train.num_chars,num_hid,mb_train.num_labels)
    #OPTIONAL: CHECK GRADIENTS HERE


    print "done"

    # train
    print "training..."
    # get default data and params
    value_dict = lstm.my_xman.inputDict()
    lr = init_lr
    train_loss = np.ndarray([0])
    best_param_dict = {}
    for i in range(epochs):
        for ii, (idxs,e,l) in enumerate(mb_train):
            # prepare input
            data_dict = mlp.data_dict(e.reshape((e.shape[0],e.shape[1]*e.shape[2])),l)
            for k,v in data_dict.iteritems():
                value_dict[k] = v
            # fwd-bwd
            vd = fwd(mlp,value_dict)
            gd = bwd(mlp,value_dict)
            value_dict = update(mlp, value_dict, gd, lr)
            message = 'TRAIN loss = %.3f' % vd['loss']
            logger.write(message+'\n')
            train_loss = np.append(train_loss, vd['loss'])
        print ii
        # validate
        tot_loss, n= 0., 0
        probs = []
        targets = []
        for (idxs,e,l) in mb_valid:
            # prepare input
            data_dict = mlp.data_dict(e.reshape((e.shape[0],e.shape[1]*e.shape[2])),l)
            for k,v in data_dict.iteritems():
                value_dict[k] = v
            # fwd
            vd = fwd(mlp, value_dict)
            tot_loss += vd['loss']
            probs.append(vd['output'])
            targets.append(l)
            n += 1
        acc = accuracy(np.vstack(probs), np.vstack(targets))
        c_loss = tot_loss/n
        if c_loss<min_loss:
            min_loss = c_loss
            for k,v in value_dict.iteritems():
                best_param_dict[k] = np.copy(v)
        message = ('Epoch %d VAL loss %.3f min_loss %.3f acc %.3f' %
                (i,c_loss,min_loss,acc))
        logger.write(message+'\n')
        print message

    np.save(train_loss_file, train_loss)

    for (idxs,e,l) in mb_test:
        # prepare input and do a fwd pass over it to compute the output probs
        
    #TODO save probabilities on test set
    # ensure that these are in the same order as the test input
    #np.save(output_file, ouput_probabilities)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', dest='max_len', type=int, default=10)
    parser.add_argument('--num_hid', dest='num_hid', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--dataset', dest='dataset', type=str, default='tiny')
    parser.add_argument('--epochs', dest='epochs', type=int, default=15)
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.5)
    parser.add_argument('--output_file', dest='output_file', type=str, default='output')
    parser.add_argument('--train_loss_file', dest='train_loss_file', type=str, default='train_loss')
    params = vars(parser.parse_args())
    main(params)
