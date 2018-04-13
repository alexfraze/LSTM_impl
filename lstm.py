"""
Long Short Term Memory for character level entity classification
"""
import sys
import argparse
import numpy as np
from xman import *
from utils import *
from autograd import *
import time

np.random.seed(0)

def fwd(network, valueDict):
    ad = Autograd(network.my_xman)
    return ad.eval(network.my_xman.operationSequence(network.my_xman.loss), valueDict)

def bwd(network, valueDict):
    ad = Autograd(network.my_xman)
    return ad.bprop(network.my_xman.operationSequence(network.my_xman.loss), valueDict,loss=np.float_(1.0))


def update(network, dataParamDict, grads, rate):
    for rname in grads:
        if network.my_xman.isParam(rname):
            dataParamDict[rname] = dataParamDict[rname] - rate*grads[rname]
    return dataParamDict

def accuracy(probs, targets):
    preds = np.argmax(probs, axis=1)
    targ = np.argmax(targets, axis=1)
    # print(preds)
    # print(targ)
    return float((preds==targ).sum())/preds.shape[0]


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
        # print self.my_xman.operationSequence(self.my_xman.loss)

    def iterate_through_sequence(self, max_len):
        for time in range(max_len-1,-1,-1):
            inp = self.inputs["x"+str(time)]
            gates = []
            for gate in ["input", "forget", "output"]:
                oo = f.mul(inp, self.params["W_" + gate]) + f.mul(self.inputs["LSTM_output"], self.params["U_" + gate]) + self.params["B_" + gate]
                gate = f.sigmoid(oo)
                gates.append(gate)
            #candidate
            oo = f.mul(inp, self.params["W_candidate"]) + f.mul(self.inputs["LSTM_output"], self.params["U_candidate"]) + self.params["B_candidate"]        
            gates.append(f.tanh(oo))
            self.inputs['cell_state'] = f.hadamard(gates[1],self.inputs['cell_state']) + f.hadamard(gates[0], gates[3])
            self.inputs["LSTM_output"] = f.hadamard(gates[2], f.tanh(self.inputs["cell_state"]))            

    def _build(self, max_len, in_size, num_hid, out_size):
        x = XMan()
        # print "INITIAZLIZING with layer amount:", num_hid
        self.params = {}
        sc = glorot(in_size, num_hid)
        bc = glorot(num_hid, num_hid)
        
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
                default=bc*np.random.uniform(low=-1.,high=1., size=(num_hid, num_hid)))
        self.params['U_forget'] = f.param(name='U_forget',
                default=bc*np.random.uniform(low=-1.,high=1., size=(num_hid, num_hid)))
        self.params['U_output'] = f.param(name='U_output',
                default=bc*np.random.uniform(low=-1.,high=1., size=(num_hid, num_hid)))
        self.params['U_candidate'] = f.param(name='U_candidate',
                default=bc*np.random.uniform(low=-1.,high=1., size=(num_hid, num_hid)))

        #Biases
        self.params['B_input'] = f.param(name='B_input',
                default=0.1*np.random.uniform(low=-1.,high=1., size=(num_hid,)))
        self.params['B_forget'] = f.param(name='B_forget',
                default=0.1*np.random.uniform(low=-1.,high=1., size=(num_hid,)))
        self.params['B_output'] = f.param(name='B_output',
                default=0.1*np.random.uniform(low=-1.,high=1., size=(num_hid,)))
        self.params['B_candidate'] = f.param(name='B_candidate',
                default=0.1*np.random.uniform(low=-1.,high=1., size=(num_hid,)))

        #Feedforward Layer
        self.params['W_forward'] = f.param(name='W_forward', default=glorot(num_hid,out_size)*np.random.uniform(low=-1.,high=1., size=(num_hid, out_size)))
        self.params['B_forward'] = f.param(name='B_forward', default=0.1*np.random.uniform(low=-1.,high=1.,size=(out_size,)))
    

        self.inputs = {}
        #cell_state and LSTM output
        self.inputs['cell_state'] = f.input(name='cell_state',
                default=np.zeros(shape=(1,num_hid)))
        self.inputs["LSTM_output"] = f.input(name='LSTM_output',
                default=np.zeros(shape=(1,num_hid)))

        for time in range(max_len):
            self.inputs['x' + str(time)] = f.input(name='x'+str(time), default=np.random.rand(1,in_size))
        self.inputs['y'] = f.input(name='y', default=np.random.rand(1,out_size))

        # x = XMan()
        # inp = self.inputs["X"]
        # outp = self.inputs["y"]
        self.iterate_through_sequence(max_len) #pass in sequence here?
        # pre_output = self.params["LSTM_output"]
        x.output = f.softMax(f.relu(f.mul(self.inputs["LSTM_output"], self.params["W_forward"])+self.params["B_forward"]))
        x.loss = f.mean(f.crossEnt(x.output, self.inputs['y']))
        return x.setup()


    def data_dict(self, X, y):
        dataDict = {}
        # dataDict['X'] = X
        # for time in range(max_len):
        dataDict['X']  = X 
        dataDict['y'] = y
        return dataDict


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
    # print "building lstm..."
    lstm = LSTM(max_len,mb_train.num_chars,num_hid,mb_train.num_labels)
    #OPTIONAL: CHECK GRADIENTS HERE
    logger = open('%s_mlp4c_L%d_H%d_B%d_E%d_lr%.3f.txt'%
            (dataset,max_len,num_hid,batch_size,epochs,init_lr),'w')
    # get default data and params
    

    # print "done"

    # train
    # print "training..."
    # get default data and params
    value_dict = lstm.my_xman.inputDict()
    lr = init_lr
    train_loss = np.ndarray([0])
    min_loss = 1e5
    best_param_dict = {}
    for i in range(epochs):
        for ii, (idxs,e,l) in enumerate(mb_train):
            V = e.shape[2]
            N = e.shape[0]
            # data_dict = lstm.data_dict(e.reshape((e.shape[0],e.shape[1]*e.shape[2])),l)
            #data pre-processing
            value_dict["y"] = l
            value_dict['cell_state'] = np.zeros((N,num_hid))
            value_dict["LSTM_output"] = np.zeros((N,num_hid))
            for segment in range(max_len):
                value_dict["x"+str(segment)] = e[:,segment,:] 
            vd = fwd(lstm,value_dict)
            # print(vd["output"])
            # time.sleep(2)
            gd = bwd(lstm,value_dict)
            value_dict = update(lstm, value_dict, gd, lr)
            message = 'TRAIN loss = %.3f' % vd['loss']
            logger.write(message+'\n')
            train_loss = np.append(train_loss, vd['loss'])
        # print ii
        # validate
        tot_loss, n= 0., 0
        probs = []
        targets = []
        for (idxs,e,l) in mb_valid:
            # prepare input

            # V = e.shape[2]
            # N = e.shape[0]
            # data_dict = lstm.data_dict(e.reshape((e.shape[0],e.shape[1]*e.shape[2])),l)
            # for k,v in data_dict.iteritems():
            #     value_dict[k] = v
            # fwd            
            value_dict["y"] = l
            value_dict['cell_state'] = np.zeros((N,num_hid))
            value_dict["LSTM_output"] = np.zeros((N,num_hid))
            for segment in range(max_len):
                value_dict["x"+str(segment)] = e[:,segment,:] 

            vd = fwd(lstm, value_dict)
            tot_loss += vd['loss']
            probs.append(vd['output'])
            # print(vd["output"])
            targets.append(l)
            n += 1
        # acc = accuracy(np.vstack(probs), np.vstack(targets))
        c_loss = tot_loss/n
        if c_loss<min_loss:
            min_loss = c_loss
            for k,v in value_dict.iteritems():
                best_param_dict[k] = np.copy(v)
        # message = ('Epoch %d VAL loss %.3f min_loss %.3f acc %.3f' %
        #         (i,c_loss,min_loss,acc))
        # logger.write(message+'\n')
        # print message

    np.save(train_loss_file, train_loss)
    
    tot_loss, n = 0., 0
    probs = []
    targets = []
    for (idxs,e,l) in mb_test:
        # prepare input
        # data_dict = lstm.data_dict(e.reshape((e.shape[0],e.shape[1]*e.shape[2])),l)
        # for k,v in data_dict.iteritems():
            # best_param_dict[k] = v


        best_param_dict["y"] = l
        best_param_dict['cell_state'] = np.zeros((N,num_hid))
        best_param_dict["LSTM_output"] = np.zeros((N,num_hid))
        for segment in range(max_len):
            best_param_dict["x"+str(segment)] = e[:,segment,:] 

        # fwd
        vd = fwd(lstm,best_param_dict)
        tot_loss += vd['loss']
        probs.append(vd['output'])
        targets.append(l)
        n += 1
    # acc = accuracy(np.vstack(probs), np.vstack(targets))
    c_loss = tot_loss/n
    # print(np.vstack(probs))
    np.save(output_file, np.vstack(probs))
    # print "done, test loss = %.3f acc = %.3f" % (c_loss, acc)

    #TODO save probabilities on test set
    # ensure that these are in the same order as the test input
    #np.save(output_file, ouput_probabilities)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', dest='max_len', type=int, default=10)
    parser.add_argument('--num_hid', dest='num_hid', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--dataset', dest='dataset', type=str, default='tiny')
    parser.add_argument('--epochs', dest='epochs', type=int, default= 25)
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.5)
    parser.add_argument('--output_file', dest='output_file', type=str, default='output')
    parser.add_argument('--train_loss_file', dest='train_loss_file', type=str, default='train_loss')
    params = vars(parser.parse_args())
    main(params)
