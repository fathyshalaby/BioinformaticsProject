# -*- coding: utf-8 -*-
"""architecture.py: short description


Author -- Michael Widrich
Created on -- 2019-03-15
Contact -- michael.widrich@jku.at

long description


=======  ==========  =================  ================================
Version  Date        Author             Description
0.1      2019-03-15  Michael Widrich    -
=======  ==========  =================  ================================

"""

import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from matplotlib import pyplot as plt


def nograd(t):
    return t.detach()

class LSTMLayer(nn.Module):
    def __init__(self, in_features, out_features,
                 W_ci=nn.init.normal_, W_ig=nn.init.normal_, W_og=nn.init.normal_, W_fg=False,
                 b_ci=nn.init.normal_, b_ig=nn.init.normal_, b_og=nn.init.normal_, b_fg=False,
                 a_ci=torch.tanh, a_ig=torch.sigmoid, a_og=torch.sigmoid, a_fg=lambda x: x, a_out=torch.tanh,
                 c_init=lambda t: nn.init.constant_(t, val=0).detach_(),
                 h_init=lambda t: nn.init.constant_(t, val=0).detach_(),
                 output_dropout_rate=0., return_all_seq_pos=False, n_tickersteps=0,
                 b_ci_tickers=nn.init.normal_, b_ig_tickers=nn.init.normal_, b_og_tickers=nn.init.normal_,
                 b_fg_tickers=False, inputformat='NLC'):
        """LSTM layer for different types of sequence predictions with inputs of shape [samples, sequence positions,
        features]

        Parameters
        -------
        in_features : int
            Number of input features
        out_features : int
            Number of output features (=number of LSTM blocks)
        W_ci, W_ig, W_og, W_fg : (list of) initializer functions or False
            Initial values or initializers for cell input, input gate, output gate, and forget gate weights; Can be list
            of 2 elements as [W_fwd, W_rec] to define different weight initializations for forward and recurrent
            connections respectively; If single element, forward and recurrent connections will use the same
            initializer/tensor; If set to False, connection will be cut;
            Shape of weights is W_fwd: [n_inputs, n_outputs], W_rec: [in_features, out_features];
        b_ci, b_ig, b_og, b_fg : initializer function or False
            Initial values or initializers for bias for cell input, input gate, output gate, and forget gate;
            If set to False, connection will be cut;
        a_ci, a_ig, a_og, a_fg, a_out : torch function
            Activation functions for cell input, input gate, output gate, forget gate, and LSTM output respectively;
        c_init : initializer function
            Initial values for cell states; Default: Zero and not trainable;
        h_init : initializer function
            Initial values for hidden states; Default: Zero and not trainable;
        output_dropout : float or False
            Dropout rate for LSTM output dropout (i.e. dropout of whole LSTM unit with rescaling of the remaining
            units); This also effects the recurrent connections;
        return_all_seq_pos : bool
            True: Return output for all sequence positions (continuous prediction);
            False: Only return output at last sequence position (single target prediction);
        n_tickersteps : int or False
            Number of ticker- or tinker-steps; n_tickersteps sequence positions without forward input will be added
            at the end of the sequence; During tickersteps, additional bias units will be added to the LSTM input;
            This allows the LSTM to perform computations after the sequence has ended;
        b_ci_tickers, b_ig_tickers, b_og_tickers, b_fg_tickers : initializer function or False
            Initializers for bias applied during ticker steps for cell input, input gate, output gate, and forget gate;
            If set to False, connection will be cut;
        inputformat : 'NCL' or 'NLC'
            Input tensor format;
            'NCL' -> (batchsize, channels, seq.length);
            'NLC' -> (batchsize, seq.length, channels);

        Returns
        -------
        h_out : tensor
            return_all_seq_pos == TRuE: LSTM output for all sequence positions in shape
            (batchsize, seq.len., out_features) or (batchsize, out_features, seq.len.), depending on inputformat;
            return_all_seq_pos == False: LSTM output for last sequence positions in shape (batchsize, out_features);
        h : list of tensors
            LSTM hidden state for all sequence positions in list of tensors with shape (batchsize, out_features);
        c : list of tensors
            LSTM cell state for all sequence positions in list of tensors with shape (batchsize, out_features);
        lstm_inlets_activations : OrderedDict of list of tensors
            OrderedDict with keys ['ci', 'ig', 'og', 'fg'] containing LSTM inlet activations for all sequence positions
            in list of tensors with shape (batchsize, out_features);
        """
        super(LSTMLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lstm_inlets = ['ci', 'ig', 'og', 'fg']
        self.n_tickersteps = n_tickersteps
        self.output_dropout_rate = output_dropout_rate
        self.return_all_seq_pos = return_all_seq_pos
        self.inputformat = inputformat
        
        # Get activation functions
        self.a = OrderedDict(zip(self.lstm_inlets + ['out'], [a_ci, a_ig, a_og, a_fg, a_out]))
        
        # Get initializer for tensors
        def try_split_w(w, i):
            try:
                return w[i]
            except TypeError:
                return w
        
        self.W_fwd_init = OrderedDict(zip(self.lstm_inlets, [try_split_w(W, 0) for W in [W_ci, W_ig, W_og, W_fg]]))
        self.W_rec_init = OrderedDict(zip(self.lstm_inlets, [try_split_w(W, 1) for W in [W_ci, W_ig, W_og, W_fg]]))
        self.b_init = OrderedDict(zip(self.lstm_inlets, [b_ci, b_ig, b_og, b_fg]))
        self.b_tickers_init = OrderedDict(
            zip(self.lstm_inlets, [b_ci_tickers, b_ig_tickers, b_og_tickers, b_fg_tickers]))
        self.c_init = c_init
        self.h_init = h_init
        
        # Create parameters where needed and store them in dictionary for easier access
        self.W_fwd = OrderedDict(zip(self.lstm_inlets,
                                     [nn.Parameter(torch.FloatTensor(in_features, out_features))
                                      if self.W_fwd_init[i] is not False else False for i in self.lstm_inlets]))
        self.W_rec = OrderedDict(zip(self.lstm_inlets,
                                     [nn.Parameter(torch.FloatTensor(out_features, out_features))
                                      if self.W_rec_init[i] is not False else False for i in self.lstm_inlets]))
        self.b = OrderedDict(zip(self.lstm_inlets,
                                 [nn.Parameter(torch.FloatTensor(out_features))
                                  if self.b_init[i] is not False else False for i in self.lstm_inlets]))
        if self.n_tickersteps > 0:
            self.b_tickers = OrderedDict(zip(self.lstm_inlets,
                                             [nn.Parameter(torch.FloatTensor(out_features))
                                              if self.b_inits[i] is not False else False for i in self.lstm_inlets]))
        else:
            self.b_tickers = False
        
        # Register parameters with module
        _ = [self.register_parameter('W_fwd_{}'.format(name), param) for name, param in self.W_fwd.items()
             if param is not False]
        _ = [self.register_parameter('W_rec_{}'.format(name), param) for name, param in self.W_rec.items()
             if param is not False]
        _ = [self.register_parameter('b_{}'.format(name), param) for name, param in self.b.items() if param
             is not False]
        if self.n_tickersteps > 0:
            _ = [self.register_parameter('b_tickers_{}'.format(name), param) for name, param in self.b_tickers.items()
                 if param is not False]
        
        self.c_first = nn.Parameter(torch.FloatTensor(out_features))
        self.h_first = nn.Parameter(torch.FloatTensor(out_features))
        
        self.output_dropout_mask = None
        
        # Cell states and LSTM outputs at each timestep for each sample will be stored in a list
        self.c = []
        self.h = []
        
        # Activations of LSTM inlets at each timestep for each sample will be stored in a list
        self.lstm_inlets_activations = OrderedDict(zip(self.lstm_inlets, [[], [], [], []]))
        
        # Initialize tensors
        self.reset_parameters()
    
    def reset_parameters(self):
        
        # Apply initializer for W, b, and b_tickersteps
        _ = [self.W_fwd_init[i](self.W_fwd[i]) for i in self.lstm_inlets if self.W_fwd_init[i] is not False]
        _ = [self.W_rec_init[i](self.W_rec[i]) for i in self.lstm_inlets if self.W_rec_init[i] is not False]
        _ = [self.b_init[i](self.b[i]) for i in self.lstm_inlets if self.b_init[i] is not False]
        
        if self.n_tickersteps > 0:
            _ = [self.b_tickers_init[i](self.b_tickers[i]) for i in self.lstm_inlets
                 if self.b_tickers_init[i] is not False]
        
        # Initialize cell state and LSTM output at timestep -1
        self.c_init(self.c_first)
        self.h_init(self.h_first)
        
        # Manage LSTM output dropout (constant dropout of same units over time, including initial h state)
        if self.output_dropout_rate > 0:
            raise NotImplementedError("Sorry, LSTM output_dropout not available yet")
            self.output_dropout_mask = nograd(nn.init.uniform(nn.Parameter(torch.zeros_like(self.c_first),
                                                                           low=0, high=1)))
            self.a['out_dropout'] = lambda t: torch.where(self.output_dropout_mask > self.output_dropout_rate,
                                                          t, nograd(t * 0))
            self.h_first = self.a['out_dropout'](self.h_first)
    
    def reset_lstm_internals(self, n_batches):
        # Cell states and LSTM outputs at each timestep for each sample will be stored in a list
        self.c = [self.c_first.repeat((n_batches, 1))]
        self.h = [self.h_first.repeat((n_batches, 1))]
        
        # Activations of LSTM inlets at each timestep for each sample will be stored in a list
        self.lstm_inlets_activations = OrderedDict(zip(self.lstm_inlets, [[], [], [], []]))
        
        # Sample new output dropout mask
        if self.output_dropout_rate > 0:
            raise NotImplementedError("Sorry, LSTM output_dropout not available yet")
    
    def make_net_act(self, net_fwd, net_rec, b, a, b_ticker=False):
        net_act = None
        if net_fwd is not False:
            net_act = net_fwd
        if net_rec is not False:
            if net_act is not None:
                net_act += net_rec
            else:
                net_act = net_rec
        if b is not False:
            if net_act is not None:
                net_act += b[None, :]
            else:
                net_act = b[None, :]
        if b_ticker is not False:
            if net_act is not None:
                net_act += b_ticker[None, :]
            else:
                net_act = b_ticker[None, :]
        if net_act is not None:
            net_act = a(net_act)
        else:
            net_act = 1
        return net_act
     
    def forward(self, x):
        if self.inputformat == 'NLC':
            n_batches, n_seqpos, n_features = x.shape
            seqpos_slice = lambda seqpos: [slice(None), seqpos]
        elif self.inputformat == 'NCL':
            n_batches, n_features, n_seqpos = x.shape
            seqpos_slice = lambda seqpos: [slice(None), slice(None), seqpos]
        else:
            raise ValueError("Input format {} not supported".format(self.inputformat))
        
        self.reset_lstm_internals(n_batches=n_batches)
        
        for seq_pos in range(n_seqpos):
            # Calculate activations for LSTM inlets and append them to self.lstm_inlets_activations
            net_fwds = [torch.mm(x[seqpos_slice(seq_pos)], self.W_fwd[inlet]) if self.W_fwd[inlet] is not False
                        else False for inlet in self.lstm_inlets]
            net_fwds = OrderedDict(zip(self.lstm_inlets, net_fwds))
            net_recs = [torch.mm(self.h[-1], self.W_rec[inlet]) if self.W_rec[inlet] is not False
                        else False for inlet in self.lstm_inlets]
            net_recs = OrderedDict(zip(self.lstm_inlets, net_recs))
            
            net_acts = [self.make_net_act(net_fwds[inlet], net_recs[inlet], self.b[inlet], self.a[inlet])
                        for inlet in self.lstm_inlets]
            _ = [self.lstm_inlets_activations[inlet].append(net_acts[i]) for i, inlet in enumerate(self.lstm_inlets)]
            
            # Calculate new cell state
            self.c.append(self.lstm_inlets_activations['ci'][-1] * self.lstm_inlets_activations['ig'][-1]
                          + self.c[-1] * self.lstm_inlets_activations['fg'][-1])
            
            # Calculate new LSTM output with new cell state
            if self.output_dropout_rate > 0:
                self.h.append(self.a['out_dropout'](self.a['out'](self.c[-1])) * self.lstm_inlets_activations['og'][-1])
            else:
                self.h.append(self.a['out'](self.c[-1]) * self.lstm_inlets_activations['og'][-1])
        
        # Process tickersteps
        for _ in range(self.n_tickersteps):
            # Calculate activations for LSTM inlets and append them to self.lstm_inlets_activations, but during
            # tickersteps add tickerstep biases and set net_fwd to False
            net_recs = [torch.mm(self.h[-1], self.W_rec[inlet]) if self.W_rec[inlet] is not False
                        else False for inlet in self.lstm_inlets]
            net_recs = OrderedDict(zip(self.lstm_inlets, net_recs))
            net_acts = [self.make_net_act(False, net_recs[inlet], self.b[inlet], self.a[inlet], self.b_tickers[inlet])
                        for inlet in self.lstm_inlets]
            _ = [self.lstm_inlets_activations[inlet].append(net_acts[i]) for i, inlet in enumerate(self.lstm_inlets)]
            
            # Calculate new cell state
            self.c.append(self.lstm_inlets_activations['ci'][-1] * self.lstm_inlets_activations['ig'][-1]
                          + self.c[-1] * self.lstm_inlets_activations['fg'][-1])
            
            # Calculate new LSTM output with new cell state
            if self.output_dropout_rate > 0:
                self.h.append(self.a['out_dropout'](self.a['out'](self.c[-1])) * self.lstm_inlets_activations['og'][-1])
            else:
                self.h.append(self.a['out'](self.c[-1]) * self.lstm_inlets_activations['og'][-1])
        
        if self.return_all_seq_pos:
            if self.inputformat == 'NLC':
                h_out = torch.stack(self.h[1:], 1)
            elif self.inputformat == 'NCL':
                h_out = torch.stack(self.h[1:], 2)
        else:
            h_out = self.h[-1]
        
        return h_out, self.h, self.c, self.lstm_inlets_activations
    
    def get_weights(self):
        """Return dictionaries for W_fwd and W_rec"""
        return self.W_fwd, self.W_rec
    
    def get_biases(self):
        """Return dictionaries for with biases and """
        return self.b
    
    def tensor_to_numpy(self, t):
        try:
            t = t.numpy()
        except TypeError:
            t = t.cpu().numpy()
        except AttributeError:
            t = np.as_array(t)
        except RuntimeError:
            t = t.clone().data.cpu().numpy()
        return t
    
    def plot_internals(self, mb_index=0, filename=None, show_plot=False, fdict=None):
        """Plot LSTM internal states"""
        if filename is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        lstm_internals_copy = OrderedDict([(k, self.tensor_to_numpy(torch.stack([t[mb_index] for t in v], 0)))
                                           for k, v in self.lstm_inlets_activations.items() if v[0] is not 1])
        lstm_internals_copy['c'] = self.tensor_to_numpy(torch.stack([t[mb_index] for t in self.c], 0))
        lstm_internals_copy['h'] = self.tensor_to_numpy(torch.stack([t[mb_index] for t in self.h], 0))
        
        plot_labels = self.lstm_inlets + ['c', 'h']
        
        if fdict is None:
            fdict = dict()
        # plt.rcParams.update({'font.size': 5})
        fig, axs = plt.subplots(3, 2, **fdict)
        axs = [a for aa in axs for a in aa]
        for i, ax in enumerate(axs[:len(plot_labels)]):
            label = plot_labels[i]
            ax.set_title(label)
            if label not in lstm_internals_copy.keys():
                continue
            
            p = ax.plot(lstm_internals_copy[label], label=label)
            ax.grid(True)
        
        plt.tight_layout()
        if filename is not None:
            fig.savefig(filename)
        
        if show_plot:
            fig.show()
        else:
            plt.close(fig)
            del fig

