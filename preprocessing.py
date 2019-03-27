# -*- coding: utf-8 -*-
"""preprocessing.py: tools for data pre-processing


Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""

import numpy as np
import torch


class PadToEqualLengths(object):
    def __init__(self, padding_dims: tuple=(1,), padding_values: tuple=(0,)):
        """Pads minibatch entries to equal length based on the maximal length in the minibatch
        
        Provides a pad_collate_fn() method for usage with torch.utils.data.DataLoader() to pad minibatch samples to
        an equal length based on the maximal length in the current minibatch. Each minibatch consists of samples and
        each sample is expected to be a tuple or list. Multiple elements per sample may be padded. Padded elements
        will be replaced by a tuple containing (padded_sequences, original_sequence_lengths).
        
        Parameters
        ----------------
        padding_dims : tuple of integers
            Dimension index to pad for each element in a sample. len(padding_dims) must be smaller or equal to the
            number of elements per sample. None values will not be padded. len(padding_dims) must be equal to
            len(padding_values).
           
        padding_values : tuple of values
            Values to pad with for each element in a sample; len(padding_values) must be smaller or equal to the number
            of elements per sample. len(padding_dims) must be equal to len(padding_values).
        
        Example
        ----------------
        > # Let's assume samples in the form of [ipnut1, input2, labels]
        > # with inputs1 being arrays of shape (seq_length, features)
        > # and inputs2 being arrays of shape (x, y, features, seq_length)
        > # and labels as arrays of shape (n_classes).
        > # This will padd all samples to the same sequence length:
        > # import torch
        > # from widis-lstm-tools.preprocessing import PadToEqualLengths
        > seq_padding = PadToEqualLengths(padding_dims=(0, 2, None), padding_values=(0, 0, None))
        > data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, collate_fn=seq_padding.pad_collate_fn)
        > for data in data_loader:
        >   inputs1, inputs2, labels = data
        >   inputs1_sequences, inputs1_original_sequence_lengths = inputs1
        >   inputs2_sequences, inputs2_original_sequence_lengths = inputs2
        
        Provides
        ----------------
        pad_collate_fn(batch_as_list)
            Function to be passed to torch.utils.data.DataLoader as collate_fn
        
        """
        if len(padding_dims) != len(padding_values):
            raise ValueError("padding_dims and padding_values must be of same length")
        
        self.padding_dims = padding_dims
        self.padding_values = padding_values
    
    def _pad_(self, sequences: list, batch_ind: int):
        if batch_ind >= len(self.padding_dims) or self.padding_dims[batch_ind] is None:
            try:
                return torch.Tensor(np.array(sequences))
            except TypeError:
                return sequences
        else:
            seq_lens = np.array([sequence.shape[self.padding_dims[batch_ind] - 1] for sequence in sequences],
                                dtype=np.int64)
            max_seq_len = np.max(seq_lens)
            new_shape = [len(sequences)] + list(sequences[0].shape)
            new_shape[self.padding_dims[batch_ind]] = max_seq_len
            if self.padding_values[batch_ind] == 0:
                padded_sequence_batch = np.zeros(new_shape, dtype=sequences[0].dtype)
            elif self.padding_values[batch_ind] == 1:
                padded_sequence_batch = np.ones(new_shape, dtype=sequences[0].dtype)
            else:
                padded_sequence_batch = np.full(new_shape, dtype=sequences[0].dtype,
                                                fill_value=self.padding_values[batch_ind])
            for i, sl in enumerate(seq_lens):
                padded_sequence_batch[i, :sl] = sequences[i]
            return torch.Tensor(padded_sequence_batch), torch.Tensor(seq_lens)
    
    def pad_collate_fn(self, batch_as_list: list):
        """Function to be passed to torch.utils.data.DataLoader as collate_fn
        
        Function for usage with torch.utils.data.DataLoader() to pad minibatch samples to
        an equal length based on the maximal length in the current minibatch. Each minibatch consists of samples and
        each sample is expected to be a tuple or list. Multiple elements per sample may be padded. Padded elements
        will be replaced by a tuple containing (padded_sequences, original_sequence_lengths).
        
        Example
        ----------------
        > # Let's assume samples in the form of [ipnut1, input2, labels]
        > # with inputs1 being arrays of shape (seq_length, features)
        > # and inputs2 being arrays of shape (x, y, features, seq_length)
        > # and labels as arrays of shape (n_classes).
        > # This will padd all samples to the same sequence length:
        > # import torch
        > # from widis-lstm-tools.preprocessing import PadToEqualLengths
        > seq_padding = PadToEqualLengths(padding_dims=(0, 2, None), padding_values=(0, 0, None))
        > data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, collate_fn=seq_padding.pad_collate_fn)
        > for data in data_loader:
        >   inputs1, inputs2, labels = data
        >   inputs1_sequences, inputs1_original_sequence_lengths = inputs1
        >   inputs2_sequences, inputs2_original_sequence_lengths = inputs2
        """
        sample_entries_as_list = [sample_entry for sample in batch_as_list for sample_entry in sample]
        padded_batch = [self._pad_(sample_entries_as_list[i::len(batch_as_list[0])], i)
                        for i in range(len(batch_as_list[0]))]

        return padded_batch


class TriangularValueEncoding(object):
    def __init__(self, max_value: int, triangle_span: int, normalize: bool = False):
        """Encodes an integer value with range [0, max_value] as multiple activations between 0 and 1 via triangles of
        width triangle_span;

        LSTM profits from having an integer input with large range split into multiple input nodes; This class encodes
        an integer as multiple nodes with activations of range [0,1]; Each node represents a triangle of width
        triangle_span; These triangles are distributed equidistantly over the integer range such that 2 triangles
        overlap by 1/2 width and the whole integer range is covered; For each integer to encode, the high of the
        triangle at this integer position is taken as node activation, i.e. max. 2 nodes have an activation > 0 for each
        integer value;

        Values are encoded via encode_value(value) and returned as float32 numpy array of length self.n_nodes;

        Parameters
        ----------
        max_value : int
            Maximum value to encode
        triangle_span : int
            Width of each triangle
        normalize : bool
            Normalize encoded values? (default: False)
        """
        if triangle_span % 2:
            raise ValueError("triangle_span for TriangularValueEncoding must be an even number!")
        
        # round max_value up to a multiple of triangle_span
        if max_value % triangle_span != 0:
            max_value = ((max_value // triangle_span) + 1) * triangle_span
        
        # Calculate number of overlapping triangle nodes
        n_nodes_half = int(max_value / triangle_span)
        n_nodes = n_nodes_half * 2 + 1
        
        self.n_nodes = int(n_nodes)
        self.n_nodes_half = n_nodes_half
        self.max_value = int(max_value)
        self.triangle_span = int(triangle_span)
        self.triangle_span_float = float(triangle_span)
        self.half_triangle_span = int(self.triangle_span / 2)
        self.half_triangle_span_float = self.triangle_span / 2.
        self.normalize = normalize
    
    def encode_values(self, values):
        """Encode value as multiple triangle node activations

        Parameters
        ----------
        values : numpy array of int
            Values to encode as array of integers

        Returns
        ----------
        float32 numpy array
            Encoded value as float32 numpy array of length self.n_nodes
        """
        value_float = np.array(values, dtype=np.float)
        encoding = np.zeros((len(values), self.n_nodes), np.float32)
        
        # Encode first half of triangle
        index = np.array(values / self.triangle_span, dtype=np.int)
        act = np.abs(0.5 - (np.mod(np.abs(value_float - self.half_triangle_span_float), self.triangle_span_float)
                            / self.triangle_span_float))
        encoding[np.arange(len(values)), index] = act
        
        # Encode second half of triangle
        index = np.array((values + self.half_triangle_span) / self.triangle_span, dtype=np.int) + self.n_nodes_half
        encoding[np.arange(len(values)), index] = 0.5 - act
        
        # Normalize encoding
        if self.normalize:
            encoding[:] = (encoding * 2) - (1 / self.n_nodes)
        else:
            encoding[:] *= 2
        
        return encoding
