# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAttn(nn.Module):
    """Global attention.

    For global attention, the attention weights are derived by comparing the
    current target hidden state with each source hidden state.

    Parameters
    ----------
    method : str
        Way to calculate attention weights. Choices are 'dot', 'general', or
        'concat'.
    encoder_hidden_dim : int
        Dimension of encoder's hidden state.
    decoder_hidden_dim : int
        Dimension of decoder's hidden state.

    """

    def __init__(self, method, encoder_hidden_dim, decoder_hidden_dim):
        super(GlobalAttn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method,
                             'is not an appropriate attention method')
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        if self.method == 'concat':  # 'concat' score function
            self.attn = nn.Sequential(
                    nn.Linear(decoder_hidden_dim + encoder_hidden_dim,
                              encoder_hidden_dim),
                    nn.Tanh())
            self.v = nn.Parameter(torch.tensor(encoder_hidden_dim,
                                               dtype=torch.float))
        elif self.method == 'general':
            self.attn = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)

    def dot_score(self, encoder_outputs, decoder_current_hidden):
        """Dot score.

        Parameters
        ----------
        encoder_outputs : :class:`torch.tensor`
            All outputs of the encoder, tensor of shape
            (seq_len, batch, encoder_hidden dim).
        decoder_current_hidden : :class:`torch.tensor`
            Decoder's current hidden state, tensor of shape
            (1, batch, decoder_hidden_dim).

        Returns
        -------
        attn_energies : `torch.tensor`
            Attention energies, tensor of shape (seq_len, batch).


        """

        assert(encoder_outputs.size(2) == decoder_current_hidden.size(2))
        attn_energies = torch.sum(encoder_outputs * decoder_current_hidden,
                                  dim=2)
        return attn_energies

    def general_score(self, encoder_outputs, decoder_current_hidden):
        """General score.

        Parameters
        ----------
        encoder_outputs : :class:`torch.tensor`
            All outputs of the encoder, tensor of shape
            (seq_len, batch, encoder_hidden dim).
        decoder_current_hidden : :class:`torch.tensor`
            Decoder's current hidden state, tensor of shape
            (1, batch, decoder_hidden_dim).

        Returns
        -------
        attn_energies : `torch.tensor`
            Attention energies, tensor of shape (seq_len, batch).

        """

        # (seq_len, batch, decoder_hidden_dim)
        energy = self.attn(encoder_outputs)
        # (seq_len, batch)
        attn_energies = torch.sum(decoder_current_hidden * energy, dim=2)
        return attn_energies

    def concat_score(self, encoder_outputs, decoder_current_hidden):
        """Concat score

        Parameters
        ----------
        encoder_outputs : :class:`torch.tensor`
            All outputs of the encoder, tensor of shape
            (seq_len, batch, encoder_hidden_dim)
        decoder_current_hidden : :class:`torch.tensor`
            Decoder's current hidden state, tensor of shape
            (1, batch, decoder_hidden_dim).

        Returns
        -------
        attn_energies : `torch.tensor`
            Attention energies, tensor of shape (seq_len, batch).

        """

        # (seq_len, batch, encoder_hidden_dim)
        decoder_output = decoder_current_hidden.expand(
                encoder_outputs.size(0), -1, -1)
        # concat the encoder hidden state and decoder hidden state
        hidden_cat = torch.cat((decoder_output, encoder_outputs), dim=2)
        # 'concat' score function (seq_len, batch, encoder_hidden_dim)
        energy = self.attn(hidden_cat)
        # (seq_len, batch)
        attn_energies = torch.sum(self.v * energy, dim=2)
        return attn_energies

    def forward(self, encoder_outputs, decoder_current_hidden):
        """Return attention weights.

        Parameters
        ----------
        encoder_outputs : `torch.tensor`
            All outputs of the encoder, tensor of shape (seq_len, batch,
            encoder_hidden_dim).
        decoder_current_hidden : `torch.tensor`
            Input hidden state, tensor of shape (1, batch, decoder_hidden_dim).

        Returns
        -------
        attn_weights : `torch.Tensor`
            Tensor of shape (batch, 1, seq_len).
            attn_weights[i, 0, :].sum() = 1.

        """

        if self.method == 'concat':
            attn_energies = self.concat_score(encoder_outputs,
                                              decoder_current_hidden).t()
        elif self.method == 'dot':
            attn_energies = self.dot_score(encoder_outputs,
                                           decoder_current_hidden).t()
        elif self.method == 'general':
            attn_energies = self.general_score(encoder_outputs,
                                               decoder_current_hidden).t()
        # softmax along seq_len (batch, 1, seq_len)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module.

    Parameters
    ----------
    d_model : int
        embed_dim.
    d_inner : int
        dff.
    dropout : float
        dropout rate.

    """

    def __init__(self, d, d_inner):
        super().__init__()
        self.w_1 = nn.Conv1d(d, d_inner, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_inner, d, 1)  # position-wise

    def forward(self, x):
        """

        Parameters
        ----------
        x : `torch.Tensor`
            Tensor of shape (batch, len, embed_dim)

        """

        output = x.transpose(1, 2)  # (batch, embed_dim (channel), len)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        return output
