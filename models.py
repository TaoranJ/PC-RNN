# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from layers import GlobalAttn


class Encoder(nn.Module):
    """Encoder.

    Parameters
    ----------
    num_categories: int
        Number of num_categories to encoder.
    embed_dim : int
        Dimension of the embedding layer.
    p_encoder_hidden_dim : int
        Dimension of hidden state of the encoder for patent time series.
    o_encoder_hidden_dim : int
        Dimension of hidden state of the encoder for the inventor/assignee time
        series.

    """

    def __init__(self, num_categories, embed_dim, p_encoder_hidden_dim,
                 o_encoder_hidden_dim):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(num_categories, embed_dim, padding_idx=0)
        p_encoder_input_dim = 1 + embed_dim
        self.p_encoder_hidden_dim = p_encoder_hidden_dim
        self.p_encoder = nn.LSTM(p_encoder_input_dim,
                                 self.p_encoder_hidden_dim,
                                 num_layers=2, bidirectional=True)
        self.o_encoder_hidden_dim = o_encoder_hidden_dim
        self.a_encoder = nn.LSTM(1, self.o_encoder_hidden_dim,
                                 num_layers=2, bidirectional=True)
        self.i_encoder = nn.LSTM(1, self.o_encoder_hidden_dim,
                                 num_layers=2, bidirectional=True)

    def forward(self, patent_src, assignee, inventor):
        """Forward propagation.

        Parameters
        ----------
        patent_src : dict
            patent_src['src_pts'] is the padded patent time sequences of shape
            (seq_len, batch). patent_src['src_pcat'] is the padded patent
            category sequences of shape (seq_len, batch). patent_src['length']
            is used to pack padded sequence.
        assignee : dict
            assignee['ts'] is the padded assignee time sequences of shape
            (seq_len, batch). assignee['length'] is used to pack padded
            sequence.
        inventor : dict
            inventor['ts'] is the padded assignee time sequences of shape
            (seq_len, batch). inventor['length'] is used to pack padded
            sequence.

        """

        p_ts, p_cat = patent_src['pts'], patent_src['pcat']
        length = patent_src['length']
        a_ts, a_length, = assignee['ts'], assignee['length']
        a_org_idx = assignee['org_idx']
        i_ts, i_length, = inventor['ts'], inventor['length']
        i_org_idx = inventor['org_idx']
        # Encoder 1
        p_inputs = torch.cat((p_ts.unsqueeze(-1), self.embed(p_cat)), dim=2)
        p_inputs = pack_padded_sequence(p_inputs, length)
        p_outputs, (p_hn, p_hc) = self.p_encoder(p_inputs)
        p_outputs, _ = pad_packed_sequence(p_outputs)
        p_outputs = p_outputs[:, :, :self.p_encoder_hidden_dim] +\
            p_outputs[:, :, self.p_encoder_hidden_dim:]
        # Encoder 2
        a_inputs = pack_padded_sequence(a_ts.unsqueeze(-1), a_length)
        a_outputs, _ = self.a_encoder(a_inputs)
        a_outputs, _ = pad_packed_sequence(a_outputs)
        a_outputs = a_outputs[:, :, :self.o_encoder_hidden_dim] \
            + a_outputs[:, :, self.o_encoder_hidden_dim:]
        a_outputs = a_outputs[:, a_org_idx, :]
        # Encoder 3:
        i_inputs = pack_padded_sequence(i_ts.unsqueeze(-1), i_length)
        i_outputs, _ = self.i_encoder(i_inputs)
        i_outputs, _ = pad_packed_sequence(i_outputs)
        i_outputs = i_outputs[:, :, :self.o_encoder_hidden_dim] \
            + i_outputs[:, :, self.o_encoder_hidden_dim:]
        i_outputs = i_outputs[:, i_org_idx, :]
        return (p_outputs, p_hn[:2], p_hc[:2]), a_outputs, i_outputs


class Decoder(nn.Module):
    """Decoder.

    Parameters
    ----------
    num_categories: int
        Number of num_categories to encoder.
    embed_dim : int
        Dimension of the embedding layer.
    p_encoder_hidden_dim : int
        Dimension of hidden state of the encoder for patent time series.
    o_encoder_hidden_dim : int
        Dimension of hidden state of the encoder for assitnee/inventor time
        sereis.
    p_decoder_hidden_dim : int
        Dimension of hidden state of the decoder for patent time series.
    p_decoder_inner_dim : int
        Dimension of innter state of the decoder for patent time series.

    """

    def __init__(self, num_categories, embed_dim, p_encoder_hidden_dim,
                 o_encoder_hidden_dim, p_decoder_hidden_dim,
                 p_decoder_inner_dim):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(num_categories, embed_dim, padding_idx=0)
        p_decoder_input_dim = 1 + embed_dim
        self.p_decoder = nn.LSTM(p_decoder_input_dim, p_decoder_hidden_dim,
                                 num_layers=2)
        # 1st attention layer
        self.p_attn = GlobalAttn(p_encoder_hidden_dim, p_decoder_hidden_dim)
        self.a_attn = GlobalAttn(o_encoder_hidden_dim, p_decoder_hidden_dim)
        self.i_attn = GlobalAttn(o_encoder_hidden_dim, p_decoder_hidden_dim)
        # 2nd attention layer
        self.p2o = nn.Sequential(
                nn.Linear(p_encoder_hidden_dim, o_encoder_hidden_dim),
                nn.ReLU())
        self.attn = GlobalAttn(o_encoder_hidden_dim, p_decoder_hidden_dim)
        self.o2p = nn.Sequential(
                nn.Linear(o_encoder_hidden_dim, p_decoder_hidden_dim),
                nn.ReLU())
        context_dim = p_decoder_hidden_dim
        # Output
        self.out = nn.Sequential(
            nn.Linear(p_decoder_hidden_dim + context_dim, p_decoder_inner_dim),
            nn.ReLU(),
            nn.Linear(p_decoder_inner_dim, p_decoder_hidden_dim),
            nn.ReLU())
        self.marker_gen = nn.Sequential(
            nn.Linear(p_decoder_hidden_dim, self.embed.num_embeddings),
            nn.LogSoftmax(dim=2))
        self.time_gen = nn.Sequential(
                nn.Linear(p_decoder_hidden_dim, 1), nn.ReLU())

    def forward(self, p_ts, p_cat, p_hn, p_hc, p_encoder_outputs,
                a_encoder_outputs, i_encoder_outputs):
        """Decoder's forward propagation."""

        p_input = torch.cat((p_ts.unsqueeze(-1), self.embed(p_cat)), dim=2)
        p_output, (p_hn, p_hc) = self.p_decoder(p_input, (p_hn, p_hc))
        p_weights = self.p_attn(p_encoder_outputs, p_output)
        p_context = p_weights.bmm(p_encoder_outputs.transpose(0, 1))
        a_weights = self.a_attn(a_encoder_outputs, p_output)
        a_context = a_weights.bmm(a_encoder_outputs.transpose(0, 1))
        i_weights = self.i_attn(i_encoder_outputs, p_output)
        i_context = i_weights.bmm(i_encoder_outputs.transpose(0, 1))
        encoder_combined = torch.cat((self.p2o(p_context.transpose(0, 1)),
                                      a_context.transpose(0, 1),
                                      i_context.transpose(0, 1)), dim=0)
        weights = self.attn(encoder_combined, p_output)
        context = weights.bmm(encoder_combined.transpose(0, 1))
        context = self.o2p(context.transpose(0, 1))
        output = self.out(torch.cat((context, p_output), dim=2))
        ts = self.time_gen(output)
        cat = self.marker_gen(output)
        return ts, cat, p_hn, p_hc


class PCRNN(nn.Module):
    """PCRNN."""

    def __init__(self, encoder, decoder):
        super(PCRNN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, patent_src, assignee, inventor, patent_tgt):
        """Forward propagation"""

        pencoder, aencoder, iencoder = self.encoder(patent_src, assignee,
                                                    inventor)

        tgt_ts_output, tgt_cat_output = [], []
        ts = patent_src['pts'][-1].unsqueeze(0)
        cat = patent_src['pcat'][-1].unsqueeze(0)
        pencoder_outputs, phn, phc = pencoder
        for t, _ in enumerate(patent_tgt['pts']):
            ts, cat, phn, phc = self.decoder(ts, cat, phn, phc,
                                             pencoder_outputs,
                                             aencoder, iencoder)
            tgt_ts_output.append(ts)
            tgt_cat_output.append(cat)
            ts, cat = ts.squeeze(-1), cat.topk(1)[1].squeeze(-1)
        tgt_ts_output = torch.cat(tgt_ts_output, dim=0)
        tgt_cat_output = torch.cat(tgt_cat_output, dim=0)
        return tgt_ts_output, tgt_cat_output
