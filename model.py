import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from torch.nn import init
from torch.nn.utils import rnn
import random

class EmbeddingLayer(nn.Module):
    """ Embedding Layer

    Attributes:
        word_embedding: use Glove pretrained embedding vectors
        nGram_embeding: use CharNGram pretrained embedding vectors
    TODO:
        add Char CNN embedding
        add Bert Embedding
    """

    def __init__(
        self, 
        word_vectors,
        num_embeddings:int,
        embed_dim: int,
        out_channels: int,
        kernel_size: int
        ):
        super(EmbeddingLayer, self).__init__()

        self.word_embedding = nn.Embedding.from_pretrained(word_vectors, freeze=True)
        self.char_embedding = CharacterEmbedding(num_embeddings, embed_dim, out_channels, kernel_size)

    def forward(self, glove_input, char_input):
        '''
        Arguments:
            glove_input: shape of [batch,  n_word]
            charNgram_input: shape of [batch, n_word, n_char]
        return:
            embedding: shape of [batch, n_word, glove_dim + charNgram_dim]
        '''
        g_emb = self.word_embedding(glove_input)
        c_emb = self.char_embedding(char_input)
        out = torch.cat([g_emb, c_emb], dim=-1)
        return out

class CharacterEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings:int,
        embed_dim: int,
        out_channels: int,
        kernel_size: int
    ):
        super(CharacterEmbedding, self).__init__()
        self.char_embed = nn.Embedding(num_embeddings, embed_dim)
        self.conv = nn.Conv1d(embed_dim, out_channels, kernel_size, padding=kernel_size//2)

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        batch_size, max_sent_length, max_word_length = tuple(chars.size())
        chars = chars.view(batch_size * max_sent_length, max_word_length)

        # char_embedding: (bsize * max_sent_length, max_word_length, embed_dim)
        char_embedding = self.char_embed(chars)

        # conv_inp dim: (bsize * max_sent_length, emb_size, max_word_length)
        conv_inp = char_embedding.transpose(1, 2)
        char_conv_out = torch.relu(self.conv(conv_inp))

        # Apply max pooling
        # char_pool_out dims: (bsize * max_sent_length, out_channels)
        char_pool_out = torch.max(char_conv_out, dim=2)[0]

        return char_pool_out.view(batch_size, max_sent_length, -1)

class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x

class EncoderRNN(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)
        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()
        for i in range(self.nlayers):
            hidden = self.get_init(bsz, i)
            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)
            output, hidden = self.rnns[i](output, hidden)
            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen: # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]

class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask=None):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot

        if mask is not None:
            att = att - 1e30 * (1 - mask[:,None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)

class GateLayer(nn.Module):
    def __init__(self, d_input, d_output):
        super(GateLayer, self).__init__()
        self.linear = nn.Linear(d_input, d_output)
        self.gate = nn.Linear(d_input, d_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        return self.linear(input) * self.sigmoid(self.gate(input))


class SelfAttention(nn.Module):

    def __init__(self, n_input: int, attn_dimension: int = 64, dropout: float = 0.4) -> None:
        super().__init__()

        self.dropout = LockedDropout(dropout)
        self.n_input = n_input
        self.n_attn = attn_dimension
        self.ws1 = nn.Linear(n_input, self.n_attn, bias=False)
        self.ws2 = nn.Linear(self.n_attn, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.init_weights()

    def init_weights(self, init_range: float = 0.1) -> None:
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(
        self, inputs: torch.Tensor
    ) -> torch.Tensor:
        # size: (bsz, sent_len, rep_dim)
        size = inputs.size()
        inputs = self.dropout(inputs)
        compressed_emb = inputs.contiguous().view(-1,size[-1])
        hbar = self.tanh(
            self.ws1(compressed_emb)
        )  # (bsz * sent_len, attention_dim)
        alphas = self.ws2(hbar)  # (bsz * sent_len, 1)
        alphas = alphas.view(size[:2]) # (bsz, sent_len)
        alphas = self.softmax(alphas)  # (bsz, sent_len)
        # (bsz, rep_dim)
        return torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)


class CoAttention(nn.Module):

    def __init__(self, hidden_dims, att_type=0, dropout=0.2):
        super(CoAttention, self).__init__()
        self.dropout = LockedDropout(dropout)
        self.G = nn.Linear(hidden_dims, hidden_dims, bias=True)
        self.att_type = att_type
    def forward(self, a1, a2):
        '''
            a1: n * L * d
            a2: n * K * d
        return:
            M1: n* L * d
            M2: n * K * d

        '''
        a1 = self.dropout(a1)
        a2 = self.dropout(a2)

        if self.att_type == 0:
            a2_ = self.G(a2)
            L = torch.bmm(a1, a2_.permute(0, 2, 1))
        elif self.att_type == 1:
            a1_ = F.relu(self.G(a1))
            a2_ = F.relu(self.G(a2))
            L = torch.bmm(a1_, a2_.permute(0, 2, 1))
        else:
            L = torch.bmm(a1, a2.permute(0,2,1))

        A1 = torch.softmax(L, 2) # N, L , K
        A2 = torch.softmax(L, 1)
        A2 = A2.permute(0,2,1) # N, K, L

        M_1 = torch.bmm(A1, a2) # N, L, d
        M_2 = torch.bmm(A2, a1) # N, K, d

        if self.att_type == 2:
            M_3 = torch.bmm(A1, M_2)
            M_1 = torch.cat([M_1,M_3], dim=-1)

        return M_1, M_2

class FusionLayer(nn.Module):

    def __init__(self, dim=100, dropout=0.2):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.linear = nn.Linear(dim*2, dim, bias=True)
        self.act = nn.ReLU()

    def forward(self, a1, a2):
        assert a1.size() == a2.size()

        mid = torch.cat([a1 - a2, a1 * a2], -1)
        return self.act(self.linear(self.dropout(mid)))

class PoolingLayer(nn.Module):
    '''
    pooling operation: max pooling and attentive pooling
    '''

    def __init__(self, max_pooling=True, dim=None, dropout=0.2):
        super(PoolingLayer, self).__init__()

        self.max_pooling = max_pooling
        if not max_pooling:
            self.dropout = LockedDropout(dropout)
            self.linear = nn.Linear(dim, 1, bias=True)

    def forward(self, x):
        '''
        x: n * L * d
        '''
        if self.max_pooling:
            out, _ = torch.max(x, 1)
            return out
        else:
            score = self.linear(self.dropout(x)) # n * L * 1
            alpha = torch.softmax(score, -1)
            out = torch.bmm(x.permute(0,2,1), alpha) # n * d * 1
            out = out.squeeze()
            return out


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = x_i'Wy for x_i in X.
    """
    def __init__(self, x_size, y_size, identity=False, dropout=0.2):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y):
        """
        x = batch * len * h1
        y = batch * h2
        """
        Wy = self.linear(y) if self.linear is not None else y  # batch * h1
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)  # batch * len
        return xWy

def generate_mask(x_size, num_turn, dropout_p=0.0, is_training=False):
    if not is_training: dropout_p = 0.0
    new_data = torch.zeros(x_size, num_turn)
    new_data = (1-dropout_p) * (new_data.zero_() + 1)
    for i in range(new_data.size(0)):
        one = random.randint(0, new_data.size(1)-1)
        new_data[i][one] = 1
    mask = 1.0/(1 - dropout_p) * torch.bernoulli(new_data)
    mask.requires_grad = False
    return mask

class SAN(nn.Module):
    def __init__(self, question_dim, support_dim, candidate_dim, num_turn=5, dropout=0.2, memo_dropout=0.4, memory_type=0,
                 device=None, san_type=1):
        '''
        san_type:
            0: baseline, do not use multi-step reanson
            1: first use self-att summary paragraph word embedding, then use bilinear-att to read value from the paragraph leval summary embedding.
            2. concate paragraph word embedding, and then use read opeartion
            3. use read operation twice
        '''
        super(SAN,self).__init__()
        self.san_type = san_type
        if san_type == 0:
            self.word_self_att = SelfAttention(support_dim, support_dim, dropout=dropout)
            self.para_self_att = SelfAttention(support_dim, support_dim, dropout=dropout)
        elif san_type == 1:
            self.word_self_att = SelfAttention(support_dim, support_dim, dropout=dropout)
            self.para_bilinear_att = BilinearSeqAttn(support_dim, question_dim, dropout=dropout)
        elif san_type == 2:
            self.word_bilinear_att = BilinearSeqAttn(support_dim, question_dim, dropout=dropout)
        elif san_type == 3:
            self.word_bilinear_att = BilinearSeqAttn(support_dim, question_dim, dropout=dropout)
            self.para_bilinear_att = BilinearSeqAttn(support_dim, question_dim, dropout=dropout)
        if san_type == 0:
            self.candidates_scorer = BilinearSeqAttn(candidate_dim, question_dim, dropout=dropout)
        else:
            self.candidates_scorer = BilinearSeqAttn(candidate_dim, support_dim, dropout=dropout)

        self.gru = nn.GRUCell(support_dim, question_dim)
        self.num_turn = num_turn
        self.dropout = nn.Dropout(p=dropout)
        self.memo_dropout=memo_dropout
        self.device = device
        self.memory_type = memory_type

    def forward(self, question_embedding, para_embedding, candidates_embedding, para_num):
        '''
        input:
            question_embedding: [batch_size, hidden_dim]
            para_embedding: [batch_size*para_num,para_length , hidden_dim]
            candidates_embedding: [batch_size, candidates_num, hidden_dim]

        '''
        batch_size = question_embedding.size(0)
        hidden = question_embedding.size(1)
        if self.san_type == 0:
            para_embedding_summary = self.word_self_att(para_embedding)
            para_embedding_summary = para_embedding_summary.contiguous().view(batch_size, para_num, hidden)
            context_summary = self.para_self_att(para_embedding_summary)
            candidates_score = self.candidates_scorer(candidates_embedding, context_summary)
            return candidates_score
        # multi-step reasoning 
        score_list = []
        if self.san_type == 1:
            para_embedding_summary = self.word_self_att(para_embedding)
            para_embedding_summary = para_embedding_summary.contiguous().view(batch_size, para_num, hidden)
            for turn in range(self.num_turn):
                # update question embedding
                qp_score_para = self.para_bilinear_att(para_embedding_summary, question_embedding)
                qp_score_para = F.softmax(qp_score_para, 1)
                S = torch.bmm(qp_score_para.unsqueeze(1), para_embedding_summary).squeeze(1)
                S = self.dropout(S)
                question_embedding = self.gru(S, question_embedding)
                # compute candidates score
                candidates_score = self.candidates_scorer(candidates_embedding, question_embedding)
                score_list.append(candidates_score)
        elif self.san_type == 2:
            for turn in range(self.num_turn):
                para_embedding = para_embedding.view(batch_size, para_num, -1, hidden)
                para_embedding = para_embedding.view(batch_size, -1, hidden)
                qp_score_para = self.word_bilinear_att(para_embedding, question_embedding)
                qp_score_para = F.softmax(qp_score_para, 1)
                S = torch.bmm(qp_score_para.unsqueeze(1), para_embedding).squeeze(1)
                S = self.dropout(S)
                question_embedding = self.gru(S, question_embedding)
                # compute candidates score
                candidates_score = self.candidates_scorer(candidates_embedding, question_embedding)
                score_list.append(candidates_score)   
        else:
            for turn in range(self.num_turn):
                # update paragraph embedding
                question_embedding_expand = question_embedding.unsqueeze(1).expand(batch_size, para_num, hidden).contiguous()
                question_embedding_expand = question_embedding_expand.view(-1,hidden)
                qp_score_word = self.word_bilinear_att(para_embedding, question_embedding_expand)
                qp_score_word = F.softmax(qp_score_word, 1)
                para_embedding_summary = torch.bmm(qp_score_word.unsqueeze(1), para_embedding).squeeze(1)
                para_embedding_summary = para_embedding_summary.contiguous().view(batch_size, para_num, hidden)
                # update query embedding
                qp_score_para = self.para_bilinear_att(para_embedding_summary, question_embedding)
                qp_score_para = F.softmax(qp_score_para, 1)
                S = torch.bmm(qp_score_para.unsqueeze(1), para_embedding_summary).squeeze(1)
                S = self.dropout(S)
                question_embedding = self.gru(S, question_embedding)
                # compute candidates score
                candidates_score = self.candidates_scorer(candidates_embedding, question_embedding)
                score_list.append(candidates_score)                    

        # Agg scores
        if self.memory_type == 0:
            mask = generate_mask(batch_size,self.num_turn, self.memo_dropout, self.training)
            mask = mask.to(self.device)
            mask = [m.contiguous() for m in torch.unbind(mask, 1)]

            score_list = [mask[idx].view(batch_size, 1).expand_as(inp) * inp for idx, inp in enumerate(score_list)]
            scores = torch.stack(score_list, 2)
            scores = torch.mean(scores, 2)
        elif self.memory_type == 1:
            scores = torch.stack(score_list, 2)
            scores = torch.mean(scores, 2)
        elif self.memory_type == 2:
            scores = score_list[-1]

        return scores

class SimpleQANet(nn.Module):

    def __init__(self, config, word_vectors, num_embeddings,  device):
        super(SimpleQANet, self).__init__()
        self.config = config
        self.device = device

        self.embedding_layer = EmbeddingLayer(word_vectors, num_embeddings, config.embed_dim, config.out_channels, config.kernel_size)
        self.rnn = EncoderRNN(config.embedding_dim, config.hidden, 1, True, True, config.dropout, False)
        self.query_supports_attention = CoAttention(config.hidden*2, att_type=2, dropout=config.dropout)
        self.linear_1 = nn.Sequential(
                        nn.Linear(config.hidden*4, config.hidden),
                        nn.ReLU()
                    )
        self.rnn2 =  EncoderRNN(config.hidden, config.hidden, 1, True, True, config.dropout, False)
        self.query_selfatt = SelfAttention(config.hidden*2, config.hidden*2, config.dropout)
        self.candidates_selfatt = SelfAttention(config.hidden*2, config.hidden*2, config.dropout)
        self.san = SAN(config.hidden*2,config.hidden*2,config.hidden*6, num_turn=config.steps, memory_type=config.memory_type, device=device, san_type=config.san_type)
        self.to(device)

    def get_candidate_vectors(self, batch, support_vectors, device):
        batch_size, candidate_num,_ = batch.candidates.shape
        _,support_num, support_length = batch.supports.shape
        hidden = support_vectors.shape[-1]

        masks = []
        for idx, candidate_mentions in enumerate(batch.mentions):
            mask = torch.zeros(candidate_num, support_num, support_length)
            for i in range(len(candidate_mentions)):
                candidate_mention = candidate_mentions[i]
                for mention in candidate_mention:
                    mask[i][mention[0]][mention[1]:mention[2]] = 1
            masks.append(mask)
        masks = torch.stack(masks).to(device)

        support_vectors = support_vectors.view(batch_size,-1,hidden).unsqueeze(1)

        masks = masks.view(batch_size,candidate_num,-1)
        masks_expand = masks.unsqueeze(-1).expand(batch_size, candidate_num, support_length*support_num, hidden)

        candidates = support_vectors * masks_expand

        candidates_max = candidates.max(-2)[0]
        candidates_mean = torch.sum(candidates,-2) / (torch.sum(masks_expand, -2) + 0.01)
        candidates_vectors = torch.cat([candidates_max, candidates_mean],-1)
        #return candidates_mean
        return candidates_vectors

    def forward(self, batch, return_label = True):
        query = batch.query
        supports = batch.supports
        candidates = batch.candidates

        candidates_mask = ((candidates > 1).sum(-1) > 0).float()


    
        query_char = batch.query_char
        supports_char = batch.supports_char
        candidates_char = batch.candidates_char

        #print(candidates.shape, candidates_char.shape, supports.shape, supports_char.shape, query.shape, query_char.shape)

        batch_size, support_num, support_len = supports.shape
        batch_size, candidate_num, candidate_len = candidates.shape
        batch_size, query_len = query.shape

        supports = supports.view(batch_size*support_num, support_len)
        candidates = candidates.view(batch_size*candidate_num, candidate_len)        

        supports_char = supports_char.view(batch_size*support_num, support_len,-1)
        candidates_char  = candidates_char .view(batch_size*candidate_num, candidate_len,-1)

        #print(candidates.shape, candidates_char.shape, supports.shape, supports_char.shape, query.shape, query_char.shape)


        q_out = self.embedding_layer(query, query_char) # [batch_size,qeustion_length, hidden_dim]
        s_out = self.embedding_layer(supports, supports_char) # [batch_szie * support_num, support_length, hidden_dim]
        c_out = self.embedding_layer(candidates, candidates_char) # [batch_size * candidates_num, candidates_length, hidden_dim]

        q_out = self.rnn(q_out) # [batch_size,qeustion_length, hidden_dim]
        c_out = self.rnn(c_out) # [batch_szie * support_num, support_length, hidden_dim]
        s_out = self.rnn(s_out) # [batch_size * candidates_num, candidates_length, hidden_dim]

        # Attention

        q_out_expand = q_out.unsqueeze(1).expand(batch_size, support_num, query_len, q_out.size(-1)).contiguous()
        q_out_expand = q_out_expand.view(batch_size*support_num, query_len, q_out.size(-1)).contiguous()

        support_att_emb, q_out_att = self.query_supports_attention(s_out, q_out_expand)
        #S_s = self.fusion(s_out, s_out_att)
        #S_q = self.fusion(q_out, q_out_att)

        support_att_emb = self.linear_1(support_att_emb)
        support_att_emb = self.rnn2(support_att_emb) # [batch_size * para_num, para_length, hidden*2]

        # Query Self-Attention
        question_summary = self.query_selfatt(q_out)

        # Candidates Self-Attention and Mention Aggreation
        candidates_summary = self.candidates_selfatt(c_out)
        candidates_summary = candidates_summary.view(batch_size, candidate_num, -1)

        candidates_vectors = self.get_candidate_vectors(batch, support_att_emb, self.device)
        candidates_summary = torch.cat([candidates_summary, candidates_vectors],-1)

        score = self.san(question_summary, support_att_emb, candidates_summary, support_num)
        score = score * candidates_mask + (-1e15)*(1-candidates_mask)

        if return_label:
            label = batch.label
            return score, label
        return score