import torch
import torch.nn as nn
import torch.nn.utils.rnn as R
import torch.nn.functional as F
from torch.autograd import Variable # variable is a little different due to version, return tensor instead of Variable
import numpy as np



class PointerNetworks(nn.Module): # pointer mechanism
    def __init__(self,voca_size, voc_embeddings,word_dim, hidden_dim,is_bi_encoder_rnn,rnn_type,rnn_layers,
                 dropout_prob,use_cuda,finedtuning,isbanor):
        super(PointerNetworks,self).__init__()
        self.word_dim = word_dim
        self.voca_size = voca_size

        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.is_bi_encoder_rnn = is_bi_encoder_rnn
        self.num_rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        self.voc_embeddings = voc_embeddings
        self.finedtuning = finedtuning

        self.nnDropout = nn.Dropout(dropout_prob)

        self.isbanor = isbanor

        if rnn_type in ['LSTM', 'GRU']:


            # decoder & encoder
            self.decoder_rnn = getattr(nn, rnn_type)(input_size=word_dim,
                                                     hidden_size=2 * hidden_dim if is_bi_encoder_rnn else hidden_dim, # bi/uni
                                                     num_layers=rnn_layers,
                                                     dropout=dropout_prob,
                                                     batch_first=True)

            self.encoder_rnn = getattr(nn, rnn_type)(input_size=word_dim,
                                       hidden_size=hidden_dim, # uni
                                       num_layers=rnn_layers,
                                       bidirectional=is_bi_encoder_rnn,
                                       dropout=dropout_prob,
                                       batch_first=True)



        else: # check type 
            print('rnn_type should be LSTM,GRU')



        self.nnSELU = nn.SELU() # Scaled Exponential Linear Unit


        self.nnEm = nn.Embedding(self.voca_size,self.word_dim) # nn embeddings

        self.initEmbeddings(self.voc_embeddings) # initial embeddings

        self.use_cuda = use_cuda





        # check bi/uni encoder
        if self.is_bi_encoder_rnn:
            self.num_encoder_bi = 2
        else:
            self.num_encoder_bi = 1

        # applies three linear transformation to the incoming data
        self.nnW1 = nn.Linear(self.num_encoder_bi * hidden_dim, self.num_encoder_bi * hidden_dim, bias=False)
        self.nnW2 = nn.Linear(self.num_encoder_bi * hidden_dim, self.num_encoder_bi * hidden_dim, bias=False)
        self.nnV = nn.Linear(self.num_encoder_bi * hidden_dim, 1, bias=False)




    def initEmbeddings(self,weights):
        self.nnEm.weight.data.copy_(torch.from_numpy(weights))
        self.nnEm.weight.requires_grad = self.finedtuning 


    def initHidden(self,hsize,batchsize): # hidden states for LSTM or GRU 


        if self.rnn_type == 'LSTM':

            h_0 = Variable(torch.zeros(self.num_encoder_bi*self.num_rnn_layers, batchsize, hsize))
            c_0 = Variable(torch.zeros(self.num_encoder_bi*self.num_rnn_layers, batchsize, hsize))

            if self.use_cuda:
                h_0 = h_0.cuda()
                c_0 = c_0.cuda()

            return (h_0, c_0)
        else: # GRU

            h_0 = Variable(torch.zeros(self.num_encoder_bi*self.num_rnn_layers, batchsize, hsize))

            if self.use_cuda:
                h_0 = h_0.cuda()


            return h_0







    def _run_rnn_packed(self, cell, x, x_lens, h=None): # encoder,sequence, length of sequence, whether initial hidden state
        x_packed = R.pack_padded_sequence(x, x_lens.data.tolist(), 
                                          batch_first=True) 

        if h is not None: 
            output, h = cell(x_packed, h)
        else:
            output, h = cell(x_packed)

        output, _ = R.pad_packed_sequence(output, batch_first=True)

        return output, h # tenser containing output features, tensor containing the hidden state





    def pointerEncoder(self,Xin,lens):
        # Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_
        self.bn_inputdata = nn.BatchNorm1d(self.word_dim, affine=False, track_running_stats=False) 


        batch_size,maxL = Xin.size()

        X = self.nnEm(Xin)  # embeddings

        if self.isbanor and maxL>1:
            X= X.permute(0,2,1) 
            X = self.bn_inputdata(X)
            X = X.permute(0, 2, 1)

        X = self.nnDropout(X)



        encoder_lstm_co_h_o = self.initHidden(self.hidden_dim, batch_size)
        o, h = self._run_rnn_packed(self.encoder_rnn, X, lens, encoder_lstm_co_h_o)
        o = o.contiguous() #  it actually makes a copy of the tensor such that the order of its elements in memory is the same as if it had been created from scratch with the same data.

        o = self.nnDropout(o)

        return o,h


    def pointerLayer(self,en,di):
        """

        :param en:  [L,H]
        :param di:  [H,]
        :return:
        """


        WE = self.nnW1(en)


        exdi = di.expand_as(en) # Expand di tensor to the same size as en

        WD = self.nnW2(exdi)

        nnV = self.nnV(self.nnSELU(WE+WD))

        nnV = nnV.permute(1,0)

        nnV = self.nnSELU(nnV)


        att_weights = F.softmax(nnV)
        logits = F.log_softmax(nnV)

        return logits,att_weights







    def training_decoder(self,hn,hend,X,Xindex,Yindex,lens):

        loss_function  = nn.NLLLoss()
        batch_loss =0
        LoopN =0
        batch_size = len(lens)
        for i in range(len(lens)): #Loop batch size

            curX_index = Xindex[i]
            curY_index = Yindex[i]
            curL = lens[i]
            curX = X[i]

            x_index_var = Variable(torch.from_numpy(curX_index.astype(np.int64)))
            if self.use_cuda:
                x_index_var = x_index_var.cuda()

            cur_lookup = curX[x_index_var]

            curX_vectors = self.nnEm(cur_lookup)  # output: [seq,features]

            curX_vectors = curX_vectors.unsqueeze(0)  # [batch, seq, features]



            if self.rnn_type =='LSTM':# need h_end,c_end


                h_end = hend[0].permute(1, 0, 2).contiguous().view(batch_size, self.num_rnn_layers,-1)
                c_end = hend[1].permute(1, 0, 2).contiguous().view(batch_size, self.num_rnn_layers,-1)

                curh0 = h_end[i].unsqueeze(0).permute(1, 0, 2)
                curc0 = c_end[i].unsqueeze(0).permute(1, 0, 2)


                h_pass = (curh0,curc0)
            else:


                h_end = hend.permute(1, 0, 2).contiguous().view(batch_size, self.num_rnn_layers,-1)
                curh0 = h_end[i].unsqueeze(0).permute(1, 0, 2)
                h_pass = curh0



            decoder_out,_ = self.decoder_rnn(curX_vectors,h_pass)
            decoder_out = decoder_out.squeeze(0)   #[seq,features]


            curencoder_hn = hn[i,0:curL,:]  # hn[batch,seq,H] -->[seq,H] i is loop batch size

            for j in range(len(decoder_out)):  #Loop di
                cur_dj = decoder_out[j]
                cur_groundy = curY_index[j]

                cur_start_index = curX_index[j]
                predict_range = list(range(cur_start_index,curL))

                # make it point backward, only consider predict_range in current time step
                # align groundtruth
                cur_groundy_var = Variable(torch.LongTensor([int(cur_groundy) - int(cur_start_index)]))
                if self.use_cuda:
                    cur_groundy_var = cur_groundy_var.cuda()

                curencoder_hn_back = curencoder_hn[predict_range,:]




                cur_logists, cur_weights = self.pointerLayer(curencoder_hn_back,cur_dj)

                batch_loss = batch_loss + loss_function(cur_logists,cur_groundy_var)
                LoopN = LoopN +1

        batch_loss = batch_loss/LoopN

        return batch_loss


    def neg_log_likelihood(self,Xin,index_decoder_x, index_decoder_y,lens):

        encoder_hn, encoder_h_end = self.pointerEncoder(Xin,lens)

        loss = self.training_decoder(encoder_hn, encoder_h_end,Xin,index_decoder_x, index_decoder_y,lens)

        return loss




    def test_decoder(self,hn,hend,X,Yindex,lens):

        loss_function = nn.NLLLoss() # The negative log likelihood loss
        batch_loss = 0
        LoopN = 0

        batch_boundary =[]
        batch_boundary_start =[]
        batch_align_matrix =[]

        batch_size = len(lens)

        for i in range(len(lens)):  # Loop batch size



            curL = lens[i]
            curY_index = Yindex[i]
            curX = X[i]
            cur_end_boundary =curY_index[-1]

            cur_boundary = []
            cur_b_start = []
            cur_align_matrix = []

            cur_sentence_vectors = self.nnEm(curX)  # output: [seq,features]


            if self.rnn_type =='LSTM':# need h_end,c_end


                h_end = hend[0].permute(1, 0, 2).contiguous().view(batch_size, self.num_rnn_layers,-1)
                c_end = hend[1].permute(1, 0, 2).contiguous().view(batch_size, self.num_rnn_layers,-1)

                curh0 = h_end[i].unsqueeze(0).permute(1, 0, 2)
                curc0 = c_end[i].unsqueeze(0).permute(1, 0, 2)

                h_pass = (curh0,curc0)
            else: # GRU: only need h_end


                h_end = hend.permute(1, 0, 2).contiguous().view(batch_size, self.num_rnn_layers,-1)
                curh0 = h_end[i].unsqueeze(0).permute(1, 0, 2)
                h_pass = curh0



            curencoder_hn = hn[i, 0:curL, :]  # hn[batch,seq,H] --> [seq,H]  i is loop batch size

            Not_break = True

            loop_in = cur_sentence_vectors[0,:].unsqueeze(0).unsqueeze(0)  #[1,1,H]
            loop_hc = h_pass


            loopstart =0

            loop_j =0
            while (Not_break): #if not end

                loop_o, loop_hc = self.decoder_rnn(loop_in,loop_hc)


                # make it point backward

                predict_range = list(range(loopstart,curL))
                curencoder_hn_back = curencoder_hn[predict_range,:]
                cur_logists, cur_weights = self.pointerLayer(curencoder_hn_back, loop_o.squeeze(0).squeeze(0))

                cur_align_vector = np.zeros(curL)
                cur_align_vector[predict_range]=cur_weights.data.cpu().numpy()[0]
                cur_align_matrix.append(cur_align_vector)

                # align groundtruth
                if loop_j > len(curY_index)-1:
                    cur_groundy = curY_index[-1]
                else:
                    cur_groundy = curY_index[loop_j]


                cur_groundy_var = Variable(torch.LongTensor([max(0,int(cur_groundy) - loopstart)]))
                if self.use_cuda:
                    cur_groundy_var = cur_groundy_var.cuda()

                batch_loss = batch_loss + loss_function(cur_logists, cur_groundy_var)


                #get predicted boundary
                topv, topi = cur_logists.data.topk(1)

                pred_index = topi[0][0]


                # align pred_index to original seq
                ori_pred_index =pred_index + loopstart


                if cur_end_boundary == ori_pred_index: # if end
                    cur_boundary.append(ori_pred_index)
                    cur_b_start.append(loopstart)
                    Not_break = False
                    loop_j = loop_j + 1
                    LoopN = LoopN + 1
                    break
                else: # not end
                    cur_boundary.append(ori_pred_index)

                    loop_in = cur_sentence_vectors[ori_pred_index+1,:].unsqueeze(0).unsqueeze(0)
                    cur_b_start.append(loopstart)

                    loopstart = ori_pred_index+1  # start =  pred_end + 1

                    loop_j = loop_j + 1
                    LoopN = LoopN + 1


            #For each instance in batch
            batch_boundary.append(cur_boundary)
            batch_boundary_start.append(cur_b_start)
            batch_align_matrix.append(cur_align_matrix)

        batch_loss = batch_loss / LoopN

        batch_boundary=np.array(batch_boundary)
        batch_boundary_start = np.array(batch_boundary_start)
        batch_align_matrix = np.array(batch_align_matrix)

        return batch_loss,batch_boundary,batch_boundary_start,batch_align_matrix








    def predict(self,Xin,index_decoder_y,lens):

        batch_size = index_decoder_y.shape[0]

        encoder_hn, encoder_h_end = self.pointerEncoder(Xin, lens)





        batch_loss, batch_boundary, batch_boundary_start, batch_align_matrix = self.test_decoder(encoder_hn,encoder_h_end,Xin,index_decoder_y,lens)

        return  batch_loss,batch_boundary,batch_boundary_start,batch_align_matrix





















