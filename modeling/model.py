import math
import torch.nn as nn
import torch

from modeling.dataloader import temporal_dataloader
from modeling.metrics import forecasting_metrics, gather_score_dict
from modeling.grud import GRUD
from modeling.custom_dataset import ConcatDataset
from modeling.datasaver import DataSaver
from modeling.encoder_layers import TransformerEncoderLayer
from modeling.attn import MultiHeadedAttn
from modeling.neural import PositionalFeedForward, PositionalEncoding
import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
import numpy as np


class cntyTRNS(pl.LightningModule):

    def __init__(self, hparams):
        super(cntyTRNS, self).__init__()
        self.hparams = hparams
        self.loss_func = nn.MSELoss(reduction='mean')
    
        self.data_tables = ['cnty_uni_all_years', 'cnty_multivar20_all_years', 'cnty_uni_all_years_age', 'cnty_multivar20_all_years_age_v3', 'cnty_multivar20_all_years_now_age', 'cnty_multivar20_all_years_age_v3_ses']
        self.data_table = self.data_tables[0]
        self.grp_field = 'cnty'
        
        self.num_trans_layers = self.hparams.num_trans_layers
        self.dropout = self.hparams.dropout
        self.inter_heads = self.hparams.inter_heads
        self.ff_dim = self.hparams.ff_dim

        if not self.hparams.multi_loaders:
            self.layer_norm = nn.LayerNorm((self.hparams.hist_len, self.hparams.input_size))
            self.pos_emb = PositionalEncoding(self.dropout, self.hparams.input_size, max_len=self.hparams.hist_len)
        else:
            self.norms = nn.ModuleList([])
            for i in range(1, 4): # max history is 3 
                self.norms.append(nn.LayerNorm((i, self.hparams.input_size)))
            self.pos_emb = PositionalEncoding(self.dropout, self.hparams.input_size, max_len=3)


        self.trns_layers = nn.ModuleList([TransformerEncoderLayer(self.hparams.input_size, self.inter_heads, self.ff_dim, self.dropout) for i in range(self.num_trans_layers)])
        if self.hparams.bidir:
            self.dense = nn.Linear(self.hparams.input_size*2, 1)
        else:
            self.dense = nn.Linear(self.hparams.input_size, 1)


        if self.hparams.bidir:
            self.backward_trns_layers = nn.ModuleList([TransformerEncoderLayer(self.hparams.input_size, self.inter_heads, self.ff_dim, self.dropout) for i in range(self.num_trans_layers)])
            self.backward_pos_emb = PositionalEncoding(self.dropout, self.hparams.input_size, max_len=self.hparams.hist_len)
    
        if self.hparams.multi:
            self.data_table = self.data_tables[1]
            self.grp_field = 'group_id'
        
        if self.hparams.age_uni:
            self.data_table = self.data_tables[2]

        if self.hparams.age:
            self.data_table = self.data_tables[3]
            self.grp_field = 'group_id'

        if self.hparams.age and self.hparams.nowcasting:
            self.data_table = self.data_tables[4]
            self.grp_field = 'group_id'

        if self.hparams.age and self.hparams.ses:
            self.data_table = self.data_tables[5]
            self.grp_field = 'group_id'

        if 'multivar' in self.data_table:
            self.hparams.multi = True
                
        print(f"Using: {self.data_table}")


    def forward(self, embeddings, labels):
        inputs = self.layer_norm(embeddings)
        
        mask = torch.ones(size=(inputs.shape[0], inputs.shape[1]), dtype=torch.bool) # mask is always 1 (no missing data)
        data = self.pos_emb(inputs)

        for trans_layer in self.trns_layers:
            data = trans_layer(data, mask)
    
        if self.hparams.bidir:
            bw_data = self.pos_emb(torch.flip(inputs, [1]))
            for bw_trans_layer in self.backward_trns_layers:
                bw_data = bw_trans_layer(bw_data, mask)

            data = torch.cat((data,bw_data), 2)

        if self.hparams.hist_len > 1 or embeddings.shape[1] > 1:
            preds = torch.flatten(self.dense(data)[:,-1])
        else:
            preds = torch.flatten(self.dense(data))

        labels = torch.flatten(labels)
        loss = self.loss_func(preds,labels)

        return loss, preds, labels
    
    def configure_optimizers(self):
        """
            Defines otpimizers/schedulers
        """

        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.reg)
        return opt

    def training_step(self, batch, batch_idx):
        """
            Called inside Lightnings train.fit()
        """
        if self.hparams.multi_loaders:
            loss = None
            for sub_batch in batch:
                inputs, labels = sub_batch[0], sub_batch[1]
                sub_loss, _, _ = self(inputs, labels)
                if self.trainer.use_dp or self.trainer.use_ddp2:
                    sub_loss = sub_loss.unsqueeze(0)
                
                # sum losses over dataset batches
                if loss is None:
                    loss = sub_loss
                else:
                    loss = loss + sub_loss
        else:
            inputs, labels = batch[0], batch[1]
            loss, _, _ = self(inputs, labels)
            if self.trainer.use_dp or self.trainer.use_ddp2:
                loss = loss.unsqueeze(0)
              
        tb_logs = {'train_loss': loss }

        return {'loss': loss, 'log': tb_logs}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[1]
        loss, preds, labels = self(inputs, labels)
        scores_dict = forecasting_metrics(loss, preds, labels)
              
        # in DP mode make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            scores = [score.unsqueeze(0) for score in scores_dict.values()]
            scores_dict = {key: value for key,value in zip(scores_dict.keys(), scores)}

        return scores_dict

    def validation_epoch_end(self, outputs):
        """
            Gathers results at end of validation loop
        """
        
        multi_gpu = self.trainer.use_dp or self.trainer.use_ddp2
        result = gather_score_dict(outputs, multi_gpu)
        
        return result

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, self.saver)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)
           
    def train_dataloader(self):
        dataloader, _, _ = temporal_dataloader((self.data_table, 'dev_cnty'), self.hparams.hist_len, self.hparams.bs, 'train', self.hparams.num_diffs, True, 'group_id', nowcasting=self.hparams.nowcasting)
        return dataloader
      
    def val_dataloader(self):
        d1, _, _ = temporal_dataloader((self.data_table, 'dev_cnty'), self.hparams.hist_len, self.hparams.bs, 'val', self.hparams.num_diffs, self.hparams.multi, self.grp_field, self.hparams.lang_only)
        return d1

    def test_dataloader(self):
        d1, _, cnties = temporal_dataloader((self.data_table, 'dev_cnty'), self.hparams.hist_len, self.hparams.bs, 'test', self.hparams.num_diffs, self.hparams.multi, self.grp_field, self.hparams.lang_only)
        return d1
    
    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        """
            parameters defined here are available through self.model_params
        """

        parser = HyperOptArgumentParser(parents=[parent_parser])

        # encoder params
        parser.add_argument('--seed', default=1337, type=int, help="random seed")
        parser.add_argument('--epochs', default=5, type=int, help="total num of epochs")
        parser.add_argument('--lr', default=1e-5, type=float, help="intiial learning rate")
        parser.add_argument('--bs', default=6, type=int, help="minibatch size") 
        parser.add_argument('--input_size', default=21, type=int, help="Size of input feature vector")
        parser.add_argument('--hist_len', default=1, type=int, help="Size of history to use for temporal forecasting")
        parser.add_argument('--reg', default=0.0, type=float, help="Amount of weight decay")
        parser.add_argument('--num_diffs', default=1, type=int, help="Number of times to difference neighbors for stationarity")
        parser.add_argument('--multi', default=False, type=bool, help="Toggle multivariate forecasting")
        parser.add_argument('--lang_only', default=False, type=bool, help="Use only language features to forecast. Helpful for debugging")
        parser.add_argument('--age', default=False, type=bool, help="Toggle between age adj rate forecast")
        parser.add_argument('--age_uni', default=False, type=bool, help="Toggle between univariate age adj rate forecast")
        parser.add_argument('--nowcasting', default=False, type=bool, help="Toggle between forecast vs nowcast")
        parser.add_argument('--bidir', default=False, type=bool, help="Toggle bidir")
        parser.add_argument('--num_trans_layers', default=1, type=int, help="Number of transformer blocks to stack")
        parser.add_argument('--multi_loaders', default=False, type=bool, help="Toggle multi loader history [NOT IMPLEMENTED]")
        parser.add_argument('--inter_heads', default=3, type=int, help="Number of attn heads")
        parser.add_argument('--ff_dim', default=64, type=int, help="Size of inner FFNN of trns-layer")
        parser.add_argument('--dropout', default=0.1, type=float, help="Dropout for trns-layer")
        parser.add_argument('--ses', default=False, type=bool, help="Toggle use of multivariate ses variables")

        return parser
