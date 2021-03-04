import sys
sys.path.append("../../")

import math, os, time, copy
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from reco_utils.recommender.newsrec.io.mind_iterator import MINDIterator
from reco_utils.recommender.deeprec.deeprec_utils import cal_metric

class Hparam():
    def __init__(self):

        self.title_size = 30
        self.his_size = 50
        self.data_format = 'news'
        self.npratio = 4

        self.show_step = 10

        self.attention_hidden_dim = 200
        self.word_emb_dim = 300

        self.dropout = 0.2
        self.head_num = 20
        self.head_dim = 20
        self.model_type = 'nrms'

        self.batch_size = 32
        self.epochs = 10
        self.learning_rate = 0.0001
        self.loss = 'cross_entropy_loss'
        self.optimizer = 'adam'
        self.support_quick_scoring = True

        self.metrics = ["group_auc", "mean_mrr", "ndcg@5;10"]

        self.data_path = "../../mind_"+'demo'
        self.train_news_file = os.path.join(self.data_path, 'train', r'news.tsv')
        self.train_behaviors_file = os.path.join(self.data_path, 'train', r'behaviors.tsv')
        self.valid_news_file = os.path.join(self.data_path, 'valid', r'news.tsv')
        self.valid_behaviors_file = os.path.join(self.data_path, 'valid', r'behaviors.tsv')
        self.wordEmb_file = os.path.join(self.data_path, "utils", "embedding.npy")
        self.userDict_file = os.path.join(self.data_path, "utils", "uid2index.pkl")
        self.wordDict_file = os.path.join(self.data_path, "utils", "word_dict.pkl")
        self.test_news_file = None
        self.test_behaviors_file = None

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(-2)], 
                         requires_grad=False)
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, head_num, head_dim, d_model, dropout, attention_hidden_dim):
        super().__init__()
        # self.Qs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(head_num)])
        # self.Vs = nn.ModuleList([nn.Linear(d_model, head_dim) for _ in range(head_num)])
        self.Qs = nn.Linear(d_model, d_model*head_num)
        self.Vs = nn.Linear(d_model, head_dim*head_num)
        self.dropout = nn.Dropout(p=dropout)
        self.V = nn.Linear(head_num*head_dim, attention_hidden_dim)
        self.q = nn.Linear(attention_hidden_dim, 1)
        self.tanh = nn.Tanh()
        self.head_num = head_num
        self.head_dim = head_dim

    def forward(self, x):
        # h = torch.cat([V(torch.matmul(F.softmax(torch.matmul(Q(x), x.transpose(-2, -1)), dim=-2), x)) for Q, V in zip(self.Qs, self.Vs)], dim=-1)
        
        size = x.size()
        size1 = size[:-2]+(self.head_num, size[-2], size[-1])
        size2 = size[:-2]+(self.head_num, size[-1], size[-2])
        size3 = size[:-2]+(self.head_num, size[-2], self.head_dim)
        size4 = size[:-2]+(size[-2], self.head_dim*self.head_num)
        size5 = [1 for _ in range(len(size)+1)]
        size5[-3] = self.head_num
        xq = self.Qs(x)
        xk = x.unsqueeze(-3).repeat(*size5)
        weights = torch.matmul(xq.reshape(size1), xk.reshape(size2))
        weights = F.softmax(weights, dim=-2)
        xv = self.Vs(x)
        h = torch.matmul(weights, xv.reshape(size3))
        h = h.reshape(size4)

        h = self.dropout(h)
        a = self.V(h)
        a = self.tanh(a)
        a = self.q(a)
        a = F.softmax(a, dim=-2)
        a = a.transpose(-2, -1)
        z = torch.matmul(a, h)
        return torch.squeeze(z)

class NewsEncoder(nn.Module):
    def __init__(self, hparam, embeddings, pe):
        super().__init__()
        self.embeddings = embeddings
        self.pe = pe
        self.enc = Encoder(head_num=hparam.head_num, head_dim=hparam.head_dim, d_model=hparam.word_emb_dim, dropout=hparam.dropout, attention_hidden_dim=hparam.attention_hidden_dim)

    def forward(self, x):
        candidates = self.embeddings(x)
        candidates = self.pe(candidates)
        candidates = self.enc(candidates)

        return candidates

class UserEncoder(nn.Module):
    def __init__(self, hparam, embeddings, pe1, pe2):
        super().__init__()
        self.embeddings = embeddings
        self.pe1 = pe1
        self.pe2 = pe2
        self.enc1 = Encoder(head_num=hparam.head_num, head_dim=hparam.head_dim, d_model=hparam.word_emb_dim, dropout=hparam.dropout, attention_hidden_dim=hparam.attention_hidden_dim)
        self.enc2 = Encoder(head_num=hparam.head_num, head_dim=hparam.head_dim, d_model=hparam.head_dim*hparam.head_num, dropout=hparam.dropout, attention_hidden_dim=hparam.attention_hidden_dim)
    
    def forward(self, x):
        histories = self.pe1(self.embeddings(x))
        histories = self.enc1(histories)
        histories = self.pe2(histories)
        histories = self.enc2(histories)

        return histories

class NRMS(nn.Module):
    def __init__(self, hparam):
        super().__init__()
        word_embeddings = torch.FloatTensor(np.load(hparam.wordEmb_file))
        self.embeddings = nn.Embedding.from_pretrained(word_embeddings, freeze=False, padding_idx=0)
        self.pe1 = PositionalEncoding(d_model=hparam.word_emb_dim, dropout=hparam.dropout, max_len=hparam.title_size*2)
        self.pe2 = PositionalEncoding(d_model=hparam.head_dim*hparam.head_num, dropout=hparam.dropout,max_len=hparam.his_size*2)
        self.ne = NewsEncoder(hparam, self.embeddings, self.pe1)
        self.ue = UserEncoder(hparam, self.embeddings, self.pe1, self.pe2)
        # self.ne = NewsEncoder(head_num=hparam.head_num, head_dim=hparam.head_dim, d_model=hparam.word_emb_dim, dropout=hparam.dropout, attention_hidden_dim=hparam.attention_hidden_dim)
        # self.ue = NewsEncoder(head_num=hparam.head_num, head_dim=hparam.head_dim, d_model=hparam.head_dim*hparam.head_num, dropout=hparam.dropout, attention_hidden_dim=hparam.attention_hidden_dim)


    def forward(self, histories, candidates):
        candidates = self.ne(candidates)
        histories = self.ue(histories)
        histories = histories.unsqueeze(-1)        
        preds = torch.matmul(candidates, histories)
        preds = torch.squeeze(preds)

        return preds

def run_eval(ue, ne, iterator, news_filename, behaviors_file):
    """Evaluate the given file and returns some evaluation metrics.

    Args:
        filename (str): A file name that will be evaluated.

    Returns:
        dict: A dictionary contains evaluation metrics.
    """

    # if self.support_quick_scoring:
    group_impr_indexes, group_labels, group_preds = run_fast_eval(
        ue, ne, iterator, news_filename, behaviors_file
    )
    # else:
    #     _, group_labels, group_preds = run_slow_eval(
    #         news_filename, behaviors_file
    #     )
    res = cal_metric(group_labels, group_preds, hparam.metrics)
    return res, group_impr_indexes, group_labels, group_preds

def user(ue, batch_user_input):
    user_input = torch.LongTensor(batch_user_input["clicked_title_batch"]).to(dev)
    user_vector = ue(user_input)
    user_vec = user_vector.cpu().detach().numpy()
    del user_input
    del user_vector
    user_index = batch_user_input["impr_index_batch"]

    return user_index, user_vec

def news(model, batch_news_input):
    news_input = torch.LongTensor(batch_news_input["candidate_title_batch"]).to(dev)
    news_vector = model(news_input)
    news_vec = news_vector.cpu().detach().numpy()
    del news_input
    del news_vector
    news_index = batch_news_input["news_index_batch"]

    return news_index, news_vec

def run_user(ue, iterator, news_filename, behaviors_file):
    # if not hasattr(self, "userencoder"):
    #     raise ValueError("model must have attribute userencoder")

    user_indexes = []
    user_vecs = []
    for batch_data_input in tqdm(
        iterator.load_user_from_file(news_filename, behaviors_file)
    ):
        user_index, user_vec = user(ue, batch_data_input)
        user_indexes.extend(np.reshape(user_index, -1))
        user_vecs.extend(user_vec)

    return dict(zip(user_indexes, user_vecs))

def run_news(model, iterator, news_filename):
    # if not hasattr(self, "newsencoder"):
    #     raise ValueError("model must have attribute newsencoder")

    news_indexes = []
    news_vecs = []
    for batch_data_input in tqdm(
        iterator.load_news_from_file(news_filename)
    ):
        news_index, news_vec = news(model, batch_data_input)
        news_indexes.extend(np.reshape(news_index, -1))
        news_vecs.extend(news_vec)

    return dict(zip(news_indexes, news_vecs))

def run_fast_eval(ue, ne, iterator, news_filename, behaviors_file):
    user_vecs = run_user(ue, iterator, news_filename, behaviors_file)
    news_vecs = run_news(ne, iterator, news_filename)
    


    group_impr_indexes = []
    group_labels = []
    group_preds = []

    for (
        impr_index,
        news_index,
        user_index,
        label,
    ) in tqdm(iterator.load_impression_from_file(behaviors_file)):
        pred = np.dot(
            np.stack([news_vecs[i] for i in news_index], axis=0),
            user_vecs[impr_index],
        )
        group_impr_indexes.append(impr_index)
        group_labels.append(label)
        group_preds.append(pred)

    return group_impr_indexes, group_labels, group_preds

def train_model(iterator, model, optimizer, hparam):
    print("Start training ...")
    # model.eval()
    # evl, _ = run_eval(model.ue, model.ne, iterator, hparam.valid_news_file, hparam.valid_behaviors_file)
    # print(evl)
    for epoch in range(hparam.epochs):
        model.train()
        step = 0
        # hparams.current_epoch = epoch
        epoch_loss = 0
        train_start = time.time()

        tqdm_util = tqdm(
            iterator.load_data_from_file(
                hparam.train_news_file, hparam.train_behaviors_file
            )
        )

        for batch_data_input in tqdm_util:
            # print("dev:", dev)
            model.zero_grad()
            histories = torch.LongTensor(batch_data_input["clicked_title_batch"]).to(dev)
            candidates = torch.LongTensor(batch_data_input["candidate_title_batch"]).to(dev)
            # print("candidates.device: ", candidates.device)
            # print("model.device: ", next(model.parameters()).is_cuda)
            preds = model(histories, candidates)
            # print("preds: ", type(preds))
            # print(preds)
            grounds = torch.LongTensor([0 for _ in range(len(preds))]).to(dev)
            step_data_loss = F.cross_entropy(preds, grounds)
            step_data_loss.backward()
            optimizer.step()

            epoch_loss += float(step_data_loss)
            step += 1
            if step % hparam.show_step == 0:
                tqdm_util.set_description(
                    "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                        step, epoch_loss / step, float(step_data_loss)
                    )
                )
            del preds
            del grounds
            del histories
            del candidates
            del step_data_loss
        train_end = time.time()
        train_time = train_end - train_start
        print("Epoch ", epoch, " training finished...")
        
        with torch.no_grad():
            print("Saving checkpoints...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(hparam.data_path, 'checkpoint'))

            model.eval()
            eval_start = time.time()

            train_info = ",".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in [("logloss loss", epoch_loss / step)]
                ]
            )

            print("Evaluating")
            eval_res, group_impr_indexes, group_labels, group_preds = run_eval(model.ue, model.ne, iterator, hparam.valid_news_file, hparam.valid_behaviors_file)
            eval_info = ", ".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in sorted(eval_res.items(), key=lambda x: x[0])
                ]
            )
            if hparam.test_news_file is not None:
                test_res = run_eval(model.ue, model.ne, iterator, hparam.test_news_file, hparam.test_behaviors_file)
                test_info = ", ".join(
                    [
                        str(item[0]) + ":" + str(item[1])
                        for item in sorted(test_res.items(), key=lambda x: x[0])
                    ]
                )
            eval_end = time.time()
            eval_time = eval_end - eval_start

            if hparam.test_news_file is not None:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                    + "\ntest info: "
                    + test_info
                )
            else:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                )

            # group_impr_indexes, group_labels, group_preds = eval_res
            with open(os.path.join(hparam.data_path, 'prediction.txt'), 'w') as f:
                for impr_index, preds in tqdm(zip(group_impr_indexes, group_preds)):
                    impr_index += 1
                    pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
                    pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
                    f.write(' '.join([str(impr_index), pred_rank])+ '\n')

            print(
                "at epoch {0:d} , train time: {1:.1f} eval time: {2:.1f}".format(
                    epoch, train_time, eval_time
                )
            )



dev = 'cuda' if torch.cuda.is_available() else 'cpu' #
print("dev: ", dev)
hparam = Hparam()
iterator = MINDIterator(hparam)
model = NRMS(hparam).to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=hparam.learning_rate)


train_model(iterator, model, optimizer, hparam)
torch.save(model, os.path.join(hparam.data_path, 'model'))

print("Training finished")


