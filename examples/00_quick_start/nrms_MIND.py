# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
# 
# <i>Licensed under the MIT License.</i>
# %% [markdown]
# # NRMS: Neural News Recommendation with Multi-Head Self-Attention
# NRMS \[1\] is a neural news recommendation approach with multi-head selfattention. The core of NRMS is a news encoder and a user encoder. In the newsencoder, a multi-head self-attentions is used to learn news representations from news titles by modeling the interactions between words. In the user encoder, we learn representations of users from their browsed news and use multihead self-attention to capture the relatedness between the news. Besides, we apply additive
# attention to learn more informative news and user representations by selecting important words and news.
# 
# ## Properties of NRMS:
# - NRMS is a content-based neural news recommendation approach.
# - It uses multi-self attention to learn news representations by modeling the iteractions between words and learn user representations by capturing the relationship between user browsed news.
# - NRMS uses additive attentions to learn informative news and user representations by selecting important words and news.
# 
# ## Data format:
# For quicker training and evaluaiton, we sample MINDdemo dataset of 5k users from [MIND small dataset](https://msnews.github.io/). The MINDdemo dataset has the same file format as MINDsmall and MINDlarge. If you want to try experiments on MINDsmall and MINDlarge, please change the dowload source. Select the MIND_type parameter from ['large', 'small', 'demo'] to choose dataset.
#  
# **MINDdemo_train** is used for training, and **MINDdemo_dev** is used for evaluation. Training data and evaluation data are composed of a news file and a behaviors file. You can find more detailed data description in [MIND repo](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md)
# 
# ### news data
# This file contains news information including newsid, category, subcatgory, news title, news abstarct, news url and entities in news title, entities in news abstarct.
# One simple example: <br>
# 
# `N46466	lifestyle	lifestyleroyals	The Brands Queen Elizabeth, Prince Charles, and Prince Philip Swear By	Shop the notebooks, jackets, and more that the royals can't live without.	https://www.msn.com/en-us/lifestyle/lifestyleroyals/the-brands-queen-elizabeth,-prince-charles,-and-prince-philip-swear-by/ss-AAGH0ET?ocid=chopendata	[{"Label": "Prince Philip, Duke of Edinburgh", "Type": "P", "WikidataId": "Q80976", "Confidence": 1.0, "OccurrenceOffsets": [48], "SurfaceForms": ["Prince Philip"]}, {"Label": "Charles, Prince of Wales", "Type": "P", "WikidataId": "Q43274", "Confidence": 1.0, "OccurrenceOffsets": [28], "SurfaceForms": ["Prince Charles"]}, {"Label": "Elizabeth II", "Type": "P", "WikidataId": "Q9682", "Confidence": 0.97, "OccurrenceOffsets": [11], "SurfaceForms": ["Queen Elizabeth"]}]	[]`
# <br>
# 
# In general, each line in data file represents information of one piece of news: <br>
# 
# `[News ID] [Category] [Subcategory] [News Title] [News Abstrct] [News Url] [Entities in News Title] [Entities in News Abstract] ...`
# 
# <br>
# 
# We generate a word_dict file to tranform words in news title to word indexes, and a embedding matrix is initted from pretrained glove embeddings.
# 
# ### behaviors data
# One simple example: <br>
# `1	U82271	11/11/2019 3:28:58 PM	N3130 N11621 N12917 N4574 N12140 N9748	N13390-0 N7180-0 N20785-0 N6937-0 N15776-0 N25810-0 N20820-0 N6885-0 N27294-0 N18835-0 N16945-0 N7410-0 N23967-0 N22679-0 N20532-0 N26651-0 N22078-0 N4098-0 N16473-0 N13841-0 N15660-0 N25787-0 N2315-0 N1615-0 N9087-0 N23880-0 N3600-0 N24479-0 N22882-0 N26308-0 N13594-0 N2220-0 N28356-0 N17083-0 N21415-0 N18671-0 N9440-0 N17759-0 N10861-0 N21830-0 N8064-0 N5675-0 N15037-0 N26154-0 N15368-1 N481-0 N3256-0 N20663-0 N23940-0 N7654-0 N10729-0 N7090-0 N23596-0 N15901-0 N16348-0 N13645-0 N8124-0 N20094-0 N27774-0 N23011-0 N14832-0 N15971-0 N27729-0 N2167-0 N11186-0 N18390-0 N21328-0 N10992-0 N20122-0 N1958-0 N2004-0 N26156-0 N17632-0 N26146-0 N17322-0 N18403-0 N17397-0 N18215-0 N14475-0 N9781-0 N17958-0 N3370-0 N1127-0 N15525-0 N12657-0 N10537-0 N18224-0`
# <br>
# 
# In general, each line in data file represents one instance of an impression. The format is like: <br>
# 
# `[Impression ID] [User ID] [Impression Time] [User Click History] [Impression News]`
# 
# <br>
# 
# User Click History is the user historical clicked news before Impression Time. Impression News is the displayed news in an impression, which format is:<br>
# 
# `[News ID 1]-[label1] ... [News ID n]-[labeln]`
# 
# <br>
# Label represents whether the news is clicked by the user. All information of news in User Click History and Impression News can be found in news data file.
# %% [markdown]
# ## Global settings and imports

# %%
import sys
sys.path.append("../../")
import os
import numpy as np
import zipfile
from tqdm import tqdm
import scrapbook as sb
# from tempfile import TemporaryDirectory
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources 
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams
from reco_utils.recommender.newsrec.models.nrms import NRMSModel
from reco_utils.recommender.newsrec.io.mind_iterator import MINDIterator
from reco_utils.recommender.newsrec.newsrec_utils import get_mind_data_set

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

# %% [markdown]
# ## Prepare parameters

# %%
epochs = 5
seed = 42
batch_size = 32

# Options: demo, small, large
MIND_type = 'demo'

# %% [markdown]
# ## Download and load data

# %%
# tmpdir = TemporaryDirectory()
# data_path = tmpdir.name
data_path = "../../mind_"+MIND_type

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
yaml_file = os.path.join(data_path, "utils", r'nrms.yaml')

mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_type)

if not os.path.exists(train_news_file):
    print(mind_url)
    print(mind_train_dataset)
#     download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)
    
if not os.path.exists(valid_news_file):
    print(mind_url)
    print(mind_dev_dataset)
#     download_deeprec_resources(mind_url, \
#                                os.path.join(data_path, 'valid'), mind_dev_dataset)
if not os.path.exists(yaml_file):

    print(mind_utils)
#     download_deeprec_resources(r'https://recodatasets.z20.web.core.windows.net/newsrec/', \
#                                os.path.join(data_path, 'utils'), mind_utils)

# %% [markdown]
# ## Create hyper-parameters

# %%
hparams = prepare_hparams(yaml_file, 
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file, 
                          userDict_file=userDict_file,
                          batch_size=batch_size,
                          epochs=epochs,
                          show_step=10)
print(hparams)

# %% [markdown]
# ## Train the NRMS model

# %%
iterator = MINDIterator


# %%
model = NRMSModel(hparams, iterator, seed=seed)


# %%
print(model.run_eval(valid_news_file, valid_behaviors_file))

#%%
model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)


# %%
# get_ipython().run_cell_magic('time', '', 'model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)')


# %%
import tensorflow as tf
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))


# %%
get_ipython().run_cell_magic('time', '', 'res_syn = model.run_eval(valid_news_file, valid_behaviors_file)\nprint(res_syn)')


# %%
sb.glue("res_syn", res_syn)

# %% [markdown]
# ## Save the model

# %%
model_path = os.path.join(data_path, "model")
os.makedirs(model_path, exist_ok=True)

model.model.save_weights(os.path.join(model_path, "nrms_ckpt"))

# %% [markdown]
# ## Output Predcition File
# This code segment is used to generate the prediction.zip file, which is in the same format in [MIND Competition Submission Tutorial](https://competitions.codalab.org/competitions/24122#learn_the_details-submission-guidelines).
# 
# Please change the `MIND_type` parameter to `large` if you want to submit your prediction to [MIND Competition](https://msnews.github.io/competition.html).

# %%
group_impr_indexes, group_labels, group_preds = model.run_fast_eval(valid_news_file, valid_behaviors_file)


# %%
with open(os.path.join(data_path, 'prediction.txt'), 'w') as f:
    for impr_index, preds in tqdm(zip(group_impr_indexes, group_preds)):
        impr_index += 1
        pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
        pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
        f.write(' '.join([str(impr_index), pred_rank])+ '\n')


# %%
f = zipfile.ZipFile(os.path.join(data_path, 'prediction.zip'), 'w', zipfile.ZIP_DEFLATED)
f.write(os.path.join(data_path, 'prediction.txt'), arcname='prediction.txt')
f.close()

# %% [markdown]
# ## Reference
# \[1\] Wu et al. "Neural News Recommendation with Multi-Head Self-Attention." in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)<br>
# \[2\] Wu, Fangzhao, et al. "MIND: A Large-scale Dataset for News Recommendation" Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. https://msnews.github.io/competition.html <br>
# \[3\] GloVe: Global Vectors for Word Representation. https://nlp.stanford.edu/projects/glove/

