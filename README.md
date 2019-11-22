# PC-RNN

This repo is an implementation of PC-RNN proposed in paper "Patent Citation Dynamics Modeling via Multi-Attention Recurrent Networks".

**Title**: Patent Citation Dynamics Modeling via Multi-Attention Recurrent Networks

**Abstract**: Modeling and forecasting forward citations to a patent is a central task for the discovery of emerging technologies and for measuring the pulse of inventive progress. Conventional methods for forecasting these forward citations cast the problem as analysis of temporal point processes which rely on the conditional intensity of previously received citations. Recent approaches model the conditional intensity as a chain of recurrent neural networks to capture memory dependency in hopes of reducing the restrictions of the parametric form of the intensity function. For the problem of patent citations, we observe that forecasting a patent’s chain of citations benefits from not only the patent’s history itself but also from the historical citations of assignees and inventors associated with that patent. In this paper, we propose a sequence-to-sequence model which employs an attention-of-attention mechanism to capture the dependencies of these multiple time sequences. Furthermore, the proposed model is able to forecast both the timestamp and the category of a patent’s next citation. Extensive experiments on a large patent citation dataset collected from USPTO demonstrate that the proposed model outperforms state-of-the-art models at forward citation forecasting.

```bib
@inproceedings{DBLP:conf/ijcai/JiCSFLR19,
  author    = {Taoran Ji and
               Zhiqian Chen and
               Nathan Self and
               Kaiqun Fu and
               Chang{-}Tien Lu and
               Naren Ramakrishnan},
  title     = {Patent Citation Dynamics Modeling via Multi-Attention Recurrent Networks},
  booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on
               Artificial Intelligence, {IJCAI} 2019, Macao, China, August 10-16,
               2019},
  pages     = {2621--2627},
  year      = {2019},
  crossref  = {DBLP:conf/ijcai/2019},
  url       = {https://doi.org/10.24963/ijcai.2019/364},
  doi       = {10.24963/ijcai.2019/364},
  timestamp = {Tue, 20 Aug 2019 16:18:18 +0200},
  biburl    = {https://dblp.org/rec/bib/conf/ijcai/JiCSFLR19},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
