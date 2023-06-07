# Paper Implementation

This repo contains an implementation of [Neural Word Embedding as Implicit Matrix Factorization](https://papers.nips.cc/paper_files/paper/2014/hash/feab05aa91085b7a8012516bc3533958-Abstract.html) by Omer Levy and Yoav Goldberg.


## Citation

```sql
@inproceedings{NIPS2014_feab05aa,
 author = {Levy, Omer and Goldberg, Yoav},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {Z. Ghahramani and M. Welling and C. Cortes and N. Lawrence and K.Q. Weinberger},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {Neural Word Embedding as Implicit Matrix Factorization},
 url = {https://proceedings.neurips.cc/paper_files/paper/2014/file/feab05aa91085b7a8012516bc3533958-Paper.pdf},
 volume = {27},
 year = {2014}
}
```

### Data

```
mkdir data
cd data
wget https://downloads.wortschatz-leipzig.de/corpora/eng_news-typical_2016_1M.tar.gz 
tar -xvzf eng_news-typical_2016_1M.tar.gz
```

### Parsing

```
python src/tools/dependency-parser.py
python src/tools/pair-data.py
```

### Word2Vec

```
python src/main.py
```

### Visualisations

```
python src/tools/theasarus.py --word money

```

#### Example similarities

Words most similar to `money`:

```txt
[('resources', 0.12),
 ('capital', 0.11),
 ('paper', 0.11),
 ('property', 0.1),
 ('total', 0.1),
 ('value', 0.09),
 ('cards', 0.09),
 ('time', 0.09),
 ('knowledge', 0.09),
 ('products', 0.09)]
```