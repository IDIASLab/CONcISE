# CONcISE
## Introduction
CONcISE is a novel two-stage online approach for timely, scalable and accurate cyberbullying detection on Instagram.

## Citation
To cite your paper, please use the following reference:
```
Mengfan Yao, Charalampos Chelmis, and Daphney-Stavroula Zois. Cyberbullying Ends Here: Towards Robust Detection of Cyberbullying in Social Meida. In 2019 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee.
```

BibTeX:
``` 
@inproceedings{yao2019cyberbullying,
  title = {Cyberbullying Ends Here: Towards Robust Detection of Cyberbullying in Social Media},
  author = {Yao, Mengfan and Chelmis, Charalampos and Zois, Daphney-Stavroula},
  booktitle = {The World Wide Web Conference},
  series = {WWW '19},
  pages = {3427--3433},
  year = {2019},
  isbn = {978-1-4503-6674-8},
  url = {http://doi.acm.org/10.1145/3308558.3313462},
  doi = {10.1145/3308558.3313462},
  organization={ACM}
}
```

### Dataset
We use a corpus of 10 thousand comments, manually annotated as harassing or not by 10 experts. We focus on harassing comments due to their commonality to a number of types of unwanted behavior, including cyberharassment and cyberbullying. We used this data to evaluate our method for [robust and timely detection of cyberbullying](https://dl.acm.org/citation.cfm?id=3313462) as well as for [harashment anticipation](https://doi.org/10.1145/3292522.3326024).

Our labeled corpus spans 22.1% of all media sessions containing at least 40% profanities in the dataset introduced by this [paper](https://dl.acm.org/citation.cfm?id=3192424.3192459). Of all media sessions containing at least 40% profanities, 47.5% had been labeled as positive if:
> there are negative words and/or comments with intent to harm someone or other, and the posts include two or more repeated negativity against a victim.

#### Dataset Access
Email us at cchelmis@albany.edu if you are interested in our dataset! We are happy to share our data with you.

## Prerequisites
python 2.7 and the following libraries
```
pandas
numpy
os
statsmodels
```

## Files
```
CONcISE.py: main function of the approach
train.csv: a sample training data 
test.csv: a sample testing data
```
