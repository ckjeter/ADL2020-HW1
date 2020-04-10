#!/usr/bin/env bash

glove=https://www.dropbox.com/s/au0m8739yyr9hjp/glove.6B.300d.txt?dl=1
extractive=https://www.dropbox.com/s/doz6yp8bd8u3r1k/Extractive.pth?dl=1
s2s=https://www.dropbox.com/s/5vzv4qnat9ms874/s2s.pth?dl=1
atten=https://www.dropbox.com/s/75yw67dh05lw2k7/atten.pth?dl=1
embed=https://www.dropbox.com/s/fg4qm6xfm485ywb/embedding.pkl?dl=1

wget -O glove/glove.6B.300d.txt "${glove}"
wget -O datasets/seq2seq/embedding.pkl "${embed}"
wget -O model/Extractive.pth "${extractive}"
wget -O model/s2s.pth "${s2s}"
wget -O model/atten.pth "${atten}"
