# Applied Deep Learning Hw1

## Introduction

This project predicts the summary of given text, including three models:

1. Extractive summary by labeling tokens

2. Abstractive summary by Seq2Seq

3. Abstractive summary by Seq2Seq + attention

The datasets are provided in /datasets, and the result would be evaluated by Rouge score.

## Installation

### Environment

```shell
conda create --name <env> --file requirements.txt
```

### Pre-trained word2vec Model

I use 6B_300d glove pre-trained embedding model

```shell
sh Download.sh
```

This command would also download the results of three models

## Extractive Summary Model

### Training

#### Model

Use two layer LSTM and one Linear layer to generate the label of each token. To avoid overfitting, dropout is involved, which can be adjusted by input argument.

#### Loss

Use BCEWithLogitsLoss with pos_weight=7, which is the portion of negative and positive sample in dataset.

#### Train

Due to the difference in text length, I pad all the text to length 400 (token count). Also, I created masks of every text to filter the padding value when calculate the loss, which is effective.

```shell
python src/train_extractive.py --batch_size 32 --epochs 20 --lr 0.001 --word_dim 300 --dropout 0.5
```

This command will use the funtions and classes in preprocess_extractive.py and dataset.py

### Evaluating

By downloading the model, the performence can be evaluated without training again.

```shell
python src/eval_extractive.py --input datasets/test.jsonl.release --output predict/predict_extractive.jsonl
```

To save the histogram of relative_location, add the argument:

```shell
--relative_location True
```

or

```shell
sh extractive.sh ${input_data_path} ${output_result_path}
```

## Seq2Seq

### Model

Use two layer bidirectional GRU as encoder, two layer bidirectional GRU and single linear layer as decoder.

### Train

```shell
python src/train_seq2seq --batch_size 64 --epochs 10 --lr 0.001 --dropout 0.25
```

This command will use the funtions and classes in preprocess_seq2seq.py and dataset.py

### Evaluating

By downloading the model, the performence can be evaluated without training again.

```shell
python src/eval_seq2seq.py --input datasets/test.jsonl.release --output predict/predict_seq2seq.jsonl
```

or

```shell
sh seq2seq.sh ${input_data_path} ${output_result_path}
```

## Seq2Seq + Attention

### Model

Same with Seq2Seq but attention

### Train

```shell
python src/train_atten --batch_size 64 --epochs 10 --lr 0.001 --dropout 0.25
```

This command will use the funtions and classes in preprocess_seq2seq.py and dataset.py

### Evaluating

By downloading the model, the performence can be evaluated without training again.

```shell
python src/eval_atten.py --input datasets/test.jsonl.release --output predict/predict_seq2seq.jsonl
```

To show the image of attention, just add the parser:

```shell
--draw_atten True
```

or

```shell
sh atten.sh ${input_data_path} ${output_result_path}
```


## Author

r08922115  
吳冠霖
