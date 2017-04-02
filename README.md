# byteNet-tensorflow

[![Join the chat at https://gitter.im/byteNet-tensorflow/Lobby](https://badges.gitter.im/byteNet-tensorflow/Lobby.svg)](https://gitter.im/byteNet-tensorflow/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

This is a tensorflow implementation of the byte-net model from DeepMind's paper [Neural Machine Translation in Linear Time][1]. 

From the abstract
>The ByteNet decoder attains state-of-the-art performance on character-level language modeling and outperforms the previous best results obtained with recurrent neural networks.  The ByteNet also achieves a performance on raw character-level machine translation that approaches that of the best neural translation models that run in quadratic time. The implicit structure learnt by the ByteNet mirrors the expected alignments between the sequences.

## ByteNet Encoder-Decoder Model:
![Model architecture](http://i.imgur.com/IE6Zq6o.jpg)

Image Source - [Neural Machine Translation in Linear Time][1] paper

The model applies dilated 1d convolutions on the sequential data, layer by layer to obain the source encoding. The decoder then applies masked 1d convolutions on the target sequence (conditioned by the encoder output) to obtain the next character in the target sequence.The character generation model is just the byteNet decoder, while the machine translation model is the combined encoder and decoder.

## Implementation Notes
1. The model has been defined in ```ByteNet/model.py```. ```ByteNet/ops.py``` contains the dilated convolution implementation (adapted from [tensorflow wavenet][2] ).
2. The model can be configured by editing model_config.py.
3. Sub-batch normalisation has not been implemented.
4. Bags of n-grams have not been used.
5. Number of residual channels 512 (Configurable in model_config.py).

## Requirements
- Python 3.3+
- TensorFlow r1.0+

## Datasets
- The character generation model has been trained on [Shakespeare text][4]. I have included the text file in the repository ```Data/shakespeare.txt```.
- The machine translation model has been trained for german to english translation. You may download the news commentary dataset from here [http://www.statmt.org/wmt16/translation-task.html][5]

## Training
1. Text Generation - Configure the model by editing ```model_config.py```. Train on a text corpus by
  
  ```python train_generator.py --data_dir=PATH_TO_FOLDER_CONTAINING_TXT_FILES```
  
  ```python train.py --help``` for more options.
2. Machine Translation - Configure the model by editing ```model_config.py```. Train translation model from source to target by:
  
  ```python train_translator.py --source_file=SOURCE_FILE_PATH --target_file=TARGET_FILE_PATH```
  
  ```python train.py --help``` for more options.

## Results
| Text Generation        | Machine Translation  |
| ----- | -----|
| Generate new samples using : ```python generate.py --seed="SOME_TEXT_TO_START_WITH"```| translate.py still in development. You can test sample translations from the dataset using ```python translate.py``` |

#### Sample Generations

```
ANTONIO:
What say you to this part of this to thee?

KING PHILIP:
What say these faith, madam?

First Citizen:
The king of England, the will of the state,
That thou dost speak to me, and the thing that shall
In this the son of this devil to the storm,
That thou dost speak to thee to the world,
That thou dost see the bear that was the foot,

```

#### Translation Results to be updated

## TODO
- Implement Sub-batch Normalization.
- Check whether bag of n-grams character encoding makes a difference

## References
- [Neural Machine Translation in Linear Time][1] paper
- [Tensorflow Wavenet][2] code

[1]:https://arxiv.org/abs/1610.10099
[2]:https://github.com/ibab/tensorflow-wavenet
[3]:https://drive.google.com/file/d/0B30fmeZ1slbBYWVSWnMyc3hXQVU/view?usp=sharing
[4]:http://cs.stanford.edu/people/karpathy/char-rnn/
[5]:http://www.statmt.org/wmt16/translation-task.html
