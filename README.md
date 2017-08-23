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
1. The character generation model is defined in ```ByteNet/generator.py``` and the translation model is defined in ```ByteNet/translator.py```. ```ByteNet/ops.py``` contains the bytenet residual block, dilated conv1d and layer normalization.
2. The model can be configured by editing model_config.py.
5. Number of residual channels 512 (Configurable in model_config.py).

## Requirements
- Python 2.7.6
- Tensorflow 1.2.0

## Datasets
- The character generation model has been trained on [Shakespeare text][4]. I have included the text file in the repository ```Data/generator_training_data/shakespeare.txt```.
- The machine translation model has been trained for german to english translation. You may download the news commentary dataset from here [http://www.statmt.org/wmt16/translation-task.html][5]

## Training
Create the following directories ```Data/tb_summaries/translator_model```, ```Data/tb_summaries/generator_model```,  ```Data/Models/generation_model```, ```Data/Models/translation_model```.

- <b>Text Generation</b>
  * Configure the model by editing ```model_config.py```.
  * Save the text files to train on, in ```Data/generator_training_data```. A sample shakespeare.txt is included in the repo.
  * Train the model by : ```python train_generator.py --text_dir="Data/generator_training_data"```
  * ```python train_generator.py --help``` for more options.
  
- <b>Machine Translation</b>
  * Configure the model by editing ```model_config.py```.
  * Save the source and target sentences in separate files in ```Data/MachineTranslation```. You may download the new commentary training corpus using [this link][6].
  * The model is trained on buckets of sentence pairs of length in mutpiples of a configurable parameter ```bucket_quant```. The sentences are padded with a special character beyond the actual length.
  * Train translation model using:
    - ```python train_translator.py --source_file=<source sentences file> --target_file=<target sentences file> --bucket_quant=50```
    - ```python train_translator.py``` --help for more options.
   
    

## Generating Samples
- Generate new samples using : 
  * ```python generate.py --seed="SOME_TEXT_TO_START_WITH" --sample_size=<SIZE OF GENERATED SEQUENCE>```
- You can test sample translations from the dataset using ```python translate.py```. 
  * This will pick random source sentences from the dataset and translate them.

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
- Evaluating the translation Model
- Implement beam search - Contributors welcomed. Currently the model samples from the probability distribution from the top k most probable predictions.
## References
- [Neural Machine Translation in Linear Time][1] paper
- [Tensorflow Wavenet][2] code
- [Sugar Tensor Source Code][7] For implementing some ops.

[1]:https://arxiv.org/abs/1610.10099
[2]:https://github.com/ibab/tensorflow-wavenet
[3]:https://drive.google.com/file/d/0B30fmeZ1slbBYWVSWnMyc3hXQVU/view?usp=sharing
[4]:http://cs.stanford.edu/people/karpathy/char-rnn/
[5]:http://www.statmt.org/wmt16/translation-task.html
[6]:http://data.statmt.org/wmt16/translation-task/training-parallel-nc-v11.tgz
[7]:https://github.com/buriburisuri/sugartensor
