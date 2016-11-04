# byteNet-tensorflow

[![Join the chat at https://gitter.im/byteNet-tensorflow/Lobby](https://badges.gitter.im/byteNet-tensorflow/Lobby.svg)](https://gitter.im/byteNet-tensorflow/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

This is a tensorflow implementation of the byteNet model from the paper [Neural Machine Translation in Linear Time][1]. 

From the abstract
>The ByteNet decoder attains state-of-the-art performance on character-level language modeling and outperforms the previous best results obtained with recurrent neural networks.

## ByteNet Encoder-Decoder Model:
![Model architecture](http://i.imgur.com/IE6Zq6o.jpg)

Image Source - [Neural Machine Translation in Linear Time][1] paper


## Implementation Notes
1. The model has been defined in ```ByteNet/model.py```. ```ByteNet/ops.py``` contains the dilated convolution implementation (adapted from [tensorflow wavenet][2] ).
2. The model can be configured by editing model_config.py.
3. Sub-batch normalisation has not been implemented.
4. The model (byteNet decoder) has been tested on character generation and not machine transalation. (Work in progress).

## Datasets
The model has been trained on Shakespeare text (the same dataset which was used in Karpathy's blog). I have included the text file in the repository ```Data/shakespeare.txt```.

## Training
Configure the model by editing ```model_config.py```. Train on a text corpus by

```python train.py --data_dir=PATH_TO_FOLDER_CONTAINING_TXT_FILES```

```python train.py --help``` for more options.

## Text Generation
Generate new text samples using

```python generate.py --seed="SOME_TEXT_TO_START_WITH" --num_chars=NUM_CHARS_TO_GENERATE --model_path=PATH_TO_TRAINED_MODEL --output_file=OUTPUT_FILE_PATH```

Note - This is not the most efficient generator implementation. Refer to tensorflow wavenet generator implementation for a faster generator.

## Samples Generated
I haven't experimented much as of now. Following are some text samples hallucinated by the network


seed = "ANTONIO"
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

seed = "PORTIA"
```
PORTIA:
What say these fairs, man? what say these samese me?

First Citizen:
The king is so indeed, sir, the sin, that I was,
That the best contrary to the court of France,
That thou wert bear to the mouth of this son,
That thou dost speak to me and the sense of France,
That thou dost stand and stand and true and souls,
```

## Pretrained Model
You may play with the pretrained model on the shakespeare dataset. Save the model in ```Data/Models```.

[Link to the trained model][3].


## TODO
- Train the model on machine-translation.
- Implement Sub-batch Normalization. (Contributors welcomed - I don't understand this well)
- Check whether bag of n-grams character encoding makes a difference
- Efficient generator implementation.

## References
- [Neural Machine Translation in Linear Time][1] paper
- [Tensorflow Wavenet][2] code


[1]:https://arxiv.org/abs/1610.10099
[2]:https://github.com/ibab/tensorflow-wavenet
[3]:https://drive.google.com/file/d/0B30fmeZ1slbBYWVSWnMyc3hXQVU/view?usp=sharing
