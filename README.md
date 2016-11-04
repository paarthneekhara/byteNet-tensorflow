# byteNet-tensorflow

This is a tensorflow implementation of the byteNet model from the paper [Neural Machine Translation in Linear Time][1]. 

From the abstract
>The ByteNet decoder attains state-of-the-art performance on character-level language modelling and outperforms the previous best results obtained with recurrent neural networks.

ByteNet Encoder-Decoder Model:
![Model architecture](http://i.imgur.com/zRkhFwJ.png)

## Implementation Notes
1. The model can be configured by editing model_config.py.
2. Sub-batch normalisation has not been implemented
3. The model (byteNet decoder) has been tested on character generation and not machine transalation. (Work in progress).

## Datasets
The model has been trained on Shakespeare text (the same dataset which was used in Karpathy's blog). I have included the text file in the repository ```Data/shakespeare.txt```.

## Training
Configure the model by editing ```model_config.py```. Use ```python train.py --data_dir=PATH_TO_FOLDER_CONTAINING_TXT_FILES``` .
```python train.py --help``` for more options.

## Text Generation
Generate new text samples using

```python generate.py --seed="SOME_TEXT_TO_START_WITH" --num_chars=NUM_CHARS_TO_GENERATE --model_path="PATH_TO_TRAINED_MODEL".```
The generated text is printed on the terminal. 

Note - This is not the most efficient generator implementation. Refer to tensorflow wavenet generator implementation for a faster generator.

## Samples Generated
seed = "ANTONIO"
```

```

## Pretrained Model
You may play with the pretrained model on the shakespeare dataset. Save the model in ```Data/Models```.

Link to the trained model.


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
[3]:
