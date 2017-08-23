import os
from os import listdir
from os.path import isfile, join
import numpy as np

class Data_Loader:
    def __init__(self, options):
        if options['model_type'] == 'translation':
            source_file = options['source_file']
            target_file = options['target_file']

            self.max_sentences = None
            if 'max_sentences' in options:
                self.max_sentences = options['max_sentences']

            with open(source_file) as f:
                self.source_lines = f.read().decode("utf-8").split('\n')
            with open(target_file) as f:
                self.target_lines = f.read().decode("utf-8").split('\n')

            if self.max_sentences:
                self.source_lines = self.source_lines[0:self.max_sentences]
                self.target_lines = self.target_lines[0:self.max_sentences]

            print "Source Sentences", len(self.source_lines)
            print "Target Sentences", len(self.target_lines)

            self.bucket_quant = options['bucket_quant']
            self.source_vocab = self.build_vocab(self.source_lines)
            self.target_vocab = self.build_vocab(self.target_lines)

            print "SOURCE VOCAB SIZE", len(self.source_vocab)
            print "TARGET VOCAB SIZE", len(self.target_vocab)
        
        elif options['model_type'] == 'generator':
            dir_name = options['dir_name']
            files = [ join(dir_name, f) for f in listdir(dir_name) if ( isfile(join(dir_name, f)) and ('.txt' in f) ) ]
            text = []
            for f in files:
                text += list(open(f).read())
            
            vocab = {ch : True for ch in text}
            print "Bool vocab", len(vocab)
            self.vocab_list = [ch for ch in vocab]
            print "vocab list", len(self.vocab_list)
            self.vocab_indexed = {ch : i for i, ch in enumerate(self.vocab_list)}
            print "vocab_indexed", len(self.vocab_indexed)

            for index, item in enumerate(text):
                text[index] = self.vocab_indexed[item]
            self.text = np.array(text, dtype='int32')

    def load_generator_data(self, sample_size):
        text = self.text
        mod_size = len(text) - len(text)%sample_size
        text = text[0:mod_size]
        text = text.reshape(-1, sample_size)
        return text, self.vocab_indexed


    def load_translation_data(self):
        source_lines = []
        target_lines = []
        for i in range(len(self.source_lines)):
            source_lines.append( self.string_to_indices(self.source_lines[i], self.source_vocab) )
            target_lines.append( self.string_to_indices(self.target_lines[i], self.target_vocab) )

        buckets = self.create_buckets(source_lines, target_lines)

        # frequent_keys = [ (-len(buckets[key]), key) for key in buckets ]
        # frequent_keys.sort()

        # print "Source", self.inidices_to_string( buckets[ frequent_keys[3][1] ][5][0], self.source_vocab)
        # print "Target", self.inidices_to_string( buckets[ frequent_keys[3][1] ][5][1], self.target_vocab)
        
        return buckets, self.source_vocab, self.target_vocab



    def create_buckets(self, source_lines, target_lines):
        
        bucket_quant = self.bucket_quant
        source_vocab = self.source_vocab
        target_vocab = self.target_vocab

        buckets = {}
        for i in xrange(len(source_lines)):
            
            source_lines[i] = np.concatenate( (source_lines[i], [source_vocab['eol']]) )
            target_lines[i] = np.concatenate( ([target_vocab['init']], target_lines[i], [target_vocab['eol']]) )
            
            sl = len(source_lines[i])
            tl = len(target_lines[i])


            new_length = max(sl, tl)
            if new_length % bucket_quant > 0:
                new_length = ((new_length/bucket_quant) + 1 ) * bucket_quant    
            
            s_padding = np.array( [source_vocab['padding'] for ctr in xrange(sl, new_length) ] )

            # NEED EXTRA PADDING FOR TRAINING.. 
            t_padding = np.array( [target_vocab['padding'] for ctr in xrange(tl, new_length + 1) ] )

            source_lines[i] = np.concatenate( [ source_lines[i], s_padding ] )
            target_lines[i] = np.concatenate( [ target_lines[i], t_padding ] )

            if new_length in buckets:
                buckets[new_length].append( (source_lines[i], target_lines[i]) )
            else:
                buckets[new_length] = [(source_lines[i], target_lines[i])]

            if i%1000 == 0:
                print "Loading", i
            
        return buckets

    def build_vocab(self, sentences):
        vocab = {}
        ctr = 0
        for st in sentences:
            for ch in st:
                if ch not in vocab:
                    vocab[ch] = ctr
                    ctr += 1

        # SOME SPECIAL CHARACTERS
        vocab['eol'] = ctr
        vocab['padding'] = ctr + 1
        vocab['init'] = ctr + 2

        return vocab

    def string_to_indices(self, sentence, vocab):
        indices = [ vocab[s] for s in sentence ]
        return indices

    def inidices_to_string(self, sentence, vocab):
        id_ch = { vocab[ch] : ch for ch in vocab } 
        sent = []
        for c in sentence:
            if id_ch[c] == 'eol':
                break
            sent += id_ch[c]

        return "".join(sent)

    def get_batch_from_pairs(self, pair_list):
        source_sentences = []
        target_sentences = []
        for s, t in pair_list:
            source_sentences.append(s)
            target_sentences.append(t)

        return np.array(source_sentences, dtype = 'int32'), np.array(target_sentences, dtype = 'int32')


def main():
    # FOR TESTING ONLY
    trans_options = {
        'model_type' : 'translation',
        'source_file' : 'Data/MachineTranslation/news-commentary-v11.de-en.de',
        'target_file' : 'Data/MachineTranslation/news-commentary-v11.de-en.en',
        'bucket_quant' : 25,
    }
    gen_options = {
        'model_type' : 'generator', 
        'dir_name' : 'Data',
    }

    dl = Data_Loader(gen_options)
    text_samples, vocab = dl.load_generator_data( 1000 )
    print dl.inidices_to_string(text_samples[1], vocab)
    print text_samples.shape
    print np.max(text_samples)
    # buckets, source_vocab, target_vocab = dl.load_translation_data()

if __name__ == '__main__':
    main()