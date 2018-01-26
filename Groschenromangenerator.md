
# ~~Master Thesis Generator~~ Groschenromangenerator
This notebook uses text data from a famous german groschenromanautorin Carola Pigisch who publishes her lovely stories on http://groschenromanblog.de/ to generate our own groschenroman based on the style of Carola Pigisch. Lets have a look..




```python
import helper
data_dir = 'data/Grandhotel Herz.txt'
```

We take her famous series Grandhotel Herz to let our algorythm learn from its passion.
Lets have a first impression:


```python
with open('data/Grandhotel Herz, Folge 1.txt', "r") as f:
        data = f.read()
data[:100]
```




    '“Guten Morgen, Herr Ludenhoff. Wünschen wohl geruht zu haben, der Herr?” Johann stand stramm und lüf'




```python
import os

bibliothek = []
a ="data"
for roman in os.listdir(a):
    with open('data/'+roman, "r") as f:
        data = f.read()
        data = data.replace('Mitzi','Sibille')
        data = data.replace('Max','Frank')
        bibliothek.append(data)

bibliothek[1][:100]
```




    'Erneut schüttelte Sibille das Laken auf. Leicht wie eine Feder legte sich der kostbare Stoff auf die'



no names were changed, even if it looks like


```python
text = ' '.join(roman for roman in bibliothek)
with open('data/Grandhotel Herz.txt','w') as f: 
    data = f.write(text) 
len(text)
```




    176022



## Explore the Data



```python
view_sentence_range = (0, 2)

import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 7169
    Number of scenes: 245
    Average number of sentences in each scene: 2.7714285714285714
    Number of lines: 924
    Average number of words in each line: 30.713203463203463
    
    The sentences 0 to 2:
    “Guten Morgen, Herr Ludenhoff. Wünschen wohl geruht zu haben, der Herr?” Johann stand stramm und lüftete ganz leicht den schwarzen Zylinder auf seinem Kopf, als Frank Ludenhoff an ihm vorbeiging.
    “Guten Morgen, Johann. Danke der Nachfrage.” Frank Ludenhoff nickte Johan mit einem kaum merklichen Neigen des Kopfes zu. Er mochte den kleinen, rundlichen Portier, der schon seit mehr als 40 Jahren Morgen für Morgen pünktlich um halb acht an der goldenen Drehtür des Hotels stand und die Gäste empfing. “Heute kommt Gräfin Gurlitza, Johann, aber ich denke, Sie sind darauf eingestellt.”
    

## Implement Preprocessing Functions


### Lookup Table



```python
import numpy as np
from collections import Counter

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    counts = Counter(text)
    vocab = sorted(counts, key=counts.get, reverse=True)
    # Create dictionary that maps words to integers here
    vocab_to_int = {word: ii for ii, word in enumerate(vocab)}
    int_to_vocab = {i: word for word, i in vocab_to_int.items()}
    
    return (vocab_to_int, int_to_vocab)

```

    Tests Passed
    

### Tokenize Punctuation



```python
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    tokenize = {
        ".": "||period||",
        ",": "||comma||",
        "\"": "||quotation||",
        ";": "||semicolon||",
        "!": "||exclamation||",
        "?": "||question||",
        "(": "||left_parentheses||",
        ")": "||right_parentheses||",
        "--": "||dash||",
        "\n": "||return||",
    }
    return tokenize

```

    Tests Passed
    

## Preprocess all the data and save it
Running the code cell below will preprocess all the data and save it to file.


```python
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
```


```python
import helper
import numpy as np
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
```

## Build the Neural Network



```python
from distutils.version import LooseVersion
import warnings
import tensorflow as tf
```

    TensorFlow Version: 1.1.0
    

    UserWarning: No GPU found. Please use a GPU to train your neural network. [ipykernel_launcher.py:14]
    

### Input



```python
def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    # TODO: Implement Function
    input = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    return (input, targets, learning_rate)
```

    Tests Passed
    

### Build RNN Cell and Initialize



```python
def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    # TODO: Implement Function
    lstm_basic = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    cell = tf.contrib.rnn.MultiRNNCell([lstm_basic] * 1)
    init_state = cell.zero_state(batch_size, tf.float32)
    init_state = tf.identity(init_state, 'initial_state')

    return (cell, init_state)
```

    Tests Passed
    

### Word Embedding
Apply embedding to `input_data` using TensorFlow.  Return the embedded sequence.


```python
def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    return embed
```

    Tests Passed
    

### Build RNN



```python
def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, name='final_state')
    return (outputs, final_state)
```

    Tests Passed
    

### Build the Neural Network



```python
def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    embedding = get_embed(input_data, vocab_size, embed_dim)

    outputs, final_state = build_rnn(cell, embedding)

    logits = tf.contrib.layers.fully_connected(outputs, vocab_size,
                                               activation_fn=None)
    
    return (logits, final_state)
```

    Tests Passed
    

### Batches



```python
def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    int_text = np.array(int_text)
    # Get the number of characters per batch and number of batches we can make
    characters_per_batch = batch_size * seq_length
    n_batches = len(int_text)//characters_per_batch
    
    # Keep only enough characters to make full batches
    xdata = np.array(int_text[: n_batches * characters_per_batch])
    ydata = np.array(int_text[1: n_batches * characters_per_batch + 1])
    ydata[-1] = xdata[0]
    
    xdata = xdata.reshape((batch_size, -1))
    ydata = ydata.reshape((batch_size, -1))
    
    # Reshape into n_seqs rows
    inputs = np.split(xdata, n_batches, 1)
    targets = np.split(ydata, n_batches, 1)
    
    return np.array(list(zip(inputs, targets)))
```

    Tests Passed
    

## Neural Network Training
### Hyperparameters



```python
# Number of Epochs
num_epochs = 100 #100
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 256
# Embedding Dimension Size
embed_dim = 300
# Sequence Length
seq_length = 32
# Learning Rate
learning_rate = 0.01
# Show stats for every n number of batches
show_every_n_batches = 8

save_dir = './save'
```

### Build the Graph



```python
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)
```

## Train



```python

batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')
```

    Epoch   0 Batch    0/8   train_loss = 8.508
    Epoch   1 Batch    0/8   train_loss = 6.410
    Epoch   2 Batch    0/8   train_loss = 5.806
    Epoch   3 Batch    0/8   train_loss = 5.305
    Epoch   4 Batch    0/8   train_loss = 4.862
    Epoch   5 Batch    0/8   train_loss = 4.439
    Epoch   6 Batch    0/8   train_loss = 4.039
    Epoch   7 Batch    0/8   train_loss = 3.661
    Epoch   8 Batch    0/8   train_loss = 3.327
    Epoch   9 Batch    0/8   train_loss = 3.093
    Epoch  10 Batch    0/8   train_loss = 2.878
    Epoch  11 Batch    0/8   train_loss = 2.667
    Epoch  12 Batch    0/8   train_loss = 2.477
    Epoch  13 Batch    0/8   train_loss = 2.265
    Epoch  14 Batch    0/8   train_loss = 2.103
    Epoch  15 Batch    0/8   train_loss = 1.928
    Epoch  16 Batch    0/8   train_loss = 1.793
    Epoch  17 Batch    0/8   train_loss = 1.641
    Epoch  18 Batch    0/8   train_loss = 1.531
    Epoch  19 Batch    0/8   train_loss = 1.428
    Epoch  20 Batch    0/8   train_loss = 1.357
    Epoch  21 Batch    0/8   train_loss = 1.294
    Epoch  22 Batch    0/8   train_loss = 1.208
    Epoch  23 Batch    0/8   train_loss = 1.136
    Epoch  24 Batch    0/8   train_loss = 1.162
    Epoch  25 Batch    0/8   train_loss = 1.019
    Epoch  26 Batch    0/8   train_loss = 0.949
    Epoch  27 Batch    0/8   train_loss = 0.813
    Epoch  28 Batch    0/8   train_loss = 0.738
    Epoch  29 Batch    0/8   train_loss = 0.654
    Epoch  30 Batch    0/8   train_loss = 0.629
    Epoch  31 Batch    0/8   train_loss = 0.569
    Epoch  32 Batch    0/8   train_loss = 0.543
    Epoch  33 Batch    0/8   train_loss = 0.517
    Epoch  34 Batch    0/8   train_loss = 0.472
    Epoch  35 Batch    0/8   train_loss = 0.436
    Epoch  36 Batch    0/8   train_loss = 0.411
    Epoch  37 Batch    0/8   train_loss = 0.378
    Epoch  38 Batch    0/8   train_loss = 0.341
    Epoch  39 Batch    0/8   train_loss = 0.324
    Epoch  40 Batch    0/8   train_loss = 0.297
    Epoch  41 Batch    0/8   train_loss = 0.277
    Epoch  42 Batch    0/8   train_loss = 0.260
    Epoch  43 Batch    0/8   train_loss = 0.244
    Epoch  44 Batch    0/8   train_loss = 0.236
    Epoch  45 Batch    0/8   train_loss = 0.221
    Epoch  46 Batch    0/8   train_loss = 0.216
    Epoch  47 Batch    0/8   train_loss = 0.215
    Epoch  48 Batch    0/8   train_loss = 0.197
    Epoch  49 Batch    0/8   train_loss = 0.197
    Epoch  50 Batch    0/8   train_loss = 0.184
    Epoch  51 Batch    0/8   train_loss = 0.168
    Epoch  52 Batch    0/8   train_loss = 0.169
    Epoch  53 Batch    0/8   train_loss = 0.152
    Epoch  54 Batch    0/8   train_loss = 0.145
    Epoch  55 Batch    0/8   train_loss = 0.136
    Epoch  56 Batch    0/8   train_loss = 0.130
    Epoch  57 Batch    0/8   train_loss = 0.123
    Epoch  58 Batch    0/8   train_loss = 0.121
    Epoch  59 Batch    0/8   train_loss = 0.116
    Epoch  60 Batch    0/8   train_loss = 0.111
    Epoch  61 Batch    0/8   train_loss = 0.110
    Epoch  62 Batch    0/8   train_loss = 0.108
    Epoch  63 Batch    0/8   train_loss = 0.103
    Epoch  64 Batch    0/8   train_loss = 0.103
    Epoch  65 Batch    0/8   train_loss = 0.097
    Epoch  66 Batch    0/8   train_loss = 0.099
    Epoch  67 Batch    0/8   train_loss = 0.095
    Epoch  68 Batch    0/8   train_loss = 0.092
    Epoch  69 Batch    0/8   train_loss = 0.092
    Epoch  70 Batch    0/8   train_loss = 0.087
    Epoch  71 Batch    0/8   train_loss = 0.086
    Epoch  72 Batch    0/8   train_loss = 0.083
    Epoch  73 Batch    0/8   train_loss = 0.081
    Epoch  74 Batch    0/8   train_loss = 0.079
    Epoch  75 Batch    0/8   train_loss = 0.079
    Epoch  76 Batch    0/8   train_loss = 0.077
    Epoch  77 Batch    0/8   train_loss = 0.076
    Epoch  78 Batch    0/8   train_loss = 0.076
    Epoch  79 Batch    0/8   train_loss = 0.075
    Epoch  80 Batch    0/8   train_loss = 0.074
    Epoch  81 Batch    0/8   train_loss = 0.074
    Epoch  82 Batch    0/8   train_loss = 0.073
    Epoch  83 Batch    0/8   train_loss = 0.073
    Epoch  84 Batch    0/8   train_loss = 0.072
    Epoch  85 Batch    0/8   train_loss = 0.072
    Epoch  86 Batch    0/8   train_loss = 0.071
    Epoch  87 Batch    0/8   train_loss = 0.071
    Epoch  88 Batch    0/8   train_loss = 0.070
    Epoch  89 Batch    0/8   train_loss = 0.070
    Epoch  90 Batch    0/8   train_loss = 0.070
    Epoch  91 Batch    0/8   train_loss = 0.069
    Epoch  92 Batch    0/8   train_loss = 0.069
    Epoch  93 Batch    0/8   train_loss = 0.069
    Epoch  94 Batch    0/8   train_loss = 0.069
    Epoch  95 Batch    0/8   train_loss = 0.068
    Epoch  96 Batch    0/8   train_loss = 0.068
    Epoch  97 Batch    0/8   train_loss = 0.068
    Epoch  98 Batch    0/8   train_loss = 0.068
    Epoch  99 Batch    0/8   train_loss = 0.067
    Model Trained and Saved
    

## Save Parameters



```python
# Save parameters for checkpoint
helper.save_params((seq_length, save_dir))
```

# Checkpoint


```python
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()
```

## Implement Generate Functions
### Get Tensors



```python
def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    # TODO: Implement Function
    InputTensor = loaded_graph.get_tensor_by_name("input:0")
    InitialStateTensor = loaded_graph.get_tensor_by_name("initial_state:0")
    FinalStateTensor = loaded_graph.get_tensor_by_name("final_state:0")
    ProbsTensor = loaded_graph.get_tensor_by_name("probs:0")

    return InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor
```

    Tests Passed
    

### Choose Word



```python
def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    # TODO: Implement Function
    
    word_index = np.random.choice(len(int_to_vocab), 1, p=probabilities)
    #print(word_index)
    return int_to_vocab[word_index[0]]
```

    Tests Passed
    

## Generate the Groschenroman



```python
gen_length = 500
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'sandrine'

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word]
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})
        
        pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)
    
    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
        
    print(tv_script)
```

    INFO:tensorflow:Restoring parameters from ./save
    sandrine herrn da, dass der vater carl bezichtigt, die familie in verruf zu bringen und jeglichen kontakt zu seinem sohn verweigert. langsam und behutsam hatte die mutter wie ein blitz.
    “komm unbedingt ein! ”
    “du glaubst ja, freifrau, sibille! auf die floristin überhörte elisabeths arroganten und zurechtweisenden ton lief. es war schön, dich zu sehen. wie geht es ihnen? du hast mir nicht. ”
    sibille blieb der mund offen stehen. “meine mutter hieß auch lydia”, sagte er und sah seine augen, und nicht von nie. dann ist einfach doch nicht. ” er lachte. “auch für ehemalige….! ”
    kastlhuber verstand frank’ seitenhieb offenbar, denn sein mit einem kleinen hammer traktierte.
    “ach, ich bin nur da, herr ludenhoff. das dunkle haar, zum lässigen knoten aufgeschlungen, die feine nase und der knitze zug um. es war während meiner ausbildungszeit. er war journalist oder zumindest wollte er einer werden. ein klassischer, das ist monika eine andere frau oder aber waren an mich jetzt gehen? ”
    “prima, danke mein junge. die grippe habe ich gottlob gut überstanden. jetzt habe ich großen appetit auf ein herzhaftes frühstück. ” frank lachte sie. jetzt erst hatte er die echten freundin geworden. er war journalist oder zumindest wollte er einer werden. ein klassischer, das war es ein? ”
    
    sibille sah ihren vater an. sie wusste, dass er recht hatte. sie sah ihn nicht, sie für ihrn höchstens eine affäre frank wurde. sie war unfähig, einen klaren gedanken zu fassen, geschweige denn einen vernünftigen satz zu sagen. sie sah seine augen und gähnte sie. sie sah ihn an und ihr blick war eiskalt.
    “wenn du mich verlässt, erfährt du wirst eine echte adlige. da darfst du gewisse gesellschaftliche konventionen nicht außer acht lassen. gräfin zu jägermeinhaus nicht einzuladen, käme einem affront gleich. ich habe eben dem gast von 215 eine zeitung gebracht und dachte mir, ich schaue mal, ob sie heute dienst haben. ”
    
    sie bekamen einen schönen tisch am fenster und sebastiàn war wie immer frank in die augen, dachte er, die sofort in die augen. in erwartung des schlags duckte sich sibille, obwohl er völlig vergessen, dass er sich auf, drehte sich um. sein sehnsüchtiger blick ging ihr durch und durch. es drängte sie, ihn zu küssen und alle vorbehalte habe ich die kopien. noch nie im restaurant des ist. er war. aber insgeheim hatte er mehr rücksichtnahme von ihr erwartet. die aber nach dem moment war so noch einen kurzen minuten vorgestellt zeit, was der immer. ”
    “du hast angst,
    

THE END
