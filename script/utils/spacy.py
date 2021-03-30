from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet

import spacy
from spacy import displacy
from nltk import Tree

import sklearn
from sklearn import preprocessing

import numpy as np
from numpy import array
from numpy import argmax
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


#===== Download SpaCy models =====

# Load the spacy model for English
en_nlp = spacy.load("en_core_web_sm")
#en_nlp = spacy.load("en_core_web_trf")

# The labels for the model are stored in a dict
labels = en_nlp.pipe_labels
labels.keys()

labels

POS = labels["tagger"]
len(POS) # 49
Dependencies = labels['parser']
len(Dependencies) # 45
NER = labels["ner"]
len(NER) # 18

### Get dummy embeddings
POS_embed = pd.get_dummies(POS)
POS_embed.head(5)
#POS_embed["DET"]
Dep_embed = pd.get_dummies(Dependencies)
Dep_embed.head(5)


#===== Functions =====

# https://gaurav5430.medium.com/using-nltk-for-lemmatizing-sentences-c1bfff963258
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def get_wn_tag(sentence, target_word = None):
    tokens = word_tokenize(sentence)
    tag_list = pos_tag(tokens)
    if target_word == None:
        wn_tags = [(pair[0], nltk_tag_to_wordnet_tag(pair[1])) for pair in tag_list]
    if target_word != None and target_word in tokens:
        wn_tags = [(pair[0], nltk_tag_to_wordnet_tag(pair[1])) for pair in tag_list if pair[0] == target_word]
    return wn_tags

def get_nltk_tags(sentence, target_word = None):
    tokens = word_tokenize(sentence)
    tag_list = pos_tag(tokens)
    if target_word != None and target_word in tokens:
        tag_list = [(pair[0], pair[1]) for pair in tag_list if pair[0] == target_word]
    return tag_list

def get_syntax(sentence, prnt = False):
    doc = en_nlp(sentence)
    sent = next(doc.sents)
    syntax_list = [(word, word.dep_) for word in sent]
    if prnt == True:
        for i in syntax_list:
            print(str(i[0]),"    : ",i[1])
    return(syntax_list)

def detailed_sntx(sentence):
    doc = en_nlp(sentence)
    for token in doc:
        print("\n\n-----", token.text,"-----\n",
                " -Lemma:", token.lemma_, "\n",
                " -PoS:", token.pos_,"\n",
                " -Tag:", token.tag_,"\n",
                " -Dependencies:", token.dep_) #,"\n",
                #" -Shape:", token.shape_,"\n",
                #" -is_alpha:", token.is_alpha,"\n",
                #" -is_stop:", token.is_stop)
                # the _ after variable names is to convert from hash

def get_syntax_vectors(sentence, dimensions):
    doc = en_nlp(sentence)

    list_out_array = []

    for token in doc:

        pos_vector = list(np.transpose(POS_embed[token.tag_]))

        dep_vector = list(np.transpose(Dep_embed[token.dep_]))

        syntax_vector = pos_vector + dep_vector

        while len(syntax_vector) < dimensions:
            syntax_vector = syntax_vector + [0]

        list_out_array.append(syntax_vector)

    out_array = np.array(list_out_array)

    return out_array




#===== Sentences to process =====
sent1 = "The bark of the wooden dog was crunchy."
sent2 = "The bark of wooden dog was old and dry."
sent3 = "The bass line could be heard from the boat as it was sailing out of the harbor."

sent4 = "Have you permitted it to become a giveaway program rather than one that has the goal of improved employee morale and, consequently, increased productivity?"
sent5 = "On April 17, 1610, the sturdy little three masted bark, Discovery, weighed anchor in St. Katherine's Pool, London, and floated down the Thames toward the sea."
sent6 = "The dentist prescribed a root procedure."

sent7 = "The man bought the soda for Jane."

sent8 = "The big green fire truck."
sent9 = "The keys to the cabbinet are on the table."
sent10 = "The sailor dogs the hatch."
sent11 = "The horse raced past the barn fell."
sent12 = "Doctor helps dog bite victim."
sent13 = "The sailors man the boat."
sent14 = "The old man the boat."
sent15 = "Man the car!"
sent16 = "Man the car."

# https://survivingenglish.wordpress.com/2017/10/16/garden-path-sentences-or-when-the-old-man-the-boat-the-young-duck-the-oars/
sent17 = "I convinced her children are noisy."
sent18 = "Mary gave the child the dog bit a band-aid."
sent19 = "The girl told the story cried."
sent20 = "The man who hunts ducks out on weekends."
sent21 = "The cotton clothing is usually made of grows in Mississippi."
sent22 = "Fat people eat accumulates."


sent23 = "bark the tree."
sent24 = "The tree bark."



used_sent = sent23


get_wn_tag(sentence = used_sent)
get_nltk_tags(used_sent)
get_syntax(used_sent)
detailed_sntx(used_sent)

# explain() gives the full name of dependencies etc.
spacy.explain("PDT")


for sent in [sent13, sent14, sent15, sent16]:
    print("\n\n===============")
    print("===============")
    print(sent)
    detailed_sntx(sent)



"""
===== Notes =====
It seems that garden path sentences are too hard for SpaCy to parse.
PoS taggers have not surpassed humans in this regard.
Is it a problem?
- Yes!
  - Could give the wrong POS tag in complex sentences.
- Yes, but...
  - The model hopefully learns to model uncertainty in the signal.
  - POS is still more accurate than WSD.
  - Garden path sentence parsing is a separate problem from WSD.
  - SpaCy seems to work just fine in non-garden path sentences.
  - Thus, the information is probably reliable most of the time.

These points stand in relation to dependencies from SpaCy as well.

PoS: coarse-grained PoS
Tag: fine-grained PoS

Since WordNet is coarse-grained regarding PoS, PoS should suffice.

Dependencies might be useful:
- There might be sligt biases between senses
  - Chicken (meat) might be slightly biased towards being an object
    compared to chicken (animal).
    - Both senses can occcur in all roles, so not an absolute rule.
    - "The chicken (meat) spoiled.", "He fed the chicken (animal)."

"""

doc = en_nlp(used_sent)

token = doc[1]
token.ner
displacy.serve(doc, style='dep')
displacy.serve(doc, style='ent')


#############################
#===== One Hot encoding =====
#############################


used_sent = "The dog chased the cat."

syntax_matrix = get_syntax_vectors(used_sent, 150)

syntax_matrix.shape

syntax_matrix[1, ] # Row is one word
syntax_matrix[:,0] # column is one variable across all words

# When tokenizing with BERT, each subword in a word gets the same syntax vector


