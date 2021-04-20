import nltk

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import argparse

WORDNET_POS = {'VERB': wn.VERB, 'NOUN': wn.NOUN, 'ADJ': wn.ADJ, 'ADV': wn.ADV}


def _get_info(lemma, pos, info_type):
    results = dict()

    wn_pos = WORDNET_POS[pos] if pos is not None else None  # None seems to extract all entries.
    morphemes = wn._morphy(lemma, pos=wn_pos) if pos is not None else []
    for i, synset in enumerate(set(wn.synsets(lemma, pos=wn_pos))):
        sense_key = None
        for l in synset.lemmas():
            if l.name().lower() == lemma.lower():
                sense_key = l.key()
                break
            elif l.name().lower() in morphemes:  # synsets group synonyms, so this part is needed when the synonyms, which didn't give name to the synset is encountered.
                sense_key = l.key()
        assert sense_key is not None
        results[sense_key] = synset.examples() if info_type == 'examples' else synset.definition()

    return results


def get_glosses(lemma, pos):
    return _get_info(lemma, pos, info_type='gloss')


def get_example_sentences(lemma, pos):
    return _get_info(lemma, pos, info_type='examples')


def get_all_wordnet_lemma_names():
    results = []
    for pos, wn_pos in WORDNET_POS.items():
        results.append((pos, wn.all_lemma_names(pos=wn_pos)))

    return results


def _get_gloss_extension(key, _lemma, conversion_dict, args):
    # synset = wn.synset_from_sense_key(key) # <-- This is flawed, use below code instead.
    synset = wn.lemma_from_key(key).synset()

    #print(synset)
    #print(f"synset made for: {key}".format(key))
    pos = str(synset.pos())
    if conversion_dict != None:
        pos = conversion_dict[pos]
        #print("alternative pos made")
    out_tuple = (_lemma,)
    if args.use_gloss_extensions:
        out_tuple += (pos, "[PAD]")
    return out_tuple


# For playing around with functions.
if __name__ == '__main__':
    conv_dict = {
        "n": "[NOUN]",
        "v": "[VERB]",
        "a": "[ADJ]",
        # Satellite adjective, adjective for my purposes
        "s": "[ADJ]",
        "r": "[ADV]"
    }
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_pos_tags",
        action='store_true',
        help="Whether to add POS to the data."
    )

    parser.add_argument(
        "--use_dependencies",
        action='store_true',
        help="Whether to add dependencies to the data."
    )

    args = parser.parse_args()

    #print(wn.get_version())

    _lemma = "quick"
#    info = _get_info(_lemma, None, 'gloss')
#    [print(key, value) for key, value in info.items()]
#    keys = list(info.keys())
    #print(keys)
#    extensions = [_get_gloss_extension(key, _lemma, conv_dict, args) for key in keys]
#    print(extensions)
    #print(wn.synset_from_sense_key("permit%2:32:06::"))
    # keys = info.keys()
    #print(wn.lemma_from_key(key))
    #lemma = wn.lemma_from_key(key)
    #synset_key = wn.synset_from_sense_key(key)
    #print(synset_key)
    #print(synset_key.lemma_names())
    #print(synset_key.pos())

    #WORDNET_POS = {'VERB': wn.VERB, 'NOUN': wn.NOUN, 'ADJ': wn.ADJ, 'ADV': wn.ADV}

    #
    noun_dict = get_glosses(_lemma, "NOUN")
    verb_dict = get_glosses(_lemma, "VERB")
    adj_dict = get_glosses(_lemma, "ADJ")
    adv_dict = get_glosses(_lemma, "ADV")
    sense_info = {**noun_dict, **verb_dict, **adj_dict, **adv_dict}

    [print(key, value) for key, value in sense_info.items()]


    # print(wn.synset_from_sense_key(info.keys()[0]))
    # snst = wn.synset_from_sense_key(keys[0])
    # print(snst)
    # for word in ["chomp", "quick"]:
    #    print(word)
    #    for pos in ["ADJ", "ADV", "NOUN", "VERB", None]:
    #        test = _get_info(word, pos, "gloss")
    #        print(pos)
    #        [print(key, ":", value) for key, value in test.items()]
    #    print("########################################")
    #    print("\n")

    # sense_info = get_glosses('bark', None)
    # sense_info = {}
    # for word in ["chomp", "quick", "rock"]:
    #    for pos_of_gloss in ["ADJ", "ADV", "NOUN", "VERB"]:
    #        sense_info_sub = get_glosses(word, pos_of_gloss)
    #        #print(pos_of_gloss)
    #        #print(sense_info_sub)
    #        if len(sense_info_sub) != 0:
    #            gloss_word_and_pos = (word, "[" + pos_of_gloss + "]")
    #            #sense_info = {key:[gloss_word_and_pos, value] for key, value in sense_info_sub.items()}
    #            for key, value in sense_info_sub.items():
    #                sense_info[key] = [gloss_word_and_pos, value]

    # print("Big dictionary")
    # [print(key, value) for key, value in sense_info.items()]
    # print()

    # print(sense_info)

    # for pos in ["ADJ", "ADV", "NOUN", "VERB", None]:
