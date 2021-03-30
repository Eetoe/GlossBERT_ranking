import nltk

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
from nltk.corpus import wordnet as wn

WORDNET_POS = {'VERB': wn.VERB, 'NOUN': wn.NOUN, 'ADJ': wn.ADJ, 'ADV': wn.ADV}


def _get_info(lemma, pos, info_type):
    results = dict()

    wn_pos = WORDNET_POS[pos] if pos is not None else None # None seems to extract all entries.
    morphemes = wn._morphy(lemma, pos=wn_pos) if pos is not None else []
    for i, synset in enumerate(set(wn.synsets(lemma, pos=wn_pos))):
        sense_key = None
        for l in synset.lemmas():
            if l.name().lower() == lemma.lower():
                sense_key = l.key()
                break
            elif l.name().lower() in morphemes: # synsets group synonyms, so this part is needed when the synonyms, which didn't give name to the synset is encountered.
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

# For playing around with functions.
if __name__ == '__main__':
    #for word in ["chomp", "quick"]:
    #    print(word)
    #    for pos in ["ADJ", "ADV", "NOUN", "VERB", None]:
    #        test = _get_info(word, pos, "gloss")
    #        print(pos)
    #        [print(key, ":", value) for key, value in test.items()]
    #    print("########################################")
    #    print("\n")

    #sense_info = get_glosses('bark', None)
    sense_info = {}
    for word in ["chomp", "quick", "rock"]:
        for pos_of_gloss in ["ADJ", "ADV", "NOUN", "VERB"]:
            sense_info_sub = get_glosses(word, pos_of_gloss)
            #print(pos_of_gloss)
            #print(sense_info_sub)
            if len(sense_info_sub) != 0:
                gloss_word_and_pos = (word, "[" + pos_of_gloss + "]")
                #sense_info = {key:[gloss_word_and_pos, value] for key, value in sense_info_sub.items()}
                for key, value in sense_info_sub.items():
                    sense_info[key] = [gloss_word_and_pos, value]


    print("Big dictionary")
    [print(key, value) for key, value in sense_info.items()]
    #print()

    #print(sense_info)

    #for pos in ["ADJ", "ADV", "NOUN", "VERB", None]:

