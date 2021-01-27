from nltk.corpus import wordnet

def word_info(word):
       syn = wordnet.synsets(word)
       for i in range(len(syn)):
              print("=====================================")
              print(word, i)
              syn_i = syn[i]
              print("PoS-tag: ", syn_i.pos())
              print ("Synset name:  ", syn_i.name())
              print ("Synset definiton: ", syn_i.definition())
              print("Other words in synset (synonyms):")
              lst=[print("-",lemma) for lemma in syn_i.lemma_names() if lemma != word]
              if len(lst) == 0:
                     print("- No synonyms found")
              del lst
              if syn_i.hypernyms() != []:
                     print ("Hypernymic synset:  ", syn_i.hypernyms()[0].name())
                     print("Synset(s) in hypernymic synset:")
                     lst=[print("-",same_hyper.name()) for same_hyper in syn_i.hypernyms()[0].hyponyms() if same_hyper != syn_i]
                     del lst
              else:
                     print("Hypernymic synset: ")
                     print("- No hypernymic synset found")
              print("Hyponymic synset(s):")
              lst=[print("-",hypo.name()) for hypo in syn_i.hyponyms()]
              if lst == []:
                     print("- No hyponymic synsets found")
              del lst
              print("=====================================")
              print("\n\n")

word_info("research")


