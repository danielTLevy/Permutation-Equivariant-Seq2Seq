from RDRPOSTagger.InitialTagger.InitialTagger import initializeCorpus, initializeSentence
from RDRPOSTagger.SCRDRlearner.Object import FWObject
from RDRPOSTagger.SCRDRlearner.SCRDRTree import SCRDRTree
from RDRPOSTagger.SCRDRlearner.SCRDRTreeLearner import SCRDRTreeLearner
from RDRPOSTagger.Utility.Config import NUMBER_OF_PROCESSES, THRESHOLD
from RDRPOSTagger.Utility.Utils import getWordTag, getRawText, readDictionary
from RDRPOSTagger.Utility.LexiconCreator import createLexicon
from multiprocessing import Pool

def unwrap_self_RDRPOSTagger(arg, **kwarg):
    return RDRPOSTagger.tagRawSentence(*arg, **kwarg)

class RDRPOSTagger(SCRDRTree):
    """
    RDRPOSTagger for a particular language
    """

    def __init__(self):
        self.root = None

    def tagRawSentence(self, DICT, rawLine):
        line = initializeSentence(DICT, rawLine)
        sen = []
        wordTags = line.split()
        for i in range(len(wordTags)):
            fwObject = FWObject.getFWObject(wordTags, i)
            word, tag = getWordTag(wordTags[i])
            node = self.findFiredNode(fwObject)
            if node.depth > 0:
                sen.append(word + "/" + node.conclusion)
            else:  # Fired at root, return initialized tag
                sen.append(word + "/" + tag)
        return " ".join(sen)

    def tagRawCorpus(self, DICT, rawCorpusPath):
        lines = open(rawCorpusPath, "r").readlines()
        # Change the value of NUMBER_OF_PROCESSES to obtain faster tagging process!
        pool = Pool(processes=NUMBER_OF_PROCESSES)
        taggedLines = pool.map(unwrap_self_RDRPOSTagger, zip([self] * len(lines), [DICT] * len(lines), lines))
        outW = open(rawCorpusPath + ".TAGGED", "w")
        for line in taggedLines:
            outW.write(line + "\n")
        outW.close()
        print("\nOutput file: " + rawCorpusPath + ".TAGGED")

french_tagger = RDRPOSTagger()
french_tagger.constructSCRDRtreeFromRDRfile("../Models/POS/French.RDR")
frenchDICT = readDictionary("../Models/POS/French.DICT")

english_tagger = RDRPOSTagger()
english_tagger.constructSCRDRtreeFromRDRfile("../Models/POS/English.RDR")
englishDICT = readDictionary("../Models/POS/English.DICT")