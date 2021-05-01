import nltk

def bleuscorer(hyps, refs):
    #print(hyps, refs)
    bleu = []
    for hyp, ref in zip(hyps, refs):
        hyp = hyp.split()
        ref = [a.split() for a in ref]
        #hyp = nltk.word_tokenize(hyp)
        #ref = [nltk.word_tokenize(a) for a in ref]
        bleu += [nltk.translate.bleu_score.sentence_bleu(ref, hyp)]
    return sum(bleu) / len(bleu)

if __name__ == '__main__':
    print(bleuscorer(['the the the the the the the', 'there is a cat', 'it is'], [["the cat is on the mat", "there is a cat on the mat"], ["there is a cat on the mat"], ["it is true"]]))
