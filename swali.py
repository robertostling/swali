import sys
import os

import mpfile
import empfile

def create_lexicon(src_filename, trg_filename):
    src_empf_filename = src_filename + '.empf'
    trg_empf_filename = trg_filename + '.empf'

    if not (os.path.exists(src_empf_filename) and
            os.path.exists(trg_empf_filename)):
        src_mpf = mpfile.MPFile(src_filename, has_ids=False)
        trg_mpf = mpfile.MPFile(trg_filename, has_ids=False)
        assert len(src_mpf.sentences) == len(trg_mpf.sentences)
        sent_ids = sorted(src_mpf.sentences.keys())
        src_mpf.write_numpy(src_empf_filename, sent_ids)
        trg_mpf.write_numpy(trg_empf_filename, sent_ids)

    src_empf = empfile.EncodedMPF(src_empf_filename)
    trg_empf = empfile.EncodedMPF(trg_empf_filename)

    src_empf.make_ngrams()
    trg_empf.make_ngrams()
    src_empf.count_ngrams()
    trg_empf.count_ngrams()
    src_empf.make_ngram_positions(set(range(len(src_empf.ngram_list))))
    trg_empf.make_ngram_positions(set(range(len(trg_empf.ngram_list))))

    lexicon = {}
    for trg_ngram_i, trg_ngram in enumerate(trg_empf.ngram_list):
        trg_verses = [
                verse_i for verse_i, _, _
                in trg_empf.ngram_positions[trg_ngram_i]]
        real_trg_verses, (result, count) = src_empf.find_ngrams_from_verses(
                trg_verses)
        scores = []
        n_ngrams = len(src_empf.ngram_list)
        for src_ngram_i, n_both in zip(result, count):
            #print(src_ngram_i, n_both, count)
            n_src = src_empf.ngram_verse_count[src_ngram_i]
            n_trg = trg_empf.ngram_verse_count[trg_ngram_i]
            score = empfile.betabinomial_similarity(
                    src_empf.n_verses, n_both, n_src, n_trg, n_ngrams)
            scores.append((score, src_empf.ngram_list[src_ngram_i]))
        scores.sort(reverse=True, key=lambda t: t[0]*len(t[1]))
        if scores and scores[0][0] > 0:
            lexicon[trg_ngram] = scores[0]

    return lexicon


def iterate_ngrams(s):
    s = f'#{s}#'
    n = len(s)
    for i in range(n-1):
        for j in range(i+1, n+1):
            yield s[i:j]


def translate(token, lexicon):
    candidates = [lexicon[ngram] for ngram in iterate_ngrams(token)
                  if ngram in lexicon]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[0]*len(t[1]))[1]


def main():
    src_filename, trg_filename, test_filename = sys.argv[1:]
    lexicon = create_lexicon(src_filename, trg_filename)
    with open(test_filename) as f:
        for line in f:
            translated = [translate(token, lexicon) for token in line.split()]
            print(' '.join([token for token in translated if token]))


if __name__ == '__main__':
    main()

