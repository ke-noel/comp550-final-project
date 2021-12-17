import gensim
import re
from nltk.tokenize import sent_tokenize

# from https://towardsdatascience.com/document-summarization-using-latent-semantic-indexing-b747ef2d2af6
# and https://medium.com/betacom/latent-semantic-indexing-in-python-85880414b4de

def get_top_sentences(sum_sentences, num_topics, sorted_vectors):
    top_sentences = []
    s_num = []
    s_index = set()
    s_count = 0
    for i in range(sum_sentences):
        for ti in range(num_topics):
            v = sorted_vectors[ti]
            si = v[i][0]
            if si not in s_index:
                s_num.append(si)
                top_sentences.append(v[i])
                s_index.add(si)
                s_count += 1
                if s_count == sum_sentences:
                    return top_sentences


def lsa_summarize(document, reference, stopwords, extra):
    sum_sentences = len(sent_tokenize(reference))
    sentences = sent_tokenize(document)
    sentences_clean = [re.sub(r'[^\w\s]', '', s.lower()) for s in sentences]

    if stopwords:
        sentence_tokens = [[w for w in s.split(' ') if w not in stopwords] for s in sentences_clean]
    else:
        sentence_tokens = [s.split(' ') for s in sentences_clean]

    corpus_dict = gensim.corpora.Dictionary(sentence_tokens)
    bow_corpus = [corpus_dict.doc2bow(s) for s in sentence_tokens]
    tfidf = gensim.models.TfidfModel(bow_corpus, smartirs='npu')
    corpus_tfidf = tfidf[bow_corpus] 

    lsa = gensim.models.LsiModel(corpus_tfidf, id2word=corpus_dict, num_topics=5)
    corpus_lsi = lsa[corpus_tfidf]

    num_topics = len(lsa.print_topics())

    vectors = list(map(lambda _: list(), range(num_topics)))
    for i, doc_vec in enumerate(corpus_lsi):
        for s in doc_vec:
            vectors[s[0]].append((i, abs(s[1])))
    sorted_vectors = list(map(lambda x: sorted(x, key=(lambda y: y[1]), reverse=True), vectors))
    
    try:
        top_sentences = get_top_sentences(sum_sentences, num_topics, sorted_vectors)
    except:
        return []

    i = 0 
    summary = []
    for i,sentence in enumerate(sentences):
        if i in [c[0] for c in top_sentences]:
            summary.append(sentence)
    
    return ' '.join(summary)