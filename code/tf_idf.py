from collections import Counter, OrderedDict
import itertools
import math
from pprint import pprint
import sys

documents = '''\
復古單寧帆布後雙背包
米色帆布橫式側背包
藍色石紋斜背包
帆布眼鏡盒
鱷魚帆布筆袋
拼木紋布質文庫書衣
單眼相機用亮彩防水包布
'''.splitlines()


class OrderedCounter(Counter, OrderedDict):
    def __repr__(self):
        return '%s(%r)' % (
            self.__class__.__name__,
            OrderedDict(self)
        )

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

def oc_to_str(oc):
    return '{' + ', '.join('%s: %.2f' % (k, v) for k, v in oc.items()) + '}'

def norm(vec):
    return math.sqrt(sum(elem ** 2 for elem in vec.values()))

def make_tf_idf(documents):
    tf = [OrderedCounter(doc) for doc in documents]
    idf = dict()
    uniq_term_per_doc = itertools.chain.from_iterable(map(set, documents))
    for t, delta_t in Counter(uniq_term_per_doc).items():
        idf[t] = 1 + math.log(len(documents) / (delta_t + 1))
        # idf[t] = math.log2((len(tf) + 1) / delta_t)
    return tf, idf

def make_d_vec(tf, idf):
    d_vec = []
    for d_doc in tf:
        prod = map(lambda t, tf: tf * idf[t], d_doc.keys(), d_doc.values())
        od = OrderedDict(zip(d_doc.keys(), prod))
        d_vec.append(od)
    return d_vec

def search(query, d_vec):
    q = OrderedCounter(query)
    q_norm = norm(q)
    cos_theta = []
    for d in d_vec:
        dq = sum(map(lambda t, tf: d.get(t, 0) * tf, q.keys(), q.values()))
        d_norm = norm(d)
        cos_val = dq / (d_norm * q_norm)
        cos_theta.append(cos_val)
    return cos_theta

if __name__ == '__main__':
    query = sys.argv[1]
    print('Searching %s in %d documents ...' % (query, len(documents)))
    pprint(documents)
    tf, idf = make_tf_idf(documents)
    d_vec = make_d_vec(tf, idf)
    cos_theta = search(query, d_vec)
    for doc_id, (d, cos_val) in enumerate(zip(d_vec, cos_theta), 1):
        print("d%d = %s, similarity = %.4f" % (doc_id, oc_to_str(d), cos_val))

    top_similar_doc_ids = sorted(
        enumerate(cos_theta), key=lambda t: t[1], reverse=True
    )[:3]
    print("Top 3 similar documents are:")
    for doc_id, cos_val in top_similar_doc_ids:
        print("d%d: %s [sim=%.4f]" %
              (doc_id + 1, documents[doc_id], cos_val))
