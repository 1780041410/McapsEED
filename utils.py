import numpy as np
import collections
def load_pretrain_emb(embedding_path, words):
    word_index={"<PAD>":0}
    for word in words:
        word_index[word]=len(word_index)

    print('len(word_index)', len(word_index))
    max_features = len(word_index)
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    with open(embedding_path, encoding="utf8", errors='ignore') as f:
        embeddings_index = dict(get_coefs(*o.strip().split(" ")) for o in f if
                                o.split(" ")[0] in word_index)
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    print('finish!')
    embedding_matrix[0]=np.asarray([0]*100, dtype='float32')
    return embedding_matrix,word_index
def load_description(description_path,len):
    entity2id = {}
    words = set()
    entity2d = []
    entity2desp = {}
    entity = []
    max_len = 0
    with open('data/FB15k-237/entity2id.txt') as file:
        for line in file:
            tmp = line.split()
            entity2id[tmp[0]] = tmp[1]

    with open(description_path, 'r') as file:
        for line in file:
            count = 0
            tmp_desp = []
            line = line.replace('\\n',' ')
            tmp = line.split()
            e = tmp[0]
            for w in range(1,tmp.__len__()):
                if w == 1:
                    word = tmp[w].replace('"','')
                else:
                    word = tmp[w].replace('\\"','')
                    word = word.replace(',','')
                    word = word.replace('.', '')
                    word = word.replace('"@en', '')
                words.add(word.lower())
                tmp_desp.append(word.lower())
                count += 1
                if count >= len:
                    break
            if count > max_len:
                max_len = count
            entity2d.append(tmp_desp)
            entity.append(e)

    for i, e in enumerate(entity):
        entity2desp[e] = entity2d[i]
    entity2desp = pad_sequence(entity2desp, 100)
    return words, entity2desp, max_len

def pad_sequence(entity2desp, max_len):
    for e in entity2desp:
        tmp_arr = entity2desp[e]
        if len(tmp_arr) < max_len:
            tmp_arr.extend(['<PAD>'] * (max_len - len(tmp_arr)))
            entity2desp[e] = tmp_arr
    return entity2desp
def get_ent_pos():#data/FB15k
    file_desp = open('data/FB15k-237/FB15k_description.txt', 'r')
    file_name = open('data/FB15k-237/FB15k_mid2name.txt', 'r')
    ent_pos = {}
    ent_name = {}
    max_len = 0
    count = 0
    for line in file_name:
        tmp = line.split()
        ent_name[tmp[0]] = tmp[1].lower()
    for line in file_desp:
        sen = line.replace('"', '')
        sen = sen.replace(',', '')
        sen = sen.replace('\\\\','')
        sen_arr = sen.split()
        tmp = ''
        for i in range(1,len(sen_arr)):
            tmp += sen_arr[i] + " "
        tmp = tmp.lower()
        ent = ent_name[sen_arr[0]]
        ent = ent.split('_')
        sen = tmp.split()

        if ent[0] in sen:
            begin = sen.index(ent[0])
        elif ent[-1] in sen:
            begin = sen.index(ent[-1])

        if ent[-1] in sen:
            end = sen.index(ent[-1])
        elif ent[0] in sen:
            end = sen.index(ent[0])
        if end - begin > 18:
            end = begin
            count += 1
        ent_pos[sen_arr[0]] = [begin,end]
        if end - begin + 1 > max_len:
            max_len = end - begin + 1

    file_desp.close()
    file_name.close()

    return ent_pos, max_len
def get_fb15k_237_desc_to_file(file=None,path=None):
    entity_des=collections.OrderedDict()
    with open('data/FB15k/FB15k_description.txt') as f:
        for line in f.readlines():
            e,desc=line.strip().split('\t')
            entity_des[e]=desc
    with open(file) as file,\
        open(path,'w') as fw:
        for line in file:
            tmp = line.split()
            try:
                fw.write(tmp[0]+'\t'+entity_des[tmp[0]]+'\n')
            except:
                fw.write(tmp[0] + '\t' +"<PAD>" + '\n')
def get_fb15k_237_name_to_file(file=None,path=None):
    entity_name = collections.OrderedDict()
    with open('data/FB15k/FB15k_mid2name.txt') as f:
        for line in f.readlines():
            e, name = line.strip().split('\t')
            entity_name[e] = name
    with open(file) as file, \
            open(path, 'w') as fw:
        for line in file:
            tmp = line.split()
            try:
                fw.write(tmp[0] + '\t' + entity_name[tmp[0]] + '\n')
            except:
                fw.write(tmp[0] + '\t' + "<PAD>" + '\n')
if __name__ == "__main__":
    # get_fb15k_237_name_to_file('data_capsnet/FB15k-237/entity2id.txt',\
    #                            'data_capsnet/FB15k-237/FB15k_237_mid2name.txt')
    # get_fb15k_237_desc_to_file('data_capsnet/FB15k-237/entity2id.txt','data_capsnet/FB15k-237/FB15k_237_description.txt')
    #get_ent_pos()
    wiki_words, entity2wiki, max_len1 = load_description("data/FB15k-237/FB15k_237_description.txt",100)
    #{所有单词--->w1,w2}
    #{实体:[单词]}
    embedding_matrix, word2id = load_pretrain_emb('data/FB15k-237/glove.6B.100d.txt', wiki_words)
    entity2wikiID={}
    for entity,words in entity2wiki.items():
        temp = entity2wikiID.get(entity, [])
        for word in words:
            temp.append(word2id[word])
        entity2wikiID[entity]=temp
    print(len(entity2wikiID))



