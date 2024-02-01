import re
from collections import Counter

class Splitwords:

    def __init__(self):
        self.dictionary = Counter(self.words(open('./data/words.txt').read()))
        self.max_word_length = max(map(len, self.dictionary))
        self.total = float(sum(self.dictionary.values()))

    def viterbi_segment(self,text,filter=True):
        probs, lasts = [1.0], [0]
        for i in range(1, len(text) + 1):
            prob_k, k = max((probs[j] * self.word_prob(text[j:i]), j) for j in range(max(0, i - self.max_word_length), i))
            probs.append(prob_k)
            lasts.append(k)
        words = []
        i = len(text)
        while 0 < i:
            if filter and (len(text[lasts[i]:i]))>1: 
                words.append(text[lasts[i]:i])
            elif not filter:
                words.append(text[lasts[i]:i])
            i = lasts[i]
        words.reverse()
        return words, probs[-1]
    
    def split_words(self,word):
        words=[]
        index=0
        if "_" in word:
            index=1
            words=word.lower().split("_")
        elif " " in word:
            index=2
            words=word.lower().split(" ")
        elif bool(re.match(r'[a-zA-Z][^A-Z]*', word)):
            index=3
            words = re.findall(r'[a-zA-Z][^A-Z]*', word)
        else:
            index=4
            words=[word]
        # print(f"{word} ({index}): {words}")
        words=[x.lower() for x in words if not x.isdigit() and len(x)>1]
        return words if len(words)>0 else [word.lower()]
    
    def splitwords2(self,word):
        word=word.lower()
        if word=="ince":
            return ["ince"]
        if word=="code2":
            return ["code"]
        with open('./data/list.txt') as file:
            word_list=sorted(file.read().split('\n'),key=lambda x:-len(x))
        words=[]
        dup_words=[]
        for i in word[::-1]:
            if i in ["_","/"]:
                if ''.join(dup_words[::-1]) in word_list:
                    words.append(''.join(dup_words[::-1]))
                dup_words=[]
                continue
            dup_words.append(i)
            if ''.join(dup_words[::-1]) in word_list:
                words.append(''.join(dup_words[::-1]))
                dup_words=[]
        return words[::-1]


    def word_prob(self,word): 
        return self.dictionary[word] / self.total
    
    def words(self,text): 
        return re.findall('[a-z]+', text.lower()) 
