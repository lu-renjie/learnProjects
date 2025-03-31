import re
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class Tokenizer:
    def __init__(self, remove_stop_words=False):
        self.wnl = WordNetLemmatizer()
        self.characters = re.compile("""[^ a-z0-9 .,'"!;:?\-() /《》$%& ]""")
        self.map = {
            # "'s": 'is',  # 's可能是所有格, 也可能是is, 也可能是us, 比如let's, 不能简单的转换
            'wo': 'will',  # won't --> ['wo', 'n't']
            'ca': 'can',   # can't --> ['ca', 'n't']
            "'re": 'are',
            "'m": 'am',
            "'d": 'would',
            "'ve": 'have',
            "'ll": 'will',
            "n't": 'not',
        }
        self.stopwords = stopwords.words('english')
        self.remove_stop_words = remove_stop_words

    def tokenize(self, sentence):
        sentence = sentence.lower()

        # 去掉奇奇怪怪的符号
        sentence = re.sub(self.characters, ' ', sentence)

        # 分词
        words = word_tokenize(sentence)

        # 词形还原
        tagged_words = pos_tag(words)
        for i, (word, tag) in enumerate(tagged_words):
            words[i] = self.lemmatize(word, tag)

        # 去除停用词, 这个会去掉很多有意义的词, 默认不去掉
        if self.remove_stop_words:
            words = [word for word in words if word not in self.stopwords]
        return words

    def lemmatize(self, word, tag):
        try:
            word = self.map[word]
            return word
        except KeyError:
            if tag.startswith('N'):
                return self.wnl.lemmatize(word, pos='n')
            elif tag.startswith('V'):
                return self.wnl.lemmatize(word, pos='v')
            elif tag.startswith('J'):
                return self.wnl.lemmatize(word, pos='a')
            elif tag.startswith('R'):
                return self.wnl.lemmatize(word, pos='r')
            else:
                return word