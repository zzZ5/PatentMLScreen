import pandas
import re
import jieba
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# 导入数据
df = pandas.read_csv("patent_verified.csv")
df["x"] = df["Application Id"].map(str) + " " + df["Application Date"].map(str) + " " + df["Publication Date"].map(str) + " " + df["Country"].map(str) + " " + df["Title"].map(str) + " " + df["Abstract"].map(str) + " " + df["Applicants"].map(str) + " " + df["Inventors"].map(str)
patents = df["x"].values.tolist()
labels = df["y"].values.tolist()

# 清洗数据

def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # 移除特殊字符
    text = re.sub(r'\s+', ' ', text)  # 移除多余空格
    return text.lower()

# 分语言处理和分词
def tokenize(text):
    words = []
    for word in text.split():
        if re.search('[\u4e00-\u9fff]', text):  # 检测中文字符
            words.extend(jieba.cut(word))
        else:
            words.append(word)
    return words

# 去除停用词
def remove_stopwords(words):
    stop_words_en = set(stopwords.words('english'))
    stop_words_zh = set(["的", "这是", "和", "..."])  # 示例中文停用词
    return [word for word in words if word not in stop_words_en and word not in stop_words_zh]


# 预处理文本
processed_texts = []
for text in patents:
    text = clean_text(text)
    words = tokenize(text)
    words = remove_stopwords(words)
    processed_texts.append(' '.join(words))

# 向量化
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_texts)

data_
tfidf_matrix.toarray()