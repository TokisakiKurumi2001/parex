import re
from keybert import KeyBERT
import pandas as pd

def clean(sent: str) -> str:
    _sent = sent
    _sent = re.sub('`|[|]|\.', "", _sent)
    _sent = re.sub('\.+', '\.', _sent)
    _sent = re.sub("\s+", " ", _sent)
    return _sent

if __name__ == "__main__":
    kw_model = KeyBERT()
    sents = []
    keywords_per_sent = []
    with open('input.txt') as file:
        for line in file:
            line = re.sub("\n", "", line)
            sent = clean(line)
            keywords = [k[0] for k in kw_model.extract_keywords(sent, keyphrase_ngram_range=(1, 1), stop_words='english')]
            if len(keywords) < 3:
                continue
            keyword = "<k>" + "</k>".join(keywords) + "</k>"
            sents.append(sent)
            keywords_per_sent.append(keyword)

    df = pd.DataFrame({'Sent': sents, 'Keywords': keywords_per_sent})
    df.to_csv('data.csv', index=False)