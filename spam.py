import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
nltk.download('punkt')
nltk.download('stopwords')

################## test file. change if data is not good 
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

def pre(text):
    tokens = nltk.word_tokenize(text.lower())
    
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    preprocessed_text = ' '.join(stemmed_tokens)
    return preprocessed_text
df['text'] = df['text'].apply(pre)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

classifier = MultinomialNB()

classifier.fit(X_train_vect, y_train)

y_pred = classifier.predict(X_test_vect)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the model: {accuracy:.2%}')
