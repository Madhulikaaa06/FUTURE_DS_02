import pandas as pd

pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 17)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_excel("customersupporttickets.xlsx")
print(df)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from collections import Counter
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')


df = pd.read_excel("customersupporttickets.xlsx")


df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'], errors='coerce')
df['First Response Time'] = pd.to_datetime(df['First Response Time'], errors='coerce')
df['Time to Resolution'] = pd.to_datetime(df['Time to Resolution'], errors='coerce')
# Drop irrelevant columns (like personal info)
df = df.drop(columns=['Customer Name', 'Customer Email'])


df['Ticket Description'] = df['Ticket Description'].fillna("")


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words and w.isalpha()]
    return tokens
df['Cleaned_Tokens'] = df['Ticket Description'].apply(clean_text)


all_words = [word for tokens in df['Cleaned_Tokens'] for word in tokens]
word_freq = Counter(all_words)
common_words = word_freq.most_common(20)


print("Top 20 most common words in support tickets:")
for word, freq in common_words:
    print(f"{word}: {freq}")


issue_keywords = {
    'login': 'Account Issues',
    'access': 'Account Issues',
    'password': 'Account Issues',
    'billing': 'Billing Issues',
    'charge': 'Billing Issues',
    'payment': 'Billing Issues',
    'setup': 'Technical Issues',
    'error': 'Technical Issues',
    'install': 'Technical Issues',
    'network': 'Connectivity Issues',
    'internet': 'Connectivity Issues'
}

def map_issue_category(tokens):
    for token in tokens:
        if token in issue_keywords:
            return issue_keywords[token]
    return 'Other'

df['Issue Category'] = df['Cleaned_Tokens'].apply(map_issue_category)


issue_counts = df['Issue Category'].value_counts()
print("\nTicket Counts by Issue Category:")
print(issue_counts)


df['Response Delay (hrs)'] = (df['First Response Time'] - df['Date of Purchase']).dt.total_seconds() / 3600
category_response_time = df.groupby('Issue Category')['Response Delay (hrs)'].mean().sort_values()

print("\nAverage First Response Delay by Issue Category (in hours):")
print(category_response_time)


satisfaction_by_issue = df.groupby('Issue Category')['Customer Satisfaction Rating'].mean()
print("\nAverage Satisfaction Rating by Issue Category:")
print(satisfaction_by_issue)


plt.figure(figsize=(10, 5))
issue_counts.plot(kind='bar', color='pink')
plt.title('Number of Tickets per Issue Category')
plt.ylabel('Ticket Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('ticket_counts_by_category.png')


