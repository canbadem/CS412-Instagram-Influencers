import numpy as np
import pandas as pd
import gzip
import json
from pprint import pprint
import re

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack  # To combine sparse matrices
from sklearn.model_selection import train_test_split

#Stopwords
nltk.download('stopwords')
turkish_stopwords = stopwords.words('turkish')

#Paths
dataset_path = "round1/content/released_dataset/training-dataset.jsonl.gz"
train_classification_labels_path = "round1/content/released_dataset/train-classification.csv"
test_regression_path = "round1/content/released_dataset/test-regression-round1.jsonl"
test_classification_path = "round1/content/released_dataset/test-classification-round1.dat"

#Loads the training labels for classification: Username - Label
def LoadTrainingClassification():
    print("Loading classification labels")
    #Read the CSV
    train_classification_df = pd.read_csv(train_classification_labels_path,)
    #Rename the columns
    train_classification_df = train_classification_df.rename(columns={'Unnamed: 0': 'user_id', 'label': 'category'})

    # Unifying labels
    train_classification_df["category"] = train_classification_df["category"].apply(str.lower)
    
    print("Loaded classification labels")
    print(train_classification_df.groupby("category").count())
    
    return train_classification_df.set_index("user_id").to_dict()["category"]

def LoadDataset():
    print("Loading dataset")
    username2category_train = LoadTrainingClassification()
    username2posts_train = dict()
    username2profile_train = dict()

    username2posts_test = dict()
    username2profile_test = dict()

    with gzip.open(dataset_path, "rt") as fh:
        #Each line is a profile
        for line in fh:
            sample = json.loads(line)
            profile = sample["profile"]
            
            username = profile["username"]
            
            #If the username can be found in the training classification labels, then its in the training data
            if username in username2category_train:
                # train data info
                username2posts_train[username] = sample["posts"]
                username2profile_train[username] = profile
            else:
                # it is test data info
                username2posts_test[username] = sample["posts"]
                username2profile_test[username] = profile
    
    print("Loaded dataset")
    return username2category_train, username2posts_train, username2profile_train, username2posts_test, username2profile_test

username2category_train, username2posts_train, username2profile_train, username2posts_test, username2profile_test = LoadDataset()

def preprocess_text(text: str):
    # lower casing Turkish Text, Don't use str.lower :)
    text = text.casefold()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove special characters and punctuation
    # HERE THE EMOJIS stuff are being removed, you may want to keep them :D
    text = re.sub(r'[^a-zçğıöşü0-9\s#@]', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def VectorizeDataset(maxFeatures, funcPostToString):
  print("Vectorizing dataset")
  # to keep the label order
  train_usernames = []
  vectorizer = TfidfVectorizer(stop_words=turkish_stopwords, max_features=maxFeatures)

  def vectorize(username2posts, shouldFit, funcPostToString):
    corpus = []
    for username, posts in username2posts.items():
      train_usernames.append(username)
      postStr = funcPostToString(username, posts)
      corpus.append(postStr)
    
    if shouldFit:
      # fit the vectorizer
      vectorizer.fit(corpus)
    
    return vectorizer.transform(corpus)

  x_post_train = vectorize(username2posts_train, True, funcPostToString)
  y_train = [username2category_train.get(uname, "NA") for uname in train_usernames]

  x_post_test = vectorize(username2posts_test, False, funcPostToString)
  
  feature_names = vectorizer.get_feature_names_out()
  df_tfidf = pd.DataFrame(x_post_train.toarray(), columns=feature_names)
  df_tfidf.head(2)
  
  print("Vectorized dataset")
  
  return x_post_train, y_train, x_post_test, df_tfidf

def VectorizeDatasetSeperately(maxFeaturesPosts, maxFeaturesBios, postsWeight, biosWeight, funcPostToString):
  print("Vectorizing dataset")
  # to keep the label order
  vectorizer_posts = TfidfVectorizer(stop_words=turkish_stopwords, max_features=maxFeaturesPosts)
  vectorizer_bios = TfidfVectorizer(stop_words=turkish_stopwords, max_features=maxFeaturesBios)
  
  def vectorize(username2posts, username2profile, shouldFit, funcPostToString, vectorizerPosts, vectorizerBios):
    corpus_posts = []
    corpus_bios = []
    usernames = []
    for username, posts in username2posts.items():
      usernames.append(username)
      profile = username2profile.get(username)
      
      postStr = funcPostToString(username, posts)
      corpus_posts.append(postStr)
    
      # Aggregating user bio
      cleaned_bio = ""
      if profile is not None:
          bio = profile["biography"]
          if bio is not None:
              cleaned_bio = preprocess_text(bio)
              
      corpus_bios.append(cleaned_bio)

    if shouldFit:
      # fit the vectorizer
      vectorizerPosts.fit(corpus_posts)
      vectorizerBios.fit(corpus_bios)
    
    x_post_train, x_bio_train = vectorizerPosts.transform(corpus_posts), vectorizerBios.transform(corpus_bios)
    x_combined_train = hstack([x_post_train.multiply(postsWeight), x_bio_train.multiply(biosWeight)])
    return x_combined_train, usernames, x_post_train, x_bio_train

  x_combined_train, train_usernames, x_post_train, x_bio_train = vectorize(username2posts_train, username2profile_train, True, funcPostToString, vectorizer_posts, vectorizer_bios)

  y_train = [username2category_train.get(uname, "NA") for uname in train_usernames]

  x_combined_test, test_usernames, x_post_test, x_bio_test = vectorize(username2posts_test, username2profile_test, False, funcPostToString, vectorizer_posts, vectorizer_bios)

  feature_names_posts = vectorizer_posts.get_feature_names_out()
  feature_names_bios = vectorizer_bios.get_feature_names_out()

  # Prefix feature names
  feature_names_posts = [f"post_{name}" for name in feature_names_posts]
  feature_names_bios = [f"bio_{name}" for name in feature_names_bios]

  combined_feature_names = feature_names_posts + feature_names_bios
  df_tfidf = pd.DataFrame(x_combined_train.toarray(), columns=combined_feature_names)

  print("Vectorized dataset")
  
  return x_combined_train, y_train, x_combined_test, df_tfidf, train_usernames, test_usernames, x_post_train, x_bio_train, x_post_test, x_bio_test, combined_feature_names

def UserAggregatorPosts(username, posts):
  cleaned_captions = []
  for post in posts:
    post_caption = post.get("caption", "")
    if post_caption is None:
      continue

    post_caption = preprocess_text(post_caption)

    if post_caption != "":
      cleaned_captions.append(post_caption)
  
  user_post_captions = "\n".join(cleaned_captions)
  return user_post_captions

def DoPCA(x_combined, y_combined, pca_components=100, val_size=0.2):
    print("Applying PCA")
    
    # Convert sparse matrix to dense
    x_combined_dense = x_combined.toarray()
    
    # Standardize data
    scaler = StandardScaler()
    x_combined_scaled = scaler.fit_transform(x_combined_dense)
    
    # Apply PCA
    pca = PCA(n_components=pca_components)
    x_combined_pca = pca.fit_transform(x_combined_scaled)
    
    print("PCA completed")
    print(f"Explained variance (first few components): {pca.explained_variance_ratio_[:10]}")
    print(f"Cumulative variance explained: {pca.explained_variance_ratio_.cumsum()[:10]}")
    
    # Train-test split
    x_train, x_val, y_train, y_val = train_test_split(
        x_combined_pca, y_combined, test_size=val_size, stratify=y_combined
    )
    
    print("Dataset split into training and validation sets")
    
    # Return all relevant outputs
    return x_train, x_val, y_train, y_val, pca