"""
Feature Engineering Module
Extracts features from text data for model training
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor: 
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.keywords = [
            'graph', 'tree', 'dynamic', 'dp', 'recursion', 'backtrack',
            'greedy', 'sort', 'search', 'binary', 'array', 'string',
            'matrix', 'linked', 'list', 'stack', 'queue', 'hash',
            'dfs', 'bfs', 'dijkstra', 'shortest', 'path', 'optimize',
            'maximum', 'minimum', 'subsequence', 'substring'
        ]
    
    def extract_text_length_features(self, df):
        features = pd.DataFrame()
        
        def safe_len(text):
            return 0 if pd.isna(text) else len(str(text))
        
        def safe_word_count(text):
            return 0 if pd.isna(text) else len(str(text).split())
        
        def safe_avg_word_len(text):
            if pd.isna(text):
                return 0
            words = str(text).split()
            return 0 if not words else sum(len(w) for w in words) / len(words)
        
        features['char_count'] = df['combined_text']. apply(safe_len)
        features['word_count'] = df['combined_text'].apply(safe_word_count)
        features['avg_word_length'] = df['combined_text'].apply(safe_avg_word_len)
        features['sentence_count'] = df['combined_text'].apply(
            lambda x: len(re.findall(r'[.!?]+', str(x))) if pd.notna(x) else 0
        )
        features['description_length'] = df['description'].apply(safe_len)
        features['input_desc_length'] = df['input_description'].apply(safe_len)
        features['output_desc_length'] = df['output_description'].apply(safe_len)
        
        return features
    
    def extract_mathematical_features(self, df):
        features = pd.DataFrame()
        
        def safe_count(text, pattern):
            return 0 if pd.isna(text) else len(re.findall(pattern, str(text)))
        
        features['math_operators'] = df['combined_text'].apply(
            lambda x: safe_count(x, r'[\+\-\*\/\=\<\>]')
        )
        features['parentheses_count'] = df['combined_text'].apply(
            lambda x: safe_count(x, r'[\(\)\[\]\{\}]')
        )
        features['number_count'] = df['combined_text'].apply(
            lambda x: safe_count(x, r'\b\d+\b')
        )
        features['has_formula'] = df['combined_text'].apply(
            lambda x: 1 if (pd.notna(x) and re.search(r'\^|_\d+', str(x))) else 0
        )
        
        return features
    
    def extract_keyword_features(self, df):
        features = pd.DataFrame()
        
        for keyword in self.keywords:
            features[f'keyword_{keyword}'] = df['combined_text']. apply(
                lambda x: len(re.findall(r'\b' + keyword + r'\b', str(x).lower())) if pd.notna(x) else 0
            )
        
        features['total_keywords'] = features. sum(axis=1)
        return features
    
    def extract_tfidf_features(self, df, fit=True):
        texts = df['combined_text'].fillna('').astype(str)
        
        if fit: 
            tfidf_matrix = self.tfidf_vectorizer. fit_transform(texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        return tfidf_matrix. toarray()
    
    def extract_all_features(self, df, fit_tfidf=True):
        print("Extracting features...")
        
        length_features = self.extract_text_length_features(df)
        math_features = self.extract_mathematical_features(df)
        keyword_features = self.extract_keyword_features(df)
        tfidf_features = self.extract_tfidf_features(df, fit=fit_tfidf)
        
        combined_features = pd.concat([
            length_features,
            math_features,
            keyword_features
        ], axis=1)
        
        tfidf_df = pd.DataFrame(
            tfidf_features,
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        combined_features = pd. concat([combined_features, tfidf_df], axis=1)
        
        print(f"Extracted {combined_features.shape[1]} features")
        
        return combined_features.values, combined_features.columns.tolist()


def main():
    df = pd.read_csv('data/processed_data.csv')
    extractor = FeatureExtractor()
    features, feature_names = extractor. extract_all_features(df)
    
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Number of features: {len(feature_names)}")
    print("\nSample feature names:")
    print(feature_names[:10])


if __name__ == "__main__":
    main()
