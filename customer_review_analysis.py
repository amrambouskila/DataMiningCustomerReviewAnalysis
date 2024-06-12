import nltk
import gensim
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
from textblob import TextBlob
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.metrics import pairwise
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from dateutil import parser
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.utils import simple_preprocess
from kneed import KneeLocator
from sklearn.manifold import TSNE
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import cross_validate, train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

sns.set()


class TopicModel:
    def __init__(self, data, text_matrix, processed_docs, dictionary, vectorizer, corpus, topic_range: range = range(2, 20), top_n: int = 10, elbow_threshold: float = 0.95, random_state: int = 18):
        self.df = data
        self.text_matrix = text_matrix
        self.processed_docs = processed_docs
        self.dictionary = dictionary
        self.vectorizer = vectorizer
        self.corpus = corpus

        self.topic_range = topic_range
        self.top_n = top_n
        self.elbow_threshold = elbow_threshold
        self.random_state = random_state

        self._coherence_scores = None
        self._optimal_topic_count = None
        self._optimal_coherence_score = None
        self._lda_model = None
        self._doc_topic_dist = None
        self._perplexity = None
        self._coherence_lda = None

    @property
    def coherence_scores(self):
        return self._coherence_scores

    @coherence_scores.setter
    def coherence_scores(self, coherence_scores):
        self._coherence_scores = coherence_scores

    @property
    def optimal_topic_count(self):
        return self._optimal_topic_count

    @optimal_topic_count.setter
    def optimal_topic_count(self, optimal_topic_count):
        self._optimal_topic_count = optimal_topic_count

    @property
    def optimal_coherence_score(self):
        return self._optimal_coherence_score

    @optimal_coherence_score.setter
    def optimal_coherence_score(self, optimal_coherence_score):
        self._optimal_coherence_score = optimal_coherence_score

    @property
    def lda_model(self):
        return self._lda_model

    @lda_model.setter
    def lda_model(self, lda_model):
        self._lda_model = lda_model

    @property
    def doc_topic_dist(self):
        return self._doc_topic_dist

    @doc_topic_dist.setter
    def doc_topic_dist(self, doc_topic_dist):
        self._doc_topic_dist = doc_topic_dist

    @property
    def perplexity(self):
        return self._perplexity

    @perplexity.setter
    def perplexity(self, perplexity):
        self._perplexity = perplexity

    @property
    def coherence(self):
        return self._coherence

    @coherence.setter
    def coherence(self, coherence):
        self._coherence = coherence

    def calculate_coherence_scores(self):
        coherence_scores = []
        for num_topics in self.topic_range:
            lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            lda_model.fit(self.text_matrix)
            topics = self.get_lda_topics(lda_model=lda_model)
            coherence_model_lda = CoherenceModel(topics=topics, texts=self.processed_docs, dictionary=self.dictionary, coherence='c_v')
            coherence_scores.append(coherence_model_lda.get_coherence())

        self.coherence_scores = coherence_scores

    def get_lda_topics(self, lda_model):
        words = self.vectorizer.get_feature_names_out()
        topics = []
        for topic_weights in lda_model.components_:
            top_words = [words[i] for i in topic_weights.argsort()[:-self.top_n-1:-1]]
            topics.append(top_words)

        return topics

    def determine_optimal_topics(self):
        # Find the "elbow" point in the coherence scores
        max_score = max(self.coherence_scores)
        optimal_score = max_score
        optimal_idx = self.coherence_scores.index(optimal_score)
        optimal_topic_count = self.topic_range[optimal_idx]
        print(optimal_topic_count)

        possible_optimal_idx = None
        possible_optimal_topic_count = None
        possible_optimal_score = None
        for i, score in enumerate(self.coherence_scores):
            if score >= self.elbow_threshold * max_score:
                optimal_idx = i
                optimal_topic_count = self.topic_range[optimal_idx]
                optimal_score = score
                if optimal_idx == np.argmax(self.coherence_scores):
                    first_derivative = np.diff(self.coherence_scores)
                    second_derivative = np.diff(first_derivative)
                    if len(second_derivative) > 0:
                        possible_optimal_idx = np.argmin(second_derivative) + 2
                        possible_optimal_topic_count = self.topic_range[optimal_idx]
                        possible_optimal_score = self.coherence_scores[optimal_idx]
                else:
                    break

        if possible_optimal_idx is not None and possible_optimal_idx < optimal_idx:
            optimal_idx = possible_optimal_idx
            optimal_topic_count = possible_optimal_topic_count
            optimal_score = possible_optimal_score

        for j, c_score in enumerate(self.coherence_scores):
            if j < optimal_idx and self.coherence_scores[j] > optimal_score:
                optimal_idx = j
                optimal_topic_count = self.topic_range[optimal_idx]
                optimal_score = self.coherence_scores[optimal_idx]

        kn = KneeLocator(self.topic_range, self.coherence_scores, curve='convex', direction='decreasing')
        possible_optimal_idx = kn.knee
        possible_optimal_topic_count = self.topic_range[possible_optimal_idx]
        possible_optimal_score = self.coherence_scores[possible_optimal_idx]
        if possible_optimal_idx is not None and possible_optimal_idx < optimal_idx:
            optimal_idx = possible_optimal_idx
            optimal_topic_count = possible_optimal_topic_count
            optimal_score = possible_optimal_score

        if optimal_idx == len(self.coherence_scores) - 1:
            print(f'Note to self: You may be able to find a better elbow if you include a wider range of topic counts')

        # Plot coherence scores and highlight the optimal point
        plt.figure(figsize=(10, 6))
        plt.plot(self.topic_range, self.coherence_scores, marker='o', label='Coherence Score')
        plt.axvline(x=optimal_topic_count, color='r', linestyle='--', label=f'Optimal Topics: {optimal_topic_count}')
        plt.scatter(optimal_topic_count, optimal_score, color='r')
        plt.xlabel('Number of Topics')
        plt.ylabel('Coherence Score')
        plt.title('Coherence Scores by Number of Topics')
        plt.legend()
        plt.grid(True)
        plt.show()

        self.optimal_topic_count = optimal_topic_count
        self.optimal_coherence_score = optimal_score

    def topic_modeling(self):
        self.lda_model = LatentDirichletAllocation(n_components=self.optimal_topic_count, random_state=self.random_state)
        self.doc_topic_dist = self.lda_model.fit_transform(self.text_matrix)

    def display_topics(self):
        feature_names = self.vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(self.lda_model.components_):
            print(f"Topic {topic_idx}:")
            print(" ".join([feature_names[i] for i in topic.argsort()[:-self.top_n - 1:-1]]))

    def plot_topics(self):
        topics = self.lda_model.components_
        fig, axes = plt.subplots(self.lda_model.n_components, 1, figsize=(8, 15), sharex=True)
        for i, (ax, topic) in enumerate(zip(axes, topics)):
            top_word_indices = topic.argsort()[-self.top_n:][::-1]
            top_word_probs = topic[top_word_indices]
            top_word_labels = [self.vectorizer.get_feature_names_out()[i] for i in top_word_indices]
            sns.barplot(x=top_word_probs, y=top_word_labels, ax=ax)
            ax.set_title(f'Topic {i + 1}')

        plt.tight_layout()
        plt.show()

    def visualize_document_topics(self):
        doc_topic_df = pd.DataFrame(self.doc_topic_dist, columns=[f'Topic {i+1}' for i in range(self.lda_model.n_components)])
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=doc_topic_df, x='Topic 1', y='Topic 2')
        plt.title('Document-Topic Distribution')
        plt.xlabel('Topic 1')
        plt.ylabel('Topic 2')
        plt.show()

    def visualize_topic_correlation_matrix(self):
        topic_corr_matrix = pd.DataFrame(self.doc_topic_dist).corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(topic_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Topic Correlation Matrix Heatmap')
        plt.xlabel('Topics')
        plt.ylabel('Topics')
        plt.show()

    def plot_topic_trends(self):
        doc_topic_df = pd.DataFrame(self.doc_topic_dist, columns=[f'Topic {i+1}' for i in range(self.lda_model.n_components)])
        self.df['dominant_topic'] = doc_topic_df.idxmax(axis=1)
        self.df['reviews.date'] = self.df['reviews.date'].astype(str).apply(lambda x: parser.parse(x) if x != 'nan' else pd.NaT)
        self.df['date'] = pd.to_datetime(self.df['reviews.date'])
        topic_trends = self.df.groupby([self.df['date'].dt.to_period('M'), 'dominant_topic']).size().unstack().fillna(0)
        topic_trends.plot(kind='line', figsize=(15, 7), title='Topic Trends Over Time')
        plt.xlabel('Time')
        plt.ylabel('Number of Documents')
        plt.show()

    def plot_wordclouds(self):
        topics = self.lda_model.components_
        fig, axes = plt.subplots(len(topics), 1, figsize=(10, 15), sharex=True)
        top_n = 10
        for i, (ax, topic) in enumerate(zip(axes, topics)):
            top_word_indices = topic.argsort()[-top_n:][::-1]
            top_word_labels = {self.vectorizer.get_feature_names_out()[j]: topic[j] for j in top_word_indices}
            wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(top_word_labels)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f'Topic {i+1}')
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def cluster_documents(self):
        num_clusters = self.lda_model.n_components
        km = KMeans(n_clusters=num_clusters, random_state=42)
        km.fit(self.doc_topic_dist)
        clusters = km.labels_
        self.df['Cluster'] = clusters

        # Use t-SNE to reduce dimensionality
        tsne_model = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000, random_state=42)
        tsne_values = tsne_model.fit_transform(self.doc_topic_dist)

        # Create a DataFrame with t-SNE results
        tsne_df = pd.DataFrame(tsne_values, columns=['Dim 1', 'Dim 2'])
        tsne_df['Cluster'] = clusters

        # Plot the t-SNE results
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=tsne_df, x='Dim 1', y='Dim 2', hue='Cluster', palette='tab10')
        plt.title('Document Clusters Based on Topic Distribution (t-SNE)')
        plt.show()

    def visualize_lda_model(self):
        panel = pyLDAvis.gensim_models.prepare(self.lda_model, self.corpus, self.dictionary)
        pyLDAvis.save_html(panel, 'lda_visualization.html')

    def plot_interactive_scatter(self):
        fig = px.scatter(self.df, x='Topic 1', y='Topic 2', color='Cluster', title='Interactive Topic Distribution')
        fig.show()

    def add_sentiment_analysis(self):
        self.df['sentiment'] = self.df['reviews.text'].apply(lambda x: TextBlob(x).sentiment.polarity)

    def visualize_sentiment_by_topic(self):
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='dominant_topic', y='sentiment', data=self.df)
        plt.title('Sentiment Distribution by Topic')
        plt.xlabel('Topic')
        plt.ylabel('Sentiment Polarity')
        plt.show()

    def create_dashboard(self):
        st.title('Amazon Reviews Topic Modeling Dashboard')

        st.header('LDA Model Visualization')
        lda_html = open('lda_visualization.html', 'r', encoding='utf-8')
        source_code = lda_html.read()
        st.components.v1.html(source_code, height=800)

        st.header('Sentiment Analysis')
        sentiment_chart = sns.histplot(self.df['sentiment'], kde=True)
        st.pyplot(sentiment_chart.figure)

        st.header('Interactive Topic Trends')
        topic_trends = self.df.groupby(
            [self.df['date'].dt.to_period('M'), 'dominant_topic']).size().unstack().fillna(0)
        topic_trends = topic_trends.reset_index().melt(id_vars=['date'], value_name='count')
        fig = px.line(topic_trends, x='date', y='count', color='dominant_topic', title='Topic Trends Over Time')
        st.plotly_chart(fig)

    def evaluate_model(self):
        perplexity = self.lda_model.log_perplexity(self.corpus)
        coherence_model_lda = CoherenceModel(model=self.lda_model, texts=self.text_matrix, dictionary=self.dictionary, coherence='c_v')
        coherence = coherence_model_lda.get_coherence()
        self.perplexity = perplexity
        self.coherence = coherence


class AssociationRules:
    def __init__(self, df, stop_words, max_features=1000, min_support=0.01, metric="lift", min_threshold=1):
        self.df = df
        self.stop_words = stop_words
        self.max_features = max_features
        self.min_support = min_support
        self.metric = metric
        self.min_threshold = min_threshold
        self.frequent_itemsets = None
        self.rules = None

    def preprocess_data(self):
        vectorizer = CountVectorizer(stop_words=self.stop_words, max_features=self.max_features, binary=True)
        X = vectorizer.fit_transform(self.df['reviews.text'])
        return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    def generate_rules(self):
        X_df = self.preprocess_data()
        self.frequent_itemsets = apriori(X_df, min_support=self.min_support, use_colnames=True)
        self.rules = association_rules(self.frequent_itemsets, metric=self.metric, min_threshold=self.min_threshold)
        self.rules = self.rules.sort_values(['confidence', 'lift'], ascending=[False, False])
        return self.rules

    def plot_rules(self):
        if self.rules is not None:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='support', y='confidence', size='lift', hue='lift', data=self.rules, legend=False, sizes=(20, 200))
            plt.title('Association Rules')
            plt.xlabel('Support')
            plt.ylabel('Confidence')
            plt.show()

    def filter_rules(self, support_threshold=None, confidence_threshold=None, lift_threshold=None):
        filtered_rules = self.rules
        if support_threshold:
            filtered_rules = filtered_rules[filtered_rules['support'] >= support_threshold]
        if confidence_threshold:
            filtered_rules = filtered_rules[filtered_rules['confidence'] >= confidence_threshold]
        if lift_threshold:
            filtered_rules = filtered_rules[filtered_rules['lift'] >= lift_threshold]
        return filtered_rules

    def display_top_rules(self, n=5):
        return self.rules.head(n)


class RecommenderSystems:
    def __init__(self, df, rating_scale=(1, 5)):
        self.df = df
        self.rating_scale = rating_scale
        self.model = None
        self.trainset = None
        self.testset = None

    def build_data(self):
        reader = Reader(rating_scale=self.rating_scale)
        data = Dataset.load_from_df(self.df[['reviews.username', 'name', 'reviews.rating']], reader)
        self.trainset, self.testset = train_test_split(data, test_size=0.25)
        return data

    def train_model(self, algorithm='SVD'):
        data = self.build_data()
        if algorithm == 'SVD':
            self.model = SVD()
        elif algorithm == 'KNN':
            self.model = KNNBasic()
        cross_validate(self.model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        self.model.fit(self.trainset)

    def evaluate_model(self):
        predictions = self.model.test(self.testset)
        y_true = [pred.r_ui for pred in predictions]
        y_pred = [pred.est for pred in predictions]
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'fscore': fscore}

    def make_recommendation(self, user_id, item_id):
        if self.model is None:
            self.train_model()
        pred = self.model.predict(user_id, item_id)
        return pred

    def get_top_n_recommendations(self, user_id, n=5):
        if self.model is None:
            self.train_model()
        all_items = self.df['name'].unique()
        user_rated_items = self.df[self.df['reviews.username'] == user_id]['name'].unique()
        items_to_predict = [item for item in all_items if item not in user_rated_items]
        predictions = [self.model.predict(user_id, item) for item in items_to_predict]
        predictions.sort(key=lambda x: x.est, reverse=True)
        return predictions[:n]


class MarketAnalytics:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MarketAnalytics, cls).__new__(cls)

        return cls._instance

    def __init__(self, data_path, stop_words, max_features: int = 1000):
        self.df = pd.read_csv(data_path)
        self.stop_words = stop_words
        self.vectorizer = CountVectorizer(stop_words=self.stop_words, max_features=max_features)
        self.text_matrix = self.preprocess_data()
        self.processed_docs, self.dictionary, self.corpus = self.preprocess_documents()
        self.topic_model = TopicModel(data=self.df, text_matrix=self.text_matrix, processed_docs=self.processed_docs, dictionary=self.dictionary, vectorizer=self.vectorizer, corpus=self.corpus)
        self.association_rules = AssociationRules(df=self.df, stop_words=self.stop_words, max_features=max_features)
        self.recommender_systems = RecommenderSystems(df=self.df)

    def preprocess_data(self):
        self.df['reviews.text'] = self.df['reviews.text'].astype(str).str.lower().str.replace(r'[^\w\s]', '')
        return self.vectorizer.fit_transform(self.df['reviews.text'])

    def preprocess_documents(self):
        lemmatizer = WordNetLemmatizer()
        processed_docs = [[lemmatizer.lemmatize(token) for token in simple_preprocess(doc) if token not in self.stop_words] for doc in self.df['reviews.text'].values]
        dictionary = Dictionary(processed_docs)
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        return processed_docs, dictionary, corpus

    def topic_modeling(self):
        self.topic_model.calculate_coherence_scores()
        self.topic_model.determine_optimal_topics()
        print(f'The optimal number of topics is {self.topic_model.optimal_topic_count} with a coherence score of {self.topic_model.optimal_coherence_score}')

        self.topic_model.topic_modeling()
        self.topic_model.display_topics()
        self.topic_model.plot_topics()
        self.topic_model.visualize_document_topics()
        self.topic_model.visualize_topic_correlation_matrix()
        self.topic_model.plot_topic_trends()
        self.topic_model.plot_wordclouds()
        self.topic_model.cluster_documents()
        self.topic_model.add_sentiment_analysis()
        self.topic_model.visualize_sentiment_by_topic()
        self.topic_model.visualize_lda_model()
        self.topic_model.plot_interactive_scatter()
        self.topic_model.create_dashboard()
        self.topic_model.evaluate_model()
        print(f'Perplexity: {self.topic_model.perplexity}, Coherence: {self.topic_model.coherence}')

    def generate_association_rules(self):
        rules = self.association_rules.generate_rules()
        self.association_rules.plot_rules()
        print(rules.head())
        return rules

    def make_recommendation(self, user_id, item_id):
        pred = self.recommender_systems.make_recommendation(user_id, item_id)
        print(pred)
        return pred

    def get_top_n_recommendations(self, user_id, n=5):
        recommendations = self.recommender_systems.get_top_n_recommendations(user_id, n)
        for recommendation in recommendations:
            print(f'Item: {recommendation.iid}, Estimated Rating: {recommendation.est}')
        return recommendations


if __name__ == '__main__':
    # This script performs topic modeling on Amazon customer reviews using LDA. It includes preprocessing,
    # topic modeling, sentiment analysis, and various visualizations. Interactive components are included
    # using pyLDAvis, seaborn, matplotlib, and Plotly.
    #
    # Steps:
    # 1. Preprocess the data
    # 2. Preprocess documents for LDA
    # 3. Calculate coherence scores for different topic numbers
    # 4. Determine the optimal number of topics
    # 5. Perform topic modeling with the optimal number of topics
    # 6. Display and visualize topics
    # 7. Perform sentiment analysis
    # 8. Visualize sentiment distribution by topic
    # 9. Visualize the LDA model with pyLDAvis
    # 10. Plot interactive scatter and topic trends
    # 11. Create a Streamlit dashboard
    # 12. Preprocess with Word2Vec (if needed for further analysis)
    # 13. Evaluate the model with perplexity and coherence scores

    stop_words = stopwords.words('english')
    analytics = MarketAnalytics(data_path='./data/customer_reviews.csv', stop_words=stop_words)
    analytics.topic_modeling()

    # Generate association rules
    rules = analytics.generate_association_rules()

    # Make a recommendation
    user_id = 'A3J0745PWSBKDI'
    item_id = 'B0002E2WKO'
    prediction = analytics.make_recommendation(user_id, item_id)

    # Get top N recommendations
    recommendations = analytics.get_top_n_recommendations(user_id, n=5)

    # TODO Once you get it all working, convert all pandas operations to polars and then try to use tableau for visualizations if possible
