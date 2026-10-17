"""
Opinion Mining and Tag-Based Categorization of User Feedback
Run:  pip install -r requirements.txt
      streamlit run app.py
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

st.set_page_config(page_title="Opinion Mining Dashboard", layout="wide")

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
@st.cache_resource
def setup():
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        nltk.download(pkg, quiet=True)
    return (set(stopwords.words("english")), WordNetLemmatizer(),
            PorterStemmer(), SentimentIntensityAnalyzer())

STOP, LEMMA, STEM, VADER = setup()
TOKENIZER = RegexpTokenizer(r"[a-z]+")

DEFAULT_CSV = Path(__file__).parent / "product_reviews_dataset.csv"

# --------------------------------------------------------------------------
# Stage 1: cleaning
# --------------------------------------------------------------------------
def clean(text, use_stem=False, remove_stop=True):
    text = re.sub(r"http\S+|www\.\S+", " ", str(text).lower())
    text = re.sub(r"<[^>]+>", " ", text)
    tokens = TOKENIZER.tokenize(text)
    if remove_stop:
        tokens = [t for t in tokens if t not in STOP]
    tokens = [LEMMA.lemmatize(t) for t in tokens if len(t) > 2]
    if use_stem:
        tokens = [STEM.stem(t) for t in tokens]
    return tokens

# --------------------------------------------------------------------------
# Stage 4: sentiment
# --------------------------------------------------------------------------
def classify(text, pos_t=0.05, neg_t=-0.05):
    c = VADER.polarity_scores(str(text))["compound"]
    label = "Positive" if c >= pos_t else "Negative" if c <= neg_t else "Neutral"
    return c, label

# --------------------------------------------------------------------------
# Stage 5: tags
# --------------------------------------------------------------------------
def is_opinion_word(w):
    return w in VADER.lexicon

def has_noun(w):
    return bool(wordnet.synsets(w, pos=wordnet.NOUN))

def refine_tags(ranked, min_df_tfidf):
    """Drop opinion words, non-nouns, and stem-duplicates; keep the best-scored variant."""
    seen, out = set(), {}
    for term, score in ranked.items():
        words = term.split()
        if any(is_opinion_word(w) for w in words):
            continue
        if not any(has_noun(w) for w in words):
            continue
        key = " ".join(STEM.stem(w) for w in words)
        if key in seen:
            continue
        seen.add(key)
        out[term] = score
    return out

def enrich_tags(refined, docs_tokens, top_k=10):
    """Add WordNet synonyms and strongest corpus co-occurring term for top tags."""
    enriched = dict(refined)
    source = {t: "refined" for t in refined}
    uni = [t for t in list(refined)[:top_k] if " " not in t]
    # co-occurrence
    vocab = sorted({w for d in docs_tokens for w in d})
    cv = CountVectorizer(vocabulary=vocab, binary=True, analyzer=lambda x: x)
    X = cv.fit_transform(docs_tokens)
    C = (X.T @ X).toarray()
    np.fill_diagonal(C, 0)
    idx = {w: i for i, w in enumerate(vocab)}
    for tag in uni:
        score = refined[tag]
        syns = wordnet.synsets(tag, pos=wordnet.NOUN)
        if syns:
            for lem in syns[0].lemma_names()[:3]:
                s = lem.replace("_", " ").lower()
                if s != tag and s not in enriched:
                    enriched[s] = score * 0.5
                    source[s] = f"synonym of '{tag}'"
        row = C[idx[tag]]
        for j in np.argsort(-row)[:5]:
            w = vocab[j]
            if row[j] > 0 and w not in enriched and not is_opinion_word(w) and has_noun(w):
                enriched[w] = score * 0.4
                source[w] = f"co-occurs with '{tag}'"
                break
    return enriched, source

def cloud(freqs, title):
    if not freqs:
        st.info("No tags to display.")
        return
    wc = WordCloud(width=900, height=420, background_color="white",
                   colormap="viridis").generate_from_frequencies(freqs)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    st.pyplot(fig)

# --------------------------------------------------------------------------
# Sidebar: data and settings
# --------------------------------------------------------------------------
st.title("Opinion Mining & Tag-Based Categorization of User Feedback")
st.caption("Pipeline: Cleaning + BoW, TF-IDF, SVD/LSA, Sentiment, Tag ranking/refinement/enrichment")

with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload CSV of comments (optional)", type="csv")
    if up:
        raw = pd.read_csv(up)
    elif DEFAULT_CSV.exists():
        raw = pd.read_csv(DEFAULT_CSV)
        st.info("Using product_reviews_dataset.csv (Hu & Liu, 2004). Upload another CSV to replace it.")
    else:
        st.warning("Upload a CSV file with a column of comments to start.")
        st.stop()
    default_idx = list(raw.columns).index("comment") if "comment" in raw.columns else 0
    col = st.selectbox("Text column", raw.columns, index=default_idx)
    comments = raw[col].dropna().astype(str).tolist()
    st.header("Preprocessing")
    use_stem = st.checkbox("Apply stemming (after lemmatization)", False)
    max_feat = st.slider("Max TF-IDF features", 50, 2000, 500, 50)
    top_n = st.slider("Top N terms to show", 5, 30, 15)
    st.header("Sentiment thresholds")
    pos_t = st.slider("Positive if compound >=", 0.0, 0.5, 0.05, 0.01)
    neg_t = -st.slider("Negative if compound <=", 0.0, 0.5, 0.05, 0.01)

df = pd.DataFrame({"comment": comments})
df["tokens"] = df["comment"].apply(lambda t: clean(t, use_stem))
df["clean"] = df["tokens"].apply(" ".join)
df = df[df["clean"].str.len() > 0].reset_index(drop=True)
res = df["comment"].apply(lambda t: classify(t, pos_t, neg_t))
df["compound"] = [r[0] for r in res]
df["sentiment"] = [r[1] for r in res]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Comments", len(df))
c2.metric("Positive", int((df.sentiment == "Positive").sum()))
c3.metric("Neutral", int((df.sentiment == "Neutral").sum()))
c4.metric("Negative", int((df.sentiment == "Negative").sum()))

tabs = st.tabs(["1. Cleaning & BoW", "2. TF-IDF", "3. Dimensionality reduction",
                "4. Sentiment", "5. Tags"])

# ---- Tab 1
with tabs[0]:
    st.subheader("Raw vs cleaned text")
    st.dataframe(df[["comment", "clean"]], use_container_width=True)
    bow = CountVectorizer()
    B = bow.fit_transform(df["clean"])
    freq = pd.DataFrame({"term": bow.get_feature_names_out(),
                         "count": np.asarray(B.sum(axis=0)).ravel()}).sort_values("count", ascending=False)
    st.write(f"Bag-of-Words matrix: {B.shape[0]} documents x {B.shape[1]} unique terms")
    st.plotly_chart(px.bar(freq.head(top_n).iloc[::-1], x="count", y="term", orientation="h",
                           title=f"Top {top_n} words by frequency"), use_container_width=True)
    with st.expander("View BoW matrix (first 15 docs x top 20 terms)"):
        top_terms = freq.head(20)["term"].tolist()
        cols = [list(bow.get_feature_names_out()).index(t) for t in top_terms]
        st.dataframe(pd.DataFrame(B[:15, cols].toarray(), columns=top_terms))

# ---- Tab 2
with tabs[1]:
    tf = TfidfVectorizer(ngram_range=(1, 2), max_features=max_feat, min_df=1, sublinear_tf=True)
    T = tf.fit_transform(df["clean"])
    terms = tf.get_feature_names_out()
    mean_scores = np.asarray(T.mean(axis=0)).ravel()
    sum_scores = np.asarray(T.sum(axis=0)).ravel()
    tdf = pd.DataFrame({"term": terms, "mean_tfidf": mean_scores}).sort_values("mean_tfidf", ascending=False)
    st.write(f"TF-IDF matrix: {T.shape[0]} documents x {T.shape[1]} features (unigrams + bigrams)")
    st.plotly_chart(px.bar(tdf.head(top_n).iloc[::-1], x="mean_tfidf", y="term", orientation="h",
                           title=f"Top {top_n} terms by mean TF-IDF"), use_container_width=True)
    n_docs = min(15, T.shape[0])
    top_idx = np.argsort(-mean_scores)[:top_n]
    heat = pd.DataFrame(T[:n_docs][:, top_idx].toarray(), columns=terms[top_idx],
                        index=[f"Doc {i+1}" for i in range(n_docs)])
    st.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="Blues",
                              title="TF-IDF heatmap (first docs x top terms)"), use_container_width=True)

# ---- Tab 3
with tabs[2]:
    max_k = max(2, min(50, T.shape[1] - 1, T.shape[0] - 1))
    svd_full = TruncatedSVD(n_components=max_k, random_state=42).fit(T)
    cum = np.cumsum(svd_full.explained_variance_ratio_)
    ev = pd.DataFrame({"components": range(1, max_k + 1), "cumulative_variance": cum})
    st.plotly_chart(px.line(ev, x="components", y="cumulative_variance", markers=True,
                            title="Cumulative explained variance (SVD / LSA)"), use_container_width=True)
    svd2 = TruncatedSVD(n_components=2, random_state=42)
    Z = svd2.fit_transform(T)
    pdf = pd.DataFrame({"SV1": Z[:, 0], "SV2": Z[:, 1], "sentiment": df["sentiment"],
                        "comment": df["comment"].str.slice(0, 90)})
    st.plotly_chart(px.scatter(pdf, x="SV1", y="SV2", color="sentiment", hover_data=["comment"],
                               color_discrete_map={"Positive": "#2ca02c", "Neutral": "#7f7f7f", "Negative": "#d62728"},
                               title="2-D LSA projection of comments"), use_container_width=True)
    st.write(f"2 components explain {svd2.explained_variance_ratio_.sum():.1%} of variance; "
             f"{max_k} components explain {cum[-1]:.1%}.")
    comp = pd.DataFrame(svd2.components_.T, index=terms, columns=["SV1", "SV2"])
    with st.expander("Top terms per component (latent topics)"):
        for c in comp.columns:
            st.write(c, ", ".join(comp[c].abs().sort_values(ascending=False).head(8).index))

# ---- Tab 4
with tabs[3]:
    cnt = df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0).reset_index()
    cnt.columns = ["sentiment", "count"]
    cmap = {"Positive": "#2ca02c", "Neutral": "#7f7f7f", "Negative": "#d62728"}
    a, b = st.columns(2)
    a.plotly_chart(px.pie(cnt, names="sentiment", values="count", color="sentiment",
                          color_discrete_map=cmap, title="Sentiment distribution"), use_container_width=True)
    b.plotly_chart(px.histogram(df, x="compound", nbins=20, color="sentiment",
                                color_discrete_map=cmap, title="Compound score distribution"), use_container_width=True)
    pick = st.multiselect("Filter", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"])
    st.dataframe(df[df.sentiment.isin(pick)][["comment", "compound", "sentiment"]], use_container_width=True)
    st.download_button("Download results CSV",
                       df[["comment", "clean", "compound", "sentiment"]].to_csv(index=False),
                       "sentiment_results.csv")

# ---- Tab 5
with tabs[4]:
    tf_tag = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    Tt = tf_tag.fit_transform(df["clean"])
    sc = pd.Series(np.asarray(Tt.sum(axis=0)).ravel(), index=tf_tag.get_feature_names_out())
    ranked = sc.sort_values(ascending=False).head(60).to_dict()
    refined = refine_tags(ranked, None)
    enriched, source = enrich_tags(refined, df["tokens"].tolist())

    st.markdown("**Ranking**: terms scored by summed TF-IDF.  \n"
                "**Refinement**: remove opinion words (VADER lexicon), non-nouns and stem-duplicates.  \n"
                "**Enrichment**: add WordNet synonyms and top co-occurring terms of the best tags.")
    t1, t2, t3 = st.tabs(["Ranked", "Refined", "Enriched"])
    with t1:
        cloud(ranked, "Ranked tags")
        st.dataframe(pd.Series(ranked, name="score").head(20))
    with t2:
        cloud(refined, "Refined tags")
        st.dataframe(pd.Series(refined, name="score").head(20))
    with t3:
        cloud(enriched, "Enriched tags")
        st.dataframe(pd.DataFrame({"score": pd.Series(enriched),
                                   "origin": pd.Series(source)}).sort_values("score", ascending=False))
    st.subheader("Tags by sentiment")
    sel = st.radio("Sentiment", ["Positive", "Neutral", "Negative"], horizontal=True)
    sub = df[df.sentiment == sel]["clean"]
    if len(sub) >= 2:
        tv = TfidfVectorizer(sublinear_tf=True)
        S = tv.fit_transform(sub)
        s_sc = pd.Series(np.asarray(S.sum(axis=0)).ravel(), index=tv.get_feature_names_out())
        s_ref = refine_tags(s_sc.sort_values(ascending=False).head(40).to_dict(), None)
        cloud(s_ref, f"Tags for {sel} comments")
    else:
        st.info("Not enough comments in this class.")