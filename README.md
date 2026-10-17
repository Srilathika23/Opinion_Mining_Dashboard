# Opinion Mining and Tag-Based Categorization of User Feedback

Text Mining case study: Streamlit dashboard for BoW, TF-IDF, SVD/LSA,
sentiment classification (VADER) and tag ranking/refinement/enrichment.

## Files
- app.py: the dashboard
- prepare_dataset.py: builds the CSV from the Hu & Liu (2004) customer reviews
- product_reviews_dataset.csv: the dataset (3,773 review sentences, 5 products)
- requirements.txt: Python packages

## Run
    pip install -r requirements.txt
    python prepare_dataset.py   # optional, the CSV is already included
    streamlit run app.py
Then upload product_reviews_dataset.csv in the sidebar and choose the "comment" column.

## Data source
Hu, M. and Liu, B. (2004). Mining and summarizing customer reviews. KDD-04.
http://www.cs.uic.edu/~liub/FBS/FBS.html
