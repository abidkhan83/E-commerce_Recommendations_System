from flask import Flask, request, render_template
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

app = Flask(__name__)

# Load files===========================================================================================================
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")

# Function to truncate product name====================================================================================
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

# Recommendation function=============================================================================================
def content_based_recommendation(train_data, item_name, top_n=10):
    # Normalize the input product name
    item_name = item_name.strip().lower()

    # Add a cleaned version of product names
    train_data['Name_clean'] = train_data['Name'].astype(str).str.strip().str.lower()

    # Step 1: exact case-insensitive match
    if item_name in train_data['Name_clean'].values:
        item_index = train_data[train_data['Name_clean'] == item_name].index[0]
    else:
        # Step 2: partial substring match
        partial_matches = train_data[train_data['Name_clean'].str.contains(item_name, na=False)]
        if not partial_matches.empty:
            item_index = partial_matches.index[0]
        else:
            # Step 3: fuzzy match
            all_names = train_data['Name_clean'].dropna().tolist()
            matches = get_close_matches(item_name, all_names, n=1, cutoff=0.5)
            if matches:
                item_index = train_data[train_data['Name_clean'] == matches[0]].index[0]
            else:
                print(f"❌ Item '{item_name}' not found in the training data.")
                return pd.DataFrame()

    # TF-IDF similarity
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'].astype(str))
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    similar_items = list(enumerate(cosine_similarities_content[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    top_similar_items = similar_items[1:top_n + 1]

    recommended_item_indices = [x[0] for x in top_similar_items]
    recommended_items_details = train_data.iloc[recommended_item_indices][
        ['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']
    ]

    return recommended_items_details

# List of predefined image URLs=======================================================================================
random_image_urls = [
    "static/img/img_1.png",
    "static/img/img_2.png",
    "static/img/img_3.png",
    "static/img/img_4.png",
    "static/img/img_5.png",
    "static/img/img_6.png",
    "static/img/img_7.png",
    "static/img/img_8.png",
]

# Routes==============================================================================================================
@app.route("/")
def index():
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template(
        'index.html',
        trending_products=trending_products.head(8),
        truncate=truncate,
        random_product_image_urls=random_product_image_urls,
        random_price=random.choice(price)
    )

@app.route("/main")
def main():
    return render_template('main.html', content_based_rec=pd.DataFrame())

@app.route("/index")
def indexredirect():
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template(
        'index.html',
        trending_products=trending_products.head(8),
        truncate=truncate,
        random_product_image_urls=random_product_image_urls,
        random_price=random.choice(price)
    )

@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod', '').strip()
        nbr_str = request.form.get('nbr', '').strip()

        # Handle missing or invalid "nbr"
        if not nbr_str.isdigit():
            nbr = 5  # default value if not provided
        else:
            nbr = int(nbr_str)

        content_based_rec = content_based_recommendation(train_data, prod, top_n=nbr)

        if content_based_rec.empty:
            message = "No recommendations available for this product."
            return render_template(
                'main.html',
                content_based_rec=pd.DataFrame(),
                message=message
            )
        else:
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(content_based_rec))]
            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
            return render_template(
                'main.html',
                content_based_rec=content_based_rec,
                truncate=truncate,
                random_product_image_urls=random_product_image_urls,
                random_price=random.choice(price)
            )

if __name__ == "__main__":
    app.run(debug=True)
