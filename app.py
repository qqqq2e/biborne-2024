 

 
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify  
 
 
import pandas as pd
import pickle
 
app = Flask(__name__)

 

# 
data = pd.read_csv('./call_log_202406211525.csv')

df = pd.DataFrame(data)

# Load TF-IDF vectorizer (replace with your actual preprocessing)
with open('tfidf_vectorizer_model.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

 
@app.route('/', methods=['GET'])
def get_home():
    return 'hi said'

@app.route('/a', methods=['GET'])
def get_home2():
    return 'hi said mbisd'
   

 

@app.route('/api/v1/get_similarity', methods=['POST'])

def get_similarity():
    # fun = get_top_k_unique_similar_problems()
    req_data = request.get_json()
    input_text = req_data['input_text']

    # Preprocess input text with the loaded vectorizer
    input_tfidf = vectorizer.transform([input_text])

    # Compute cosine similarities
    cosine_similarities = cosine_similarity(input_tfidf, vectorizer.transform(df['problem'])).flatten()

    # Sort and get top 5 similar problems
    top_5_indices = cosine_similarities.argsort()[-10:][::-1]
    top_5_problems = df.iloc[top_5_indices]['problem'].values
    top_5_solutions = df.iloc[top_5_indices]['solution'].values
    top_5_similarities = cosine_similarities[top_5_indices]
   
    top_k_indices = cosine_similarities.argsort()[::-1]  # sort indices in descending order of similarity
    unique_extensions = []
    unique_similarities = []
    seen_extensions = set()
   
    for index in top_k_indices:
        extension = data.iloc[index]['extension_number']
        if extension not in seen_extensions:
            unique_extensions.append(int(extension))
            unique_similarities.append(cosine_similarities[index])
            seen_extensions.add(extension)
        if len(unique_extensions) == 5:
            break

 
    print(1111)
    # Prepare response
    response = {
        'top_5_problems': top_5_problems.tolist() ,
        'top_5_problems_similarities': top_5_similarities .tolist(),
        'top_5_extension_number': unique_extensions  ,
        'top_5_solutions':top_5_solutions.tolist(),
        'top_5_extension_number_similarities': unique_similarities   
    }

    return jsonify(response)

     
















if __name__ == '__main__':
    app.run(debug=True)
