 

 
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify  
# import spacy
from flask_cors import CORS

 
import pandas as pd
import pickle
import assemblyai as aai
 

app = Flask(__name__)
 
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://biborne-2024.onrender.com/api/v1/get_similarity",],
        "methods": ["POST"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})


# 
data = pd.read_csv('./call_log_202406211525.csv')

df = pd.DataFrame(data)
# nlp = spacy.load("en_core_web_md")

# def find_most_similar(text, text_list):
#     # Process the input text
#     doc1 = nlp(text)
    
#     # Initialize variables to track the most similar text and its score
#     most_similar_text = None
#     highest_similarity = 0.0
    
#     # Iterate through the list of texts and compute similarity
#     for candidate_text in text_list:
#         doc2 = nlp(candidate_text)
#         similarity = doc1.similarity(doc2)
#         if similarity > highest_similarity:
#             highest_similarity = similarity
#             most_similar_text = candidate_text
            
#     return most_similar_text


# Load TF-IDF vectorizer (replace with your actual preprocessing)
with open('tfidf_vectorizer_model.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():

    return render_template('index.html')
# Replace with your AssemblyAI API key
aai.settings.api_key = "9d3a5b1f02ad41cf8e11c4dcf9771bc9"

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        # Get the audio file URL from the request
        data = request.get_json()
        audio_url = data.get('audio_url')
        
        if not audio_url:
            return jsonify({"error": "Audio URL is required"}), 400

        # Configuration for transcription
        config = aai.TranscriptionConfig(speaker_labels=True,language_code='fr')
        
        # Transcribe the audio from the URL
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_url, config=config)
        
        # Format the response
        result = {
            "transcript": [
                {"speaker": utterance.speaker, "text": utterance.text}
                for utterance in transcript.utterances
            ]
        }
        print(jsonify(result));
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/audio', methods=['POST'])
def audio():
    if 'audio_url' not in request.json:
        return jsonify({'error': 'No audio_url provided'}), 400

    audio_url = request.json['audio_url']

    transcriber = aai.Transcriber()

    try:
        # Configure for French language transcription
        config = aai.TranscriptionConfig(language_code="fr")
        transcript = transcriber.transcribe(audio_url, config=config)
        
        # Polling for the transcription result
        while transcript.status == aai.TranscriptStatus.processing:
            transcript = transcriber.get_transcript(transcript.id)
            
        if transcript.status == aai.TranscriptStatus.error:
            return jsonify({'error': transcript.error}), 500
        
        return jsonify({'text': transcript.text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/highlight', methods=['POST'])
def highlight():
    if 'audio_url' not in request.json:
        return jsonify({'error': 'No audio_url provided'}), 400

    audio_url = request.json['audio_url']
    try:
        # Configure the transcription with auto_highlights
        
        config = aai.TranscriptionConfig(auto_highlights=True, language_code="fr")
        transcriber = aai.Transcriber()
        
        # Start transcription
        transcript = transcriber.transcribe(
            audio_url,
            config=config
        )
        
        # Poll for completion
        while transcript.status == 'queued' or transcript.status == 'processing':
            transcript = transcriber.get_transcript(transcript.id)
        
        if transcript.status == 'failed':
            return jsonify({'error': 'Transcription failed'}), 500

        # Extract auto_highlight results
        highlights = transcript.auto_highlights.results if transcript.auto_highlights else []
        highlight_results = [{'text': result.text, 'count': result.count, 'rank': result.rank} for result in highlights]

        return jsonify({
            'text': transcript.text,
            'highlights': highlight_results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
# @app.route('/find_similar', methods=['POST'])
# def find_similar():
#     data = request.json
    
#     # Extract the text and the list of texts from the request
#     text = data.get('text')
#     text_list = data.get('text_list')
    
#     if not text or not text_list:
#         return jsonify({'error': 'Invalid input'}), 400
    
#     # Find the most similar text
#     most_similar = find_most_similar(text, text_list)
    
#     return jsonify({'most_similar': most_similar})


# @app.route('/test', methods=['POST'])
# def test():
#     data = request.json
    
#     # Extract the text and the list of texts from the request
#     text = data.get('text')
#     text_list = data.get('text_list')
    
#     if not text or not text_list:
#         return jsonify({'error': 'Invalid input'}), 400
    
#     # Find the most similar text
#     most_similar = find_most_similar(text, text_list)
    
#     return jsonify({'most_similar': most_similar})

def make_fetch_request(url, headers, method='GET', data=None):
    if method == 'POST':
        response = requests.post(url, headers=headers, json=data)
    else:
        response = requests.get(url, headers=headers)
    return response.json()

 

@app.route('/api/v1/get_similarity', methods=['POST'])

def get_similarity():
    # fun = get_top_k_unique_similar_problems()
    df['solution'] = df['solution'].fillna('')

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
    top_5_extension_resolu = df.iloc[top_5_indices]['extension_number'].values

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
        'top_5_extension_number_similarities': unique_similarities   ,
        'top_5_extension_resolu':top_5_extension_resolu.tolist(),
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True,port=5001)
      