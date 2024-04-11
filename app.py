from flask import Flask, render_template, request
import pandas as pd
import pymongo
import requests

app = Flask(__name__)

client = pymongo.MongoClient(
    "mongodb+srv://#####:#######@cluster0.dgvwz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

hf_token = "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"
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

db = client.media
collection = db.jobs

# Dummy job data
jobs = [
    {"title": "Software Engineer", "location": "New York", "company": "Tech Inc."},
    {"title": "Data Scientist", "location": "San Francisco", "company": "Data Co."},
    {"title": "Product Manager", "location": "Seattle", "company": "Prod Ltd."},
]

def generate_embedding(text: str) -> list[float]:
    response = requests.post(
        embedding_url,
        headers={"Authorization": f"Bearer {hf_token}"},
        json={"inputs": text})

    if response.status_code != 200:
        raise ValueError(
            f"Request failed with status code {response.status_code}: {response.text}")

    return response.json()

@app.route('/')
def index():
    return render_template('index.html', jobs=jobs)

@app.route('/search', methods=['POST'])
def search():
    search_keyword = request.form['keyword']
    query = search_keyword
    print("job completed")
    results = collection.aggregate([
        {   "$vectorSearch": {
                "queryVector": generate_embedding(query),
                "path": "rest_embedding",
                "numCandidates": 300,
                "limit": 5,
                "index": "vector_jobs_index",
            }
        }
    ]);

    search_results = []

    for result in results:
        job = {
            "JobId": result["Job Id"],
            "Qualifications": result["Qualifications"],
            "title": result["Job Title"],
            "Company": result["Company"],
            "Experience": result["Experience"],
            "JobPortal": result["Job Portal"],
            "JobDescription": result["Job Description"]
        }

        search_results.append(job)
    
    return render_template('index.html', jobs=search_results)

@app.route('/embedding/')
def embedding():
    data_URL =  "jobs.csv" # path of the raw csv file

    review_df = pd.read_csv(data_URL)
    review_df.head()

    review_df = review_df['Experience'] + " " + review_df['Qualifications'] + " " + review_df['Salary Range'] + " " + review_df['location'] + " " + review_df['Job Title'] + " " + review_df['Job Description']
    review_df = review_df.sample(10)
    review_df["embedding"] = review_df.astype(str).apply(generate_embedding)

    # Make the index start from 0
    review_df.reset_index(drop=True)

    for doc in collection.find({'Job Id': {"$exists": True}}):
        review_df = doc['Experience'] + " " + doc['Qualifications'] + " " + doc['Salary Range'] + " " + doc['location'] + " " + doc['Job Title'] + " " + doc['Job Description']
        
        doc['rest_embedding'] = generate_embedding(review_df)
        collection.replace_one({'_id': doc['_id']}, doc)

    return "success"

if __name__ == '__main__':
    app.run(debug=True)
