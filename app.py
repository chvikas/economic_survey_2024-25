from flask import Flask, render_template, request, redirect, url_for, session
from utils.data_loader import load_and_split_data
from utils.pinecone_setup import setup_pinecone_vectorstore
from utils.llm_setup import setup_llm
import markdown
app = Flask(__name__)
app.secret_key = 'your_very_secure_secret_key'  # Replace with actual secret key
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Security setting
app.config['SESSION_REFRESH_EACH_REQUEST'] = True

# Load data
file_path = 'chunks.json'
splits = load_and_split_data(file_path)
# Setup pinecone
docsearch = setup_pinecone_vectorstore(splits)
# Set the LLM Model
qa = setup_llm(docsearch, llm_name="groq")

# Add cache-control headers to prevent browser caching
@app.after_request
def add_no_cache(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        session.clear()  # Clear only at start of new conversation
        query = request.form.get("query", "").strip()
        
        if query:
            try:
                result = qa.invoke(query)
                # Safer parsing with error handling
                raw_text = result["result"].split("</think>")[1]
                # Convert markdown to HTML
                html_content = markdown.markdown(raw_text)
                session['result'] = html_content
            except (KeyError, IndexError) as e:
                session["result"] = "Error processing response"
        
        return redirect(url_for('index'), code=303)

    # Get result AFTER processing POST (if any)
    result = session.pop("result", None)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)