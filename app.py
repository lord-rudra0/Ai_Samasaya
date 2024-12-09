import google.generativeai as genai
from flask import Flask, request, render_template, jsonify, Response, redirect, url_for, send_from_directory, send_file, make_response, abort
import os
import pdfplumber
from google.cloud import aiplatform
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()

genai.configure(api_key=os.getenv("GOGGLE_API_KEY"))

# Directory for uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Function to divide text into chapters based on headings
def divide_into_chapters(text):
    chapters = {}
    current_chapter = "Introduction"
    for line in text.splitlines():
        # Detect "Chapter" heading
        if line.strip().lower().startswith("chapter"):
            current_chapter = line.strip()
            chapters[current_chapter] = ""
        else:
            chapters[current_chapter] = chapters.get(current_chapter, "") + line + "\n"
    return chapters

# Function to summarize a chapter
def summarize_text(text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Please summarize the following text:\n\n{text}"
    response = model.generate_content(prompt)
    return response.text

# Function to generate questions based on chapter content
def generate_questions(text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Generate 5 quiz questions based on the following text:\n\n{text}"
    response = model.generate_content(prompt)
    return response.text

# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Route to handle the uploaded PDF
@app.route('/pdf', methods=['POST'])
def handle_pdf():
    if 'pdf' not in request.files:
        return render_template("error.html", message="No PDF uploaded.")

    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return render_template("error.html", message="No file selected.")

    # Save the uploaded file
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
    pdf_file.save(pdf_path)

    # Extract text from the uploaded PDF
    text = extract_text_from_pdf(pdf_path)

    # Divide the text into chapters
    chapters = divide_into_chapters(text)

    # Summarize and generate questions for each chapter
    results = {}
    for chapter, content in chapters.items():
        summary = summarize_text(content)
        questions = generate_questions(content)
        results[chapter] = {"summary": summary, "questions": questions}

    # Render results in PDF.html
    return render_template("pdf.html", results=results)

if __name__ == '__main__':
    app.run(debug=True)
