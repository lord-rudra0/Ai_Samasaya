import os
from flask import Flask, request, render_template
import pdfplumber
from google.cloud import aiplatform
from dotenv import load_dotenv
import google.generativeai as genai

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini AI
genai.configure(api_key=os.getenv("GOGGLE_API_KEY"))

# Upload folder for PDF files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Function to divide text into chapters
def divide_into_chapters(text):
    chapters = {}
    current_chapter = "Introduction"
    for line in text.splitlines():
        if line.strip().lower().startswith("chapter"):
            current_chapter = line.strip()
            chapters[current_chapter] = ""
        else:
            chapters[current_chapter] = chapters.get(current_chapter, "") + line + "\n"
    return chapters

# Function to summarize a chapter
def summarize_text(text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
        Act like a professional teacher and summarize the following text in a way that is:
        1. Divided into chapters with clear headings for each chapter.
        2. Short and simple, covering all the important points.
        3. Easy to understand for students, using clear and concise language.
        4. Including examples, illustrations, and bullet points for clarity.
        5. Covering all essential topics and concepts with keywords explained simply.
        
        Here is the text to summarize:
        
        {text}
    """
    response = model.generate_content(prompt)
    return response.text

# Function to generate questions based on chapter content
def generate_questions(text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
        Act like a teacher and generate quiz questions based on the following text:
        - Generate 5 multiple-choice questions (MCQs) with 4 options each.
        - Ensure each question tests a unique concept or topic from the text.
        - Provide the correct answer for each question for validation.
        Here is the text:
        
        {text}
    """
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

    # Generate summaries and questions
    results = {}
    for chapter, content in chapters.items():
        summary = summarize_text(content)
        questions = generate_questions(content)
        results[chapter] = {"summary": summary, "questions": questions}

    # Render results in PDF.html
    return render_template("pdf.html", results=results)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
