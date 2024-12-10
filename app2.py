import google.generativeai as genai
from flask import Flask, request, render_template, jsonify, Response, redirect, url_for, send_from_directory, send_file, make_response, abort, session
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
    prompt = f"""
        Act like a professional teacher and summarize the following text in a way that is:
        1.divided into chapters with clear headings for each chapter. 
        2.  Short and simple, covering all the important points.
        3. Easy to understand for students, using clear and concise language.
        4. Breaking down the text into chapters with clear headings for each chapter.
        5. each chapter should show in different paragraph. and start from new line.
        if there is a topic then give atleast 2-3 lines about it.
        5. Including all the important keywords and concepts, explained in an easy-to-digest manner.
        6. Including examples and illustrations to help students understand the concepts better. but not too many. and not too complex. and keep sort summary.
        7. Including summaries, bullet points, or key takeaways at the end of each chapter. 
        8. all the important keywords and concepts, explained in an easy-to-digest manner.
        9. if there a topic then give atleast 2-3 lines about it.
        10. if there is a list of items then give atleast 2-3 lines about it.
        11. if there is a definition then give atleast 2-3 lines about it.
        12. if there is a example then give atleast 2-3 lines about it.
        13. if there is a concept then give atleast 2-3 lines about it.
        14. if there is a theory then give atleast 2-3 lines about it.
        15. if there is a formula then give atleast 2-3 lines about it.
        16. if there is a law then give atleast 2-3 lines about it.
        17. if there is a rule then give atleast 2-3 lines about it.
        18. if there is a principle then give atleast 2-3 lines about it.
        19. if there is a theorem then give atleast 2-3 lines about it.
        20. after each chapter give  space and start from new line. 
        21. divide the text into chapters with clear headings for each chapter.
        
        Here is the text to summarize:

    {text}
    """

    response = model.generate_content(prompt)
    return response.text

# Function to generate questions based on chapter content
def generate_questions(text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    Act like a professional teacher and summarize the following text in a way that is:
    your teaching a class and you want to ask questions from the following text.
    Generate  5 random quiz questions based on the following text:
    each question should be unique and should test the students' understanding of the concepts discussed in the text.
    Each question should be clear, concise, and easy to understand, with a single correct answer.
    each question should be mcq type and should have 4 options. 
    Each question should cover a different aspect of the text, testing the students' knowledge of different concepts.
    Each question should be relevant and important, focusing on key points discussed in the text.
    Each question should be well-structured, with proper grammar, punctuation, and spelling.
    Each question should be challenging and engaging, encouraging students to think critically and apply their knowledge.
    each question should be ask for answer when answer is entered it should show if it is correct or not.
    Here is the text to generate questions from:
    generate 5 mcq from the summary of the chapter.
    
     \n\n{text}
    """
    
   
    response = model.generate_content(prompt)
    return response.text
    
# Home route
@app.route('/')
def home():
    return render_template("index.html")



@app.route('/pdf', methods=['POST'])
def handle_pdf():
    if 'pdf' not in request.files:
        return render_template("error.html", message="No PDF uploaded.")
    
    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return render_template("error.html", message="No file selected.")

    
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
    pdf_file.save(pdf_path)

    
    text = extract_text_from_pdf(pdf_path)


    chapters = divide_into_chapters(text)

    results = {}
    for chapter, content in chapters.items():
        summary = summarize_text(content)
        questions = generate_questions(content)  
        results[chapter] = {"summary": summary, "questions": questions}

    
    return render_template("pdf.html", results=results)


@app.route("/check_answers", methods=["POST"])
def check_answers():
    # Retrieve the results from session
    results = session.get('results', {})

    user_answers = request.form
    feedback = {}

    def validate_answer(question, answer):
        correct_answers = {"Question 1": 2, "Question 2": 1}  # Placeholder for actual validation logic
        return "Correct" if correct_answers.get(question) == int(answer) else "Incorrect"

    for chapter, data in user_answers.items():
        feedback_for_chapter = []
        for question_id, answer in data.items():
            question_text = results.get(chapter, {}).get('questions', {}).get(question_id, '')
            feedback_for_chapter.append({
                'question': question_text,
                'answer': answer,
                'feedback': validate_answer(question_text, answer)
            })
        feedback[chapter] = feedback_for_chapter

    return render_template("results.html", feedback=feedback)




if __name__ == '__main__':
    app.run(debug=True)
