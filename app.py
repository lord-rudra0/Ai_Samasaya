import google.generativeai as genai
from flask import Flask, request, render_template, jsonify, Response, redirect, url_for, send_from_directory, send_file, make_response, abort
import os
import pdfplumber
from google.cloud import aiplatform
from dotenv import load_dotenv
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

app = Flask(__name__)

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

VIDEO_FOLDER = 'generated_videos'
IMAGE_FOLDER = 'generated_images'
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

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
        1. Divided into chapters with clear headings for each chapter.
        2. Short and simple, covering all the important points.
        3. Easy to understand for students, using clear and concise language.
        4. Breaking down the text into chapters with clear headings for each chapter.
        5. Including all the important keywords and concepts, explained in an easy-to-digest manner.
        6. Including examples and illustrations to help students understand the concepts better. but not too many, and not too complex. Keep the summary short.
        7. Including summaries, bullet points, or key takeaways at the end of each chapter.
        8. All the important keywords and concepts, explained in an easy-to-digest manner.
        9. If there is a topic, give at least 2-3 lines about it.
        10. If there is a list of items, give at least 2-3 lines about it.
        11. If there is a definition, give at least 2-3 lines about it.
        12. If there is an example, give at least 2-3 lines about it.
        13. If there is a concept, give at least 2-3 lines about it.
        14. If there is a theory, give at least 2-3 lines about it.
        15. If there is a formula, give at least 2-3 lines about it.
        16. If there is a law, give at least 2-3 lines about it.
        17. If there is a rule, give at least 2-3 lines about it.
        18. If there is a principle, give at least 2-3 lines about it.
        19. If there is a theorem, give at least 2-3 lines about it.
        20. After each chapter, leave space and start from a new line.
        Don't give too many notes, warnings, or cautions unrelated to the text.

        Here is the text to summarize:

    {text}
    """

    response = model.generate_content(prompt)
    summarized_text = response.text
    return summarized_text.split('\n\n')

# Function to generate questions based on chapter content
def generate_questions(text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    Act like a professional teacher and generate 5 to 10 random quiz questions based on the following text.
    Each question should be unique and should test the students' understanding of the concepts discussed in the text.
    Each question should be clear, concise, and easy to understand, with a single correct answer.
    Each question should be of MCQ type with 4 options.
    Each question should cover a different aspect of the text, testing the students' knowledge of different concepts.
    Each question should be well-structured, with proper grammar, punctuation, and spelling.
    Each question should be challenging and engaging, encouraging students to think critically and apply their knowledge.
    Each question should ask for an answer, and the answer should be displayed when submitted.
    Each time the page refreshes, it should show different questions. Next 5 questions should be different from the previous ones.
    The next 5 questions should be 2 or 3-line questions, not MCQs. Answer for each question is mandatory.

    Here is the text to generate questions from:

    {text}
    """
    response = model.generate_content(prompt)
    return response.text

# Function to load label mappings
def load_label_mappings(dataset_dir):
    labels = os.listdir(dataset_dir)
    labels.sort()  # Optional: Sort labels alphabetically
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    return label_to_index, index_to_label

# Function to create hand sign images from the chapter summary
def create_hand_sign_images(text, save_dir, label_to_index, index_to_label, dataset_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    created_images = []
    for char in text.upper():
        if char in label_to_index:
            label = index_to_label[label_to_index[char]]
            image_folder = os.path.join(dataset_dir, label)
            if os.path.isdir(image_folder) and os.listdir(image_folder):
                image_path = os.path.join(image_folder, os.listdir(image_folder)[0])
                image = Image.open(image_path)
                save_path = os.path.join(save_dir, f"{char}.png")
                image.save(save_path)
                created_images.append(save_path)
            else:
                created_images.append(None)  # If no image found for the label
        else:
            created_images.append(None)  # If no mapping for the character
    return created_images

# Function to create video from the images
def create_video_from_images(image_paths, output_video_path, frame_size=(64, 64), fps=1):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    for img_path in image_paths:
        if img_path is not None:
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, frame_size)
            out.write(img_resized)
        else:
            blank_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            out.write(blank_frame)

    out.release()

@app.route('/')
def home():
    return render_template('index.html')  # Ensure you have index.html in your templates folder

@app.route('/pdf', methods=['POST'])
def handle_pdf():
    if 'pdf' not in request.files:
        return render_template("error.html", message="No PDF uploaded.")

    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return render_template("error.html", message="No file selected.")

    # Save the uploaded PDF file
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
    pdf_file.save(pdf_path)

    # Extract text from the uploaded PDF
    text = extract_text_from_pdf(pdf_path)

    # Divide the text into chapters
    chapters = divide_into_chapters(text)

   
    dataset_dir = "/home/rudra-thakur/Ai_Samasaya/handdataset/"
    label_to_index, index_to_label = load_label_mappings(dataset_dir)

    video_paths = []  # Store paths of generated videos
    results = {}

    # Process each chapter
    for chapter, content in chapters.items():
        summary = summarize_text(content)
        if isinstance(summary, list):
            summary = " ".join(summary)

        questions = generate_questions(content)

        # Generate hand sign images for the chapter summary
        chunk_save_dir = os.path.join(IMAGE_FOLDER, chapter)
        hand_sign_images = create_hand_sign_images(summary, chunk_save_dir, label_to_index, index_to_label, dataset_dir)

        # Create video from the hand sign images
        chunk_video_path = os.path.join(VIDEO_FOLDER, f"{chapter}.mp4")
        create_video_from_images(hand_sign_images, chunk_video_path)

        # Store video path for later rendering
        video_url = url_for('serve_video', filename=f"{chapter}.mp4")
        results[chapter] = {
            'summary': summary,
            'questions': questions,
            'video_url': video_url
        }

    return render_template('index.html', results=results)

@app.route('/uploads/<filename>')
def serve_pdf(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/generated_videos/<filename>')
def serve_video(filename):
    return send_from_directory(VIDEO_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
