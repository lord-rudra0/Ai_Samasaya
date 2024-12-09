import google.generativeai as genai
from google.generativeai import GenerativeModel
from google.generativeai import generate_content
from flask import Flask, request, render_template, jsonify, Response, redirect, url_for, send_from_directory, send_file, make_response, abort
import os
import json
from dotenv import load_dotenv
import pdfplumber

app = Flask(__name__)

load_dotenv()

# genai.configure(api_key="GOOGLE_API_KEY")
# model = genai.GenerativeModel("gemini-1.5-flash")
# response = model.generate_content("Explain how AI works")
# print(response.text)

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/pdf', methods=['POST'])
def pdf():
    if request.method == 'POST':
        pdf = request.files['pdf']
        pdf.save(pdf.filename)
        return render_template("pdf.html", pdf=pdf.filename)
    
@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        pdf = request.form['pdf']
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(pdf)
        return render_template("pdf.html", pdf=pdf, text=response.text)
    

    
    
