from AdvanceCache.Caching import AdvancedCache
from Ratelimiter.Limiter import RateLimiter
from prompt_templates.Templates import build_prompt_template, build_prompt_template_pdf, extract_topic_from_pdf_content
from utils.Extraction_pdf import extract_text_from_pdf
from flask import Flask, request, jsonify, render_template
from utils.Helpers import validate_input, validate_generated_content, generate_cache_key, allowed_file, enhance_response, calculate_difficulty_score
from utils.Debuger_pdf import debug_pdf_info

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import json
import hashlib
import time
import logging
from datetime import datetime, timedelta
from functools import wraps
import re
from typing import Dict, List, Optional, Tuple
import threading
from collections import defaultdict
from dotenv import load_dotenv
import httpx
import PyPDF2
from werkzeug.utils import secure_filename
from io import BytesIO
import tempfile
from flask_cors import CORS
from pydantic import SecretStr
import asyncio
import concurrent.futures

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
api_key = os.getenv('API_KEY')
json_sort_key = os.getenv("JSON_SORT_KEYS", "JSON_SORT_KEYS")

# OpenAI API key configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", api_key)
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable must be set")

CORS(
    app,
    resources={r"/*": {"origins": ["*"]}},
    supports_credentials=True,
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)

app.config[json_sort_key] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Optimized configuration for batching
chat_model = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=4096,
    request_timeout=45
)

cache = AdvancedCache(max_size=2000, ttl_seconds=7200)
rate_limiter = RateLimiter(max_requests=100, window_seconds=3600)

# Optimized batching configuration
BATCH_SIZE = 30
MAX_QUESTIONS = 60
BATCH_DELAY = 1.0
MIN_BATCH_THRESHOLD = 25

def create_optimized_prompt(topic, difficulty, batch_size, question_type="academic"):
    """Create highly optimized prompt for maximum efficiency"""
    return f"""Generate {batch_size} MCQs on {topic} ({difficulty} level).

JSON format only:
{{
"questions": [
  {{
    "question": "Question text?",
    "options": ["A) Option1", "B) Option2", "C) Option3", "D) Option4"],
    "correct_answer": "A",
    "explanation": "Brief reason",
    "difficulty": "{difficulty}",
    "type": "{question_type}"
  }}
]
}}

Rules:
- Exactly {batch_size} unique questions
- Valid JSON only
- Concise content
- No extra text"""

def extract_json_from_response(response_text):
    """Extract and parse JSON from response"""
    try:
        # Clean the response
        cleaned = response_text.strip()
        
        # Find JSON block
        json_start = cleaned.find('{')
        json_end = cleaned.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            return None
            
        json_str = cleaned[json_start:json_end]
        
        # Parse JSON
        data = json.loads(json_str)
        
        if "questions" in data and isinstance(data["questions"], list):
            return data["questions"]
            
    except Exception as e:
        logger.error(f"JSON extraction error: {str(e)}")
        
    return None

def generate_single_batch(topic, difficulty, batch_size, question_type, batch_num):
    """Generate a single batch with optimized error handling"""
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Batch {batch_num}, attempt {attempt + 1}: generating {batch_size} questions")
            
            prompt = create_optimized_prompt(topic, difficulty, batch_size, question_type)
            
            # Direct API call
            response = chat_model.invoke(prompt)
            
            # Extract content
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse questions
            questions = extract_json_from_response(content)
            
            if questions and len(questions) > 0:
                # Validate questions
                valid_questions = []
                for q in questions:
                    if (isinstance(q, dict) and 
                        "question" in q and 
                        "options" in q and 
                        "correct_answer" in q and
                        isinstance(q["options"], list) and
                        len(q["options"]) >= 4):
                        valid_questions.append(q)
                
                if valid_questions:
                    logger.info(f"Batch {batch_num} success: {len(valid_questions)} valid questions")
                    return valid_questions
            
            logger.warning(f"Batch {batch_num}, attempt {attempt + 1} failed")
            
        except Exception as e:
            logger.error(f"Batch {batch_num}, attempt {attempt + 1} error: {str(e)}")
            
        if attempt < max_retries - 1:
            time.sleep(1)
    
    logger.error(f"Batch {batch_num} failed after {max_retries} attempts")
    return []

def generate_questions_batched(topic, difficulty, total_questions, question_type="academic"):
    """Generate questions using optimized batching"""
    try:
        all_questions = []
        
        # Calculate batches
        full_batches = total_questions // BATCH_SIZE
        remainder = total_questions % BATCH_SIZE
        
        logger.info(f"Generating {total_questions} questions: {full_batches} full batches + {remainder} remainder")
        
        # Generate full batches
        for batch_num in range(full_batches):
            batch_questions = generate_single_batch(topic, difficulty, BATCH_SIZE, question_type, batch_num + 1)
            all_questions.extend(batch_questions)
            
            # Add delay between batches
            if batch_num < full_batches - 1 or remainder > 0:
                time.sleep(BATCH_DELAY)
        
        # Generate remainder batch if needed
        if remainder > 0:
            batch_questions = generate_single_batch(topic, difficulty, remainder, question_type, full_batches + 1)
            all_questions.extend(batch_questions)
        
        # Remove duplicates and trim to exact count
        unique_questions = []
        seen_questions = set()
        
        for q in all_questions:
            q_text = q.get("question", "").strip().lower()
            if q_text not in seen_questions:
                seen_questions.add(q_text)
                unique_questions.append(q)
                
                if len(unique_questions) >= total_questions:
                    break
        
        logger.info(f"Generated {len(unique_questions)} unique questions out of {total_questions} requested")
        return unique_questions
        
    except Exception as e:
        logger.error(f"Batched generation error: {str(e)}")
        return []

def generate_questions_direct(topic, difficulty, num_questions, question_type="academic"):
    """Direct generation for smaller question counts"""
    try:
        questions = generate_single_batch(topic, difficulty, num_questions, question_type, 1)
        return questions[:num_questions] if questions else []
    except Exception as e:
        logger.error(f"Direct generation error: {str(e)}")
        return []

def create_response_data(questions, topic, difficulty, question_type, num_questions):
    """Create standardized response data"""
    return {
        "questions": questions,
        "metadata": {
            "topic": topic,
            "difficulty": difficulty,
            "question_type": question_type,
            "requested_questions": num_questions,
            "generated_questions": len(questions),
            "model_used": "gpt-4o-mini",
            "timestamp": datetime.now().isoformat(),
            "generation_method": "batched" if num_questions >= MIN_BATCH_THRESHOLD else "direct",
            "success_rate": f"{len(questions)}/{num_questions}"
        },
        "summary": {
            "total_questions": len(questions),
            "difficulty_level": difficulty,
            "completion_status": "complete" if len(questions) >= num_questions * 0.8 else "partial"
        }
    }

@app.route('/generate_mcqs', methods=['POST'])
def generate_mcqs():
    """Optimized MCQ generation endpoint"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or "topic" not in data:
            return jsonify({"error": "Topic is required"}), 400
        
        # Extract parameters
        topic = data["topic"].strip()
        difficulty = data.get("difficulty", "medium").lower()
        num_questions = int(data.get("num_questions", 5))
        question_type = data.get("question_type", "academic").lower()
        
        # Validate parameters
        if not topic:
            return jsonify({"error": "Topic cannot be empty"}), 400
        
        if num_questions < 1 or num_questions > MAX_QUESTIONS:
            return jsonify({"error": f"Questions must be between 1 and {MAX_QUESTIONS}"}), 400
        
        if difficulty not in ["easy", "medium", "hard", "expert"]:
            return jsonify({"error": "Invalid difficulty level"}), 400
        
        # Rate limiting
        client_ip = request.remote_addr
        if not rate_limiter.is_allowed(client_ip):
            return jsonify({"error": "Rate limit exceeded"}), 429
        
        # Check cache
        cache_key = f"mcq_v2_{hashlib.md5(f'{topic}_{difficulty}_{num_questions}_{question_type}'.encode()).hexdigest()}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for topic: {topic}")
            return jsonify({**cached_result, "cached": True})
        
        # Generate questions
        if num_questions >= MIN_BATCH_THRESHOLD:
            questions = generate_questions_batched(topic, difficulty, num_questions, question_type)
        else:
            questions = generate_questions_direct(topic, difficulty, num_questions, question_type)
        
        if not questions:
            return jsonify({
                "error": "Generation failed",
                "message": "Unable to generate questions. Please try again."
            }), 500
        
        # Create response
        response_data = create_response_data(questions, topic, difficulty, question_type, num_questions)
        
        # Cache result
        cache.set(cache_key, response_data)
        
        logger.info(f"Generated {len(questions)}/{num_questions} questions for topic: {topic}")
        return jsonify({**response_data, "cached": False})
        
    except ValueError as e:
        return jsonify({"error": "Invalid input", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"Generate MCQs error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/generate_mcqs_from_pdf', methods=['POST'])
def generate_mcqs_from_pdf():
    """Optimized PDF MCQ generation"""
    try:
        # File validation
        if 'pdf_file' not in request.files:
            return jsonify({"error": "No PDF file provided"}), 400
        
        file = request.files['pdf_file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Only PDF files allowed"}), 400
        
        # Parameters
        num_questions = int(request.form.get('num_questions', 10))
        difficulty = request.form.get('difficulty', 'medium').lower()
        question_type = request.form.get('question_type', 'academic').lower()
        
        # Validate parameters
        if num_questions < 1 or num_questions > MAX_QUESTIONS:
            return jsonify({"error": f"Questions must be between 1 and {MAX_QUESTIONS}"}), 400
        
        # Rate limiting
        client_ip = request.remote_addr
        if not rate_limiter.is_allowed(client_ip):
            return jsonify({"error": "Rate limit exceeded"}), 429
        
        # Extract text from PDF
        success, text_content = extract_text_from_pdf(file)
        if not success:
            return jsonify({"error": "PDF processing failed", "message": text_content}), 400
        
        # Generate topic
        topic = extract_topic_from_pdf_content(text_content)
        if not topic:
            topic = f"Document Content - {secure_filename(file.filename).replace('.pdf', '')}"
        
        # Check cache
        text_hash = hashlib.md5(text_content.encode()).hexdigest()
        cache_key = f"pdf_v2_{text_hash}_{difficulty}_{num_questions}_{question_type}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return jsonify({**cached_result, "cached": True})
        
        # Generate questions
        if num_questions >= MIN_BATCH_THRESHOLD:
            questions = generate_questions_batched(topic, difficulty, num_questions, question_type)
        else:
            questions = generate_questions_direct(topic, difficulty, num_questions, question_type)
        
        if not questions:
            return jsonify({
                "error": "Generation failed",
                "message": "Unable to generate questions from PDF"
            }), 500
        
        # Create response
        response_data = create_response_data(questions, topic, difficulty, question_type, num_questions)
        response_data["metadata"]["source_file"] = secure_filename(file.filename)
        response_data["metadata"]["content_length"] = len(text_content)
        
        # Cache result
        cache.set(cache_key, response_data)
        
        logger.info(f"Generated {len(questions)}/{num_questions} questions from PDF: {file.filename}")
        return jsonify({**response_data, "cached": False})
        
    except ValueError as e:
        return jsonify({"error": "Invalid input", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"PDF MCQ generation error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.2.0-optimized",
        "cache_size": len(cache.cache),
        "model": "gpt-4o-mini",
        "batch_size": BATCH_SIZE,
        "max_questions": MAX_QUESTIONS,
        "features": {
            "batching": True,
            "pdf_support": True,
            "caching": True,
            "rate_limiting": True
        }
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """API statistics"""
    return jsonify({
        "cache_stats": {
            "size": len(cache.cache),
            "max_size": cache.max_size,
            "ttl": cache.ttl_seconds
        },
        "batching_config": {
            "batch_size": BATCH_SIZE,
            "max_questions": MAX_QUESTIONS,
            "threshold": MIN_BATCH_THRESHOLD,
            "delay": BATCH_DELAY
        },
        "model_info": {
            "name": "gpt-4o-mini",
            "max_tokens": 4096,
            "temperature": 0.7
        }
    })

@app.route('/info')
def info():
    """API documentation"""
    return render_template("index.html")

@app.route('/')
def home():
    """Home page"""
    return render_template("Home.html")

@app.route('/debug_pdf', methods=['POST'])
def debug_pdf():
    """Debug PDF processing"""
    try:
        if 'pdf_file' not in request.files:
            return jsonify({"error": "No PDF file provided"}), 400
        
        file = request.files['pdf_file']
        debug_info = debug_pdf_info(file)
        success, text_content = extract_text_from_pdf(file)
        
        return jsonify({
            "filename": file.filename,
            "debug_info": debug_info,
            "extraction_success": success,
            "text_length": len(text_content) if success else 0,
            "text_sample": text_content[:200] if success else None
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    logger.info("Starting Optimized MCQ Generator v3.2.0")
    logger.info(f"Model: gpt-4o-mini | Batch Size: {BATCH_SIZE} | Max Questions: {MAX_QUESTIONS}")
    logger.info("Features: Batching, PDF Support, Caching, Rate Limiting")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )