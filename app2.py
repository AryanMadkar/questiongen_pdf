from AdvanceCache.Caching import AdvancedCache
from Ratelimiter.Limiter import RateLimiter
from prompt_templates.Templates import build_prompt_template, build_prompt_template_pdf, extract_topic_from_pdf_content
from utils.Extraction_pdf import extract_text_from_pdf
from utils.Helpers import validate_input, validate_generated_content, generate_cache_key, allowed_file, enhance_response, calculate_difficulty_score
from utils.Debuger_pdf import debug_pdf_info

from flask import Flask, request, jsonify, render_template
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from flask_jwt_extended import (
    JWTManager, jwt_required, get_jwt_identity, verify_jwt_in_request
)
from flask_cors import CORS
from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr
from werkzeug.utils import secure_filename

import os
import json
import hashlib
import time
import logging
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import httpx

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# JWT Configuration
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-super-secret-key')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
app.config['JWT_TOKEN_LOCATION'] = ['cookies']
app.config['JWT_COOKIE_SECURE'] = True  # Set to True in production
app.config['JWT_COOKIE_CSRF_PROTECT'] = False  # Set to True in production for CSRF protection
app.config['JWT_COOKIE_SAMESITE'] = 'Lax'

jwt = JWTManager(app)

# CORS Configuration
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True,
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)

# App Configuration
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# API Keys
api_key = os.getenv('API_KEY')
GROQ_API_KEY = os.getenv("GROQ_API_KEY", api_key)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-4123bdffbb73488a86715471da66c0a6")

if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable must be set")

# Initialize LLM
chat_model = ChatGroq(
    api_key=SecretStr(GROQ_API_KEY) if GROQ_API_KEY is not None else None,
    model="deepseek-r1-distill-llama-70b",
    temperature=0.7,
    max_tokens=8192
)

# Initialize cache and rate limiter
cache = AdvancedCache(max_size=4000, ttl_seconds=14200)
rate_limiter = RateLimiter(max_requests=50, window_seconds=3600)

# JWT Error handlers
@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({
        'error': 'Token has expired',
        'message': 'Please log in again to access this resource'
    }), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    return jsonify({
        'error': 'Invalid token',
        'message': 'Please provide a valid authentication token'
    }), 401

@jwt.unauthorized_loader
def missing_token_callback(error):
    return jsonify({
        'error': 'Authorization required',
        'message': 'Please log in to access this resource'
    }), 401

# Custom JWT verification decorator for cookies
def jwt_required_custom(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            verify_jwt_in_request()
            return f(*args, **kwargs)
        except Exception as e:
            logger.warning(f"JWT verification failed: {str(e)}")
            return jsonify({
                'error': 'Authentication failed',
                'message': 'Invalid or missing authentication token'
            }), 401
    return decorated_function

# Enhanced validation function for question count
def validate_question_count(json_data, expected_count):
    """Validate that the generated questions match the expected count"""
    if not json_data or "questions" not in json_data:
        return False, "No questions found in response"
    
    actual_count = len(json_data["questions"])
    
    # Allow small deviation for large counts
    if expected_count <= 20:
        tolerance = 0  # Exact match required
    elif expected_count <= 40:
        tolerance = 1  # Allow ±1
    else:
        tolerance = 2  # Allow ±2
    
    if abs(actual_count - expected_count) > tolerance:
        return False, f"Expected {expected_count} questions, got {actual_count}"
    
    return True, f"Generated {actual_count} questions"

# Enhanced content generation with retry logic
def generate_with_count_validation(chain, params, expected_count, max_retries=5):
    """Generate content with validation for question count"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Generation attempt {attempt + 1} for {expected_count} questions")
            
            enhanced_params = params.copy()
            enhanced_params["count_emphasis"] = f"IMPORTANT: Generate exactly {expected_count} questions. No more, no less."
            
            response = chain.invoke(enhanced_params)
            
            # Validate generated content
            is_valid_content, json_data = validate_generated_content(response)
            if is_valid_content and json_data:
                # Validate question count
                is_valid_count, count_message = validate_question_count(json_data, expected_count)
                
                if is_valid_count:
                    logger.info(f"Successfully generated {len(json_data['questions'])} questions on attempt {attempt + 1}")
                    return True, json_data
                else:
                    logger.warning(f"Question count validation failed on attempt {attempt + 1}: {count_message}")
                    
                    # If we have questions but wrong count, try to adjust
                    if "questions" in json_data:
                        actual_count = len(json_data["questions"])
                        if actual_count > expected_count:
                            # Trim excess questions
                            json_data["questions"] = json_data["questions"][:expected_count]
                            logger.info(f"Trimmed questions from {actual_count} to {expected_count}")
                            return True, json_data
                        elif actual_count < expected_count and actual_count > 0:
                            # If we're close but short, and this is the last attempt, accept it
                            shortage = expected_count - actual_count
                            if attempt == max_retries - 1 and shortage <= 3:
                                logger.warning(f"Accepting {actual_count} questions (short by {shortage}) on final attempt")
                                return True, json_data
            
            logger.warning(f"Invalid content generated on attempt {attempt + 1}")
            
        except Exception as e:
            logger.error(f"Generation attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(1)  # Brief pause before retry
    
    return False, None

# Routes
@app.route('/')
def home():
    """Home page"""
    return render_template("Home.html")

@app.route('/generate_mcqs', methods=['POST'])
@jwt_required_custom
def generate_mcqs():
    """Generate MCQs with JWT authentication"""
    try:
        # Get authenticated user info
        current_user = get_jwt_identity()
        logger.info(f"MCQ generation request from user: {current_user}")
        
        data = request.get_json()
        
        # Validate input
        is_valid, error_msg = validate_input(data)
        if not is_valid:
            return jsonify({"error": "Validation failed", "message": error_msg}), 400
        
        # Extract parameters
        topic = data["topic"].strip()
        difficulty = data.get("difficulty", "medium").lower()
        num_questions = data.get("num_questions", 5)
        question_type = data.get("question_type", "academic")
        
        # Validate question count
        if num_questions < 1 or num_questions > 60:
            return jsonify({"error": "Number of questions must be between 1 and 60"}), 400
        
        # Check rate limiting
        client_ip = request.remote_addr
        if not rate_limiter.is_allowed(client_ip):
            return jsonify({"error": "Rate limit exceeded"}), 429
        
        # Check cache
        cache_key = generate_cache_key(topic, difficulty, num_questions, question_type)
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for topic: {topic}")
            return jsonify({**cached_result, "cached": True})
        
        # Build LangChain prompt
        prompt_template = build_prompt_template(question_type)
        timestamp = datetime.now().isoformat()
        
        # Create LangChain chain
        chain = (
            RunnablePassthrough.assign(timestamp=lambda _: timestamp)
            | prompt_template
            | chat_model
            | StrOutputParser()
        )
        
        # Generate with validation
        success, json_data = generate_with_count_validation(
            chain, 
            {
                "topic": topic,
                "difficulty": difficulty,
                "num_questions": num_questions,
                "timestamp": timestamp
            }, 
            num_questions
        )
        
        if success and json_data:
            # Enhance response
            enhanced_data = enhance_response(json_data)
            
            # Add metadata
            enhanced_data["metadata"]["user_id"] = current_user
            enhanced_data["metadata"]["requested_questions"] = num_questions
            enhanced_data["metadata"]["generated_questions"] = len(enhanced_data.get("questions", []))
            
            # Add question type distribution stats
            question_types_count = {}
            for question in enhanced_data.get("questions", []):
                q_type = question.get("question_type", "unknown")
                question_types_count[q_type] = question_types_count.get(q_type, 0) + 1
            
            enhanced_data["metadata"]["question_type_distribution"] = question_types_count
            
            # Cache the result
            cache.set(cache_key, enhanced_data)
            
            logger.info(f"Successfully generated {len(enhanced_data.get('questions', []))} questions for user: {current_user}")
            return jsonify({**enhanced_data, "cached": False})
        
        return jsonify({
            "error": "Generation failed",
            "message": f"Unable to generate {num_questions} valid questions after multiple attempts"
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in generate_mcqs: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }), 500

@app.route('/generate_mcqs_from_pdf', methods=['POST'])
@jwt_required_custom
def generate_mcqs_from_pdf():
    """Generate MCQs from PDF with JWT authentication"""
    try:
        # Get authenticated user info
        current_user = get_jwt_identity()
        logger.info(f"PDF MCQ generation request from user: {current_user}")
        
        # Check if file is present
        if 'pdf_file' not in request.files:
            return jsonify({"error": "No PDF file provided"}), 400
        
        file = request.files['pdf_file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({"error": "Only PDF files are allowed"}), 400
        
        # Additional file validation
        if file.content_length and file.content_length > 16 * 1024 * 1024:
            return jsonify({"error": "File too large. Maximum size is 16MB"}), 400
        
        # Get parameters
        try:
            num_questions = int(request.form.get('num_questions', 10))
        except ValueError:
            return jsonify({"error": "Invalid num_questions parameter"}), 400
            
        difficulty = request.form.get('difficulty', 'medium').lower()
        question_type = request.form.get('question_type', 'academic').lower()
        
        # Validate parameters
        if num_questions < 1 or num_questions > 60:
            return jsonify({"error": "Number of questions must be between 1 and 60"}), 400
        
        if difficulty not in ["easy", "medium", "hard", "expert"]:
            return jsonify({"error": "Invalid difficulty level"}), 400
        
        if question_type not in ["academic", "practical", "conceptual"]:
            return jsonify({"error": "Invalid question type"}), 400
        
        # Check rate limiting
        client_ip = request.remote_addr
        if not rate_limiter.is_allowed(client_ip):
            return jsonify({"error": "Rate limit exceeded"}), 429
        
        logger.info(f"Processing PDF upload from user {current_user}: {file.filename}")
        
        # Extract text from PDF
        success, text_content = extract_text_from_pdf(file)
        if not success:
            logger.error(f"PDF processing failed: {text_content}")
            return jsonify({
                "error": "PDF processing failed", 
                "message": text_content,
                "suggestions": [
                    "Ensure the PDF is not encrypted",
                    "Check if the PDF contains extractable text",
                    "Try with a different PDF file"
                ]
            }), 400
        
        logger.info(f"Successfully extracted {len(text_content)} characters from PDF")
        
        # Generate topic from text
        topic = extract_topic_from_pdf_content(text_content)
        logger.info(f"Topic extraction result: {topic}")
        
        # Validate extracted topic
        if not topic or topic == "Document Content Analysis":
            topic = f"Document Analysis - {secure_filename(file.filename).replace('.pdf', '')}"
        
        # Check cache
        text_hash = hashlib.md5(text_content.encode()).hexdigest()
        cache_key = f"pdf_{text_hash}_{difficulty}_{num_questions}_{question_type}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            logger.info("Cache hit for PDF content")
            return jsonify({**cached_result, "cached": True})
        
        # Build LangChain prompt for PDF
        prompt_template = build_prompt_template_pdf(question_type)
        timestamp = datetime.now().isoformat()
        
        # Create LangChain chain
        chain = (
            RunnablePassthrough.assign(timestamp=lambda _: timestamp)
            | prompt_template
            | chat_model
            | StrOutputParser()
        )
        
        # Generate with validation
        success, json_data = generate_with_count_validation(
            chain, 
            {
                "topic": topic,
                "difficulty": difficulty,
                "num_questions": num_questions,
                "text_content": text_content,
                "timestamp": timestamp
            }, 
            num_questions
        )
        
        if success and json_data:
            # Enhance response
            enhanced_data = enhance_response(json_data)
            
            # Add PDF-specific metadata
            enhanced_data["metadata"]["user_id"] = current_user
            enhanced_data["metadata"]["source_file"] = secure_filename(file.filename)
            enhanced_data["metadata"]["content_length"] = len(text_content)
            enhanced_data["metadata"]["auto_generated_topic"] = topic
            enhanced_data["metadata"]["requested_questions"] = num_questions
            enhanced_data["metadata"]["generated_questions"] = len(enhanced_data.get("questions", []))
            
            # Cache the result
            cache.set(cache_key, enhanced_data)
            
            logger.info(f"Successfully generated {len(enhanced_data.get('questions', []))} questions from PDF for user: {current_user}")
            return jsonify({**enhanced_data, "cached": False})
        
        return jsonify({
            "error": "Generation failed",
            "message": f"Unable to generate {num_questions} valid questions after multiple attempts"
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in generate_mcqs_from_pdf: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": "An unexpected error occurred while processing the PDF"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint - No authentication required"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "cache_size": len(cache.cache),
        "authentication": "JWT enabled",
        "features": {
            "pdf_support": True,
            "multi_type_questions": True,
            "question_count_validation": True,
            "jwt_authentication": True
        }
    })

@app.route('/protected-health', methods=['GET'])
@jwt_required_custom
def protected_health():
    """Protected health check with user info"""
    current_user = get_jwt_identity()
    return jsonify({
        "status": "healthy",
        "user": current_user,
        "timestamp": datetime.now().isoformat(),
        "message": "Authentication successful"
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "File too large", "message": "Maximum file size is 16MB"}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    logger.info("Starting JWT-Protected MCQ Generator API v3.0.0...")
    logger.info(f"JWT Secret Key configured: {'Yes' if app.config['JWT_SECRET_KEY'] else 'No'}")
    logger.info(f"Cache configured: max_size={cache.max_size}, ttl={cache.ttl_seconds}s")
    logger.info(f"Rate limiting: {rate_limiter.max_requests} requests per hour")
    logger.info(f"LangChain model: deepseek-r1-distill-llama-70b")
    logger.info("PDF upload support: ENABLED (max 16MB)")
    logger.info("JWT Authentication: ENABLED (Cookie-based)")
    logger.info("Protected routes: /generate_mcqs, /generate_mcqs_from_pdf, /protected-health")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )