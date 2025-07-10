from AdvanceCache.Caching import AdvancedCache
from Ratelimiter.Limiter import RateLimiter
from prompt_templates.Templates import build_prompt_template,build_prompt_template_pdf,extract_topic_from_pdf_content
from utils.Extraction_pdf import extract_text_from_pdf
from flask import Flask, request, jsonify, render_template,render_template_string
from utils.Helpers import validate_input,validate_generated_content,generate_cache_key,allowed_file,enhance_response,calculate_difficulty_score
from utils.Debuger_pdf import debug_pdf_info
from utils.llm_response_optimizer import optimize_llm_json_output

from flask import Flask, request, jsonify, render_template,render_template_string
from langchain_google_genai import ChatGoogleGenerativeAI
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
import PyPDF2 # type: ignore
from werkzeug.utils import secure_filename
from io import BytesIO
import tempfile
from flask_cors import CORS
from pydantic import SecretStr
import bcrypt

from flask_jwt_extended import (
    JWTManager, create_access_token,
    jwt_required, get_jwt_identity,
    get_jwt
)

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
httpx_client = httpx.Client()

app = Flask(__name__)
api_key = os.getenv('API_KEY')
json_sort_key = os.getenv("JSON_SORT_KEYS", "JSON_SORT_KEYS")

# JWT Configuration
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-super-secret-key-change-this-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
jwt = JWTManager(app)

# Google Gemini API Configuration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable must be set")

# CORS Configuration
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True,
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)

app.config[json_sort_key] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize Google Gemini Chat Model
chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    max_tokens=8192,
    convert_system_message_to_human=True
)

# Initialize cache and rate limiter
cache = AdvancedCache(max_size=4000, ttl_seconds=14200)
rate_limiter = RateLimiter(max_requests=50, window_seconds=3600)

# Simple user store (in production, use a proper database)
users = {
    "admin": {
        "password": bcrypt.hashpw("admin123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
        "role": "admin"
    },
    "user": {
        "password": bcrypt.hashpw("user123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
        "role": "user"
    }
}

# JWT token blacklist (in production, use Redis or database)
blacklisted_tokens = set()

@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    return jwt_payload['jti'] in blacklisted_tokens

# Enhanced validation function for question count
def validate_question_count(json_data, expected_count):
    """
    Validate that the generated questions match the expected count
    """
    if not json_data or "questions" not in json_data:
        return False, "No questions found in response"
    
    actual_count = len(json_data["questions"])
    
    # Allow small deviation (±2) for very large counts, but be strict for smaller counts
    if expected_count <= 20:
        tolerance = 0  # Exact match required
    elif expected_count <= 40:
        tolerance = 1  # Allow ±1
    else:
        tolerance = 2  # Allow ±2
    
    if abs(actual_count - expected_count) > tolerance:
        return False, f"Expected {expected_count} questions, got {actual_count}"
    
    return True, f"Generated {actual_count} questions"

# Enhanced content generation with retry logic for question count
def generate_with_count_validation(chain, params, expected_count, max_retries=5):
    """
    Generate content with validation for question count
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Generation attempt {attempt + 1} for {expected_count} questions")
            
            # Invoke LangChain with explicit count emphasis
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

# Authentication Routes
@app.route('/register', methods=['POST'])
def register():
    """User registration endpoint"""
    try:
        data = request.get_json()
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({'error': 'Username and password required'}), 400
        
        username = data['username']
        password = data['password']
        
        if username in users:
            return jsonify({'error': 'Username already exists'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters long'}), 400
        
        # Hash password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        # Add user (in production, save to database)
        users[username] = {
            'password': hashed_password,
            'role': 'user'
        }
        
        return jsonify({'message': 'User registered successfully'}), 201
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({'error': 'Username and password required'}), 400
        
        username = data['username']
        password = data['password']
        
        if username not in users:
            return jsonify({'error': 'Invalid username or password'}), 401
        
        user = users[username]
        if not bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            return jsonify({'error': 'Invalid username or password'}), 401
        
        # Create access token
        access_token = create_access_token(
            identity=username,
            additional_claims={'role': user['role']}
        )
        
        return jsonify({
            'access_token': access_token,
            'user': {
                'username': username,
                'role': user['role']
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """User logout endpoint"""
    try:
        token = get_jwt()
        jti = token['jti']
        blacklisted_tokens.add(jti)
        return jsonify({'message': 'Successfully logged out'}), 200
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return jsonify({'error': 'Logout failed'}), 500

@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    """Protected route example"""
    current_user = get_jwt_identity()
    return jsonify({'message': f'Hello {current_user}', 'user': current_user}), 200

@app.route('/info')
def home():
    """API documentation page"""
    return render_template("index.html")

@app.route('/')
def home2():
    """API documentation page"""
    return render_template("Home.html")

@app.route('/generate_mcqs_from_pdf', methods=['POST'])
@jwt_required()
def generate_mcqs_from_pdf():
    """Generate MCQs from uploaded PDF file with enhanced question types and improved topic extraction"""
    try:
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
        
        # Get optional parameters
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
        
        logger.info(f"Processing PDF upload: {file.filename}, Size: {file.content_length if file.content_length else 'Unknown'}")
        
        # Debug PDF info (optional - can be removed in production)
        debug_info = debug_pdf_info(file)
        logger.info(f"PDF Debug Info: {debug_info}")
        
        # Extract text from PDF
        success, text_content = extract_text_from_pdf(file)
        if not success:
            logger.error(f"PDF processing failed: {text_content}")
            return jsonify({
                "error": "PDF processing failed", 
                "message": text_content,
                "debug_info": debug_info,
                "suggestions": [
                    "Ensure the PDF is not encrypted",
                    "Check if the PDF contains extractable text (not just images)",
                    "Try with a different PDF file",
                    "Ensure the PDF is not corrupted"
                ]
            }), 400
        
        logger.info(f"Successfully extracted {len(text_content)} characters from PDF")
        logger.debug(f"Text sample: {text_content[:200]}...")
        
        # Generate topic from text using enhanced extraction
        topic = extract_topic_from_pdf_content(text_content)
        logger.info(f"Enhanced topic extraction result: {topic}")
        
        # Validate extracted topic
        if not topic or topic == "Document Content Analysis":
            logger.warning("Could not extract meaningful topic, using fallback")
            # Try to get a better topic by analyzing content
            words = text_content.lower().split()
            if len(words) > 50:
                # Use first meaningful sentence as topic
                sentences = text_content.split('.')
                for sentence in sentences[:3]:
                    sentence = sentence.strip()
                    if 20 <= len(sentence) <= 100 and not sentence.lower().startswith(('the', 'a', 'an')):
                        topic = sentence
                        break
            if not topic or topic == "Document Content Analysis":
                topic = f"Document Analysis - {secure_filename(file.filename).replace('.pdf', '')}"
        
        # Check cache (based on text hash)
        text_hash = hashlib.md5(text_content.encode()).hexdigest()
        cache_key = f"pdf_{text_hash}_{difficulty}_{num_questions}_{question_type}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            logger.info("Cache hit for PDF content")
            return jsonify({**cached_result, "cached": True})
        
        # Build LangChain prompt for PDF content with enhanced question types
        prompt_template = build_prompt_template_pdf(question_type)
        timestamp = datetime.now().isoformat()
        
        # Create LangChain chain
        chain = (
            RunnablePassthrough.assign(timestamp=lambda _: timestamp)
            | prompt_template
            | chat_model
            | StrOutputParser()
        )
        
        # Generate with enhanced retry logic and count validation
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
            # Enhance response with PDF-specific metadata
            enhanced_data = enhance_response(json_data)
            
            # Add PDF-specific metadata with enhanced question types info
            enhanced_data["metadata"]["source_file"] = secure_filename(file.filename) # type: ignore
            enhanced_data["metadata"]["content_length"] = len(text_content)
            enhanced_data["metadata"]["auto_generated_topic"] = topic
            enhanced_data["metadata"]["extraction_method"] = "Enhanced PyPDF2 with NLP"
            enhanced_data["metadata"]["topic_extraction_strategy"] = "Multi-strategy analysis"
            enhanced_data["metadata"]["requested_questions"] = num_questions
            enhanced_data["metadata"]["generated_questions"] = len(enhanced_data.get("questions", []))
            enhanced_data["metadata"]["generated_by"] = current_user
            
            # Add question type distribution stats
            question_types_count = {}
            for question in enhanced_data.get("questions", []):
                q_type = question.get("question_type", "unknown")
                question_types_count[q_type] = question_types_count.get(q_type, 0) + 1
            
            enhanced_data["metadata"]["question_type_distribution"] = question_types_count
            enhanced_data["metadata"]["enhanced_features"] = {
                "multi_type_questions": True,
                "subject_performance_tracking": True,
                "question_categories": ["general_knowledge", "quantitative_aptitude", "verbal_ability", "technical", "logical_reasoning"]
            }
            
            # Cache the result
            cache.set(cache_key, enhanced_data)
            
            logger.info(f"Successfully generated {len(enhanced_data.get('questions', []))} questions with enhanced types for topic: {topic}")
            logger.info(f"Question type distribution: {question_types_count}")
            return jsonify({**enhanced_data, "cached": False})
        
        return jsonify({
            "error": "Generation failed",
            "message": f"Unable to generate {num_questions} valid questions after multiple attempts",
            "debug_info": {
                "topic": topic,
                "text_length": len(text_content),
                "text_sample": text_content[:200] + "..." if len(text_content) > 200 else text_content,
                "extraction_method": "Enhanced multi-strategy"
            }
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in generate_mcqs_from_pdf: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": "An unexpected error occurred while processing the PDF",
            "details": str(e)
        }), 500

@app.route('/debug_pdf', methods=['POST'])
@jwt_required()
def debug_pdf():
    """Debug endpoint to check PDF processing with enhanced topic extraction"""
    try:
        current_user = get_jwt_identity()
        logger.info(f"PDF debug request from user: {current_user}")
        
        if 'pdf_file' not in request.files:
            return jsonify({"error": "No PDF file provided"}), 400
        
        file = request.files['pdf_file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Get PDF debug info
        debug_info = debug_pdf_info(file)
        
        # Try text extraction
        success, text_content = extract_text_from_pdf(file)
        
        result = {
            "filename": file.filename,
            "debug_info": debug_info,
            "extraction_success": success,
            "extraction_result": text_content if success else None,
            "extraction_error": text_content if not success else None,
            "text_length": len(text_content) if success else 0,
            "text_sample": text_content[:500] + "..." if success and len(text_content) > 500 else text_content if success else None,
            "debugged_by": current_user
        }
        
        if success:
            # Try enhanced topic generation
            topic = extract_topic_from_pdf_content(text_content)
            result["enhanced_topic_extraction"] = topic
            
            # Show extraction strategies used
            lines = text_content.strip().split('\n')
            result["topic_extraction_debug"] = {
                "first_5_lines": lines[:5] if lines else [],
                "text_word_count": len(text_content.split()),
                "extraction_strategy_used": "Multi-strategy analysis"
            }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Debug PDF error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_mcqs', methods=['POST'])
@jwt_required()
def generate_mcqs():
    """Enhanced MCQ generation endpoint with improved question types and count validation"""
    try:
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
        
        # Build LangChain prompt with enhanced question types
        prompt_template = build_prompt_template(question_type)
        timestamp = datetime.now().isoformat()
        
        # Create LangChain chain
        chain = (
            RunnablePassthrough.assign(timestamp=lambda _: timestamp)
            | prompt_template
            | chat_model
            | StrOutputParser()
        )
        
        # Generate with enhanced retry logic and count validation
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
            
            # Add metadata with question count info
            enhanced_data["metadata"]["requested_questions"] = num_questions
            enhanced_data["metadata"]["generated_questions"] = len(enhanced_data.get("questions", []))
            enhanced_data["metadata"]["generated_by"] = current_user
            
            # Add question type distribution stats
            question_types_count = {}
            for question in enhanced_data.get("questions", []):
                q_type = question.get("question_type", "unknown")
                question_types_count[q_type] = question_types_count.get(q_type, 0) + 1
            
            enhanced_data["metadata"]["question_type_distribution"] = question_types_count
            enhanced_data["metadata"]["enhanced_features"] = {
                "multi_type_questions": True,
                "subject_performance_tracking": True,
                "question_categories": ["general_knowledge", "quantitative_aptitude", "verbal_ability", "technical", "logical_reasoning"]
            }
            
            # Cache the result
            cache.set(cache_key, enhanced_data)
            
            logger.info(f"Successfully generated {len(enhanced_data.get('questions', []))} questions for topic: {topic}")
            logger.info(f"Question type distribution: {question_types_count}")
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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with enhanced features info"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.5.0",
        "cache_size": len(cache.cache),
        "uptime": "Available",
        "langchain": True,
        "pdf_support": True,
        "ai_model": "Google Gemini 1.5 Flash",
        "jwt_auth": True,
        "enhanced_features": {
            "multi_type_questions": True,
            "enhanced_pdf_extraction": True,
            "subject_performance_tracking": True,
            "question_count_validation": True,
            "question_categories": ["general_knowledge", "quantitative_aptitude", "verbal_ability", "technical", "logical_reasoning"]
        }
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get API statistics with enhanced features"""
    return jsonify({
        "cache_stats": {
            "current_size": len(cache.cache),
            "max_size": cache.max_size,
            "ttl_seconds": cache.ttl_seconds
        },
        "rate_limit_stats": {
            "max_requests_per_hour": rate_limiter.max_requests,
            "window_seconds": rate_limiter.window_seconds
        },
        "supported_features": {
            "question_types": ["academic", "practical", "conceptual"],
            "difficulty_levels": ["easy", "medium", "hard", "expert"],
            "max_questions": 60,
            "question_count_validation": True,
            "bloom_taxonomy": True,
            "explanations": True,
            "analytics": True,
            "langchain_integration": True,
            "pdf_upload": True,
            "max_file_size_mb": 16,
            "ai_model": "Google Gemini 1.5 Flash",
            "jwt_authentication": True,
            "enhanced_question_categories": {
                "general_knowledge": "Broad factual information and common knowledge",
                "quantitative_aptitude": "Mathematical calculations and numerical reasoning",
                "verbal_ability": "Language skills, comprehension, vocabulary",
                "technical": "Subject-specific technical concepts and procedures",
                "logical_reasoning": "Critical thinking, patterns, logical deduction"
            },
            "enhanced_pdf_processing": {
                "multi_strategy_topic_extraction": True,
                "improved_content_analysis": True,
                "better_fallback_mechanisms": True
            }
        }
    })

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

# JWT Error Handlers
@app.errorhandler(422)
def handle_unprocessable_entity(e):
    return jsonify({"error": "Invalid token"}), 422

@app.errorhandler(401)
def handle_unauthorized(e):
    return jsonify({"error": "Unauthorized - Invalid or missing token"}), 401

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({"error": "Token has expired"}), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    return jsonify({"error": "Invalid token"}), 401

@jwt.unauthorized_loader
def missing_token_callback(error): 
    return jsonify({"error": "Authorization token is required"}), 401

if __name__ == "__main__":
    logger.info("Starting Enhanced MCQ Generator API v2.5.0 with Google Gemini and JWT Authentication...")
    logger.info(f"Cache configured: max_size={cache.max_size}, ttl={cache.ttl_seconds}s")
    logger.info(f"Rate limiting: {rate_limiter.max_requests} requests per hour")
    logger.info(f"AI Model: Google Gemini 1.5 Flash")
    logger.info("PDF upload support: ENABLED (max 16MB)")
    logger.info("JWT Authentication: ENABLED")
    logger.info("Enhanced features: Multi-type questions, Enhanced PDF extraction, Subject performance tracking, Question count validation")
    logger.info("Question categories: General Knowledge, Quantitative Aptitude, Verbal Ability, Technical, Logical Reasoning")
    logger.info("Default users: admin/admin123, user/user123")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )