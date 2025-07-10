from AdvanceCache.Caching import AdvancedCache
from Ratelimiter.Limiter import RateLimiter
from prompt_templates.Templates import build_prompt_template, build_prompt_template_pdf, extract_topic_from_pdf_content
from utils.Extraction_pdf import extract_text_from_pdf
from utils.Helpers import validate_input, validate_generated_content, generate_cache_key, allowed_file, enhance_response, calculate_difficulty_score
from utils.Debuger_pdf import debug_pdf_info
from flask_jwt_extended import create_access_token, set_access_cookies, unset_jwt_cookies
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
from werkzeug.security import generate_password_hash, check_password_hash

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

# In-memory user storage (use database in production)
users = {
    "admin": {
        "password": generate_password_hash("admin123"), 
        "role": "admin"
    },
}

# JWT Configuration
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-super-secret-key-change-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
app.config['JWT_TOKEN_LOCATION'] = ['cookies', 'headers']  # Support both cookies and headers
app.config['JWT_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['JWT_COOKIE_CSRF_PROTECT'] = False  # Set to True in production
app.config['JWT_COOKIE_SAMESITE'] = 'Lax'
app.config['JWT_COOKIE_HTTPONLY'] = True  # Prevent XSS attacks

jwt = JWTManager(app)

# CORS Configuration - More restrictive in production
CORS(
    app,
    resources={r"/*": {"origins": "*"}},  # Restrict to specific origins in production
    supports_credentials=True,
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"]
)

# App Configuration
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# API Keys with proper validation
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not GROQ_API_KEY and not DEEPSEEK_API_KEY:
    raise ValueError("Either GROQ_API_KEY or DEEPSEEK_API_KEY environment variable must be set")

# Initialize LLM with fallback logic
try:
    if GROQ_API_KEY:
        chat_model = ChatGroq(
            api_key=SecretStr(GROQ_API_KEY),
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=8192
        )
        logger.info("Using Groq model")
    elif DEEPSEEK_API_KEY:
        chat_model = ChatDeepSeek(
            api_key=SecretStr(DEEPSEEK_API_KEY),
            model="deepseek-r1-distill-llama-70b",
            temperature=0.7,
            max_tokens=8192
        )
        logger.info("Using DeepSeek model")
except Exception as e:
    logger.error(f"Failed to initialize chat model: {str(e)}")
    raise

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

# Custom JWT verification decorator
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

# Admin role checker
def admin_required(f):
    @wraps(f)
    @jwt_required_custom
    def decorated_function(*args, **kwargs):
        user = get_jwt_identity()
        if not user or user.get("role") != "admin":
            return jsonify({"error": "Admin access required"}), 403
        return f(*args, **kwargs)
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
def generate_with_count_validation(chain, params, expected_count, max_retries=3):
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
    """Home page - No authentication required for public access"""
    return render_template("Home.html")

@app.route('/dashboard')
@jwt_required_custom
def dashboard():
    """Protected dashboard for authenticated users"""
    current_user = get_jwt_identity()
    return render_template("dashboard.html", user=current_user)

@app.route('/generate_mcqs', methods=['POST'])
@jwt_required_custom
def generate_mcqs():
    """Generate MCQs with JWT authentication"""
    try:
        # Get authenticated user info
        current_user = get_jwt_identity()
        logger.info(f"MCQ generation request from user: {current_user}")
        
        # Validate request content type
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate input
        is_valid, error_msg = validate_input(data)
        if not is_valid:
            return jsonify({"error": "Validation failed", "message": error_msg}), 400
        
        # Extract parameters with proper validation
        topic = data.get("topic", "").strip()
        if not topic:
            return jsonify({"error": "Topic is required"}), 400
        
        difficulty = data.get("difficulty", "medium").lower()
        num_questions = data.get("num_questions", 5)
        question_type = data.get("question_type", "academic").lower()
        
        # Validate parameters
        if difficulty not in ["easy", "medium", "hard", "expert"]:
            return jsonify({"error": "Invalid difficulty level"}), 400
            
        if question_type not in ["academic", "practical", "conceptual"]:
            return jsonify({"error": "Invalid question type"}), 400
        
        # Validate question count
        try:
            num_questions = int(num_questions)
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid num_questions parameter"}), 400
            
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
            enhanced_data["metadata"]["user_id"] = current_user.get("username", "unknown")
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
            
            logger.info(f"Successfully generated {len(enhanced_data.get('questions', []))} questions for user: {current_user.get('username')}")
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
        
        # Get parameters with proper validation
        try:
            num_questions = int(request.form.get('num_questions', 10))
        except (ValueError, TypeError):
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
        
        logger.info(f"Processing PDF upload from user {current_user.get('username')}: {file.filename}")
        
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
            enhanced_data["metadata"]["user_id"] = current_user.get("username", "unknown")
            enhanced_data["metadata"]["source_file"] = secure_filename(file.filename)
            enhanced_data["metadata"]["content_length"] = len(text_content)
            enhanced_data["metadata"]["auto_generated_topic"] = topic
            enhanced_data["metadata"]["requested_questions"] = num_questions
            enhanced_data["metadata"]["generated_questions"] = len(enhanced_data.get("questions", []))
            
            # Cache the result
            cache.set(cache_key, enhanced_data)
            
            logger.info(f"Successfully generated {len(enhanced_data.get('questions', []))} questions from PDF for user: {current_user.get('username')}")
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
        "user": current_user.get("username", "unknown"),
        "role": current_user.get("role", "user"),
        "timestamp": datetime.now().isoformat(),
        "message": "Authentication successful"
    })

@app.route('/register', methods=['POST'])
def register():
    """User registration endpoint"""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        username = data.get("username", "").strip()
        password = data.get("password", "")
        role = data.get("role", "user").lower()

        # Validate input
        if not username or not password:
            return jsonify({"error": "Username and password are required"}), 400
        
        if len(username) < 3:
            return jsonify({"error": "Username must be at least 3 characters long"}), 400
            
        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters long"}), 400
            
        if role not in ["user", "admin"]:
            return jsonify({"error": "Invalid role. Must be 'user' or 'admin'"}), 400

        if username in users:
            return jsonify({"error": "User already exists"}), 409

        # Hash password and store user
        users[username] = {
            "password": generate_password_hash(password), 
            "role": role
        }
        
        logger.info(f"New user registered: {username} with role: {role}")
        return jsonify({"message": f"User {username} registered successfully"}), 201
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({"error": "Registration failed"}), 500

@app.route('/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        username = data.get("username", "").strip()
        password = data.get("password", "")

        if not username or not password:
            return jsonify({"error": "Username and password are required"}), 400

        user = users.get(username)
        if not user or not check_password_hash(user["password"], password):
            return jsonify({"error": "Invalid credentials"}), 401

        # Create JWT token
        access_token = create_access_token(
            identity={"username": username, "role": user["role"]}
        )
        
        response = jsonify({
            "message": "Login successful",
            "user": {"username": username, "role": user["role"]}
        })
        set_access_cookies(response, access_token)
        
        logger.info(f"User logged in: {username}")
        return response
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({"error": "Login failed"}), 500

@app.route('/logout', methods=['POST'])
@jwt_required_custom
def logout():
    """User logout endpoint"""
    try:
        current_user = get_jwt_identity()
        response = jsonify({"message": "Logged out successfully"})
        unset_jwt_cookies(response)
        
        logger.info(f"User logged out: {current_user.get('username', 'unknown')}")
        return response
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return jsonify({"error": "Logout failed"}), 500

@app.route('/admin-panel', methods=['GET'])
@admin_required
def admin_panel():
    """Admin panel endpoint - Admin access required"""
    user = get_jwt_identity()
    return jsonify({
        "message": f"Welcome to the admin panel, {user['username']}",
        "user": user,
        "total_users": len(users),
        "cache_stats": {
            "size": len(cache.cache),
            "max_size": cache.max_size,
            "ttl": cache.ttl_seconds
        }
    })

@app.route('/users', methods=['GET'])
@admin_required
def list_users():
    """List all users - Admin only"""
    user_list = []
    for username, user_data in users.items():
        user_list.append({
            "username": username,
            "role": user_data["role"]
        })
    return jsonify({"users": user_list})

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

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

if __name__ == "__main__":
    logger.info("Starting JWT-Protected MCQ Generator API v3.0.0...")
    logger.info(f"JWT Secret Key configured: {'Yes' if app.config['JWT_SECRET_KEY'] else 'No'}")
    logger.info(f"Cache configured: max_size={cache.max_size}, ttl={cache.ttl_seconds}s")
    logger.info(f"Rate limiting: {rate_limiter.max_requests} requests per hour")
    logger.info("PDF upload support: ENABLED (max 16MB)")
    logger.info("JWT Authentication: ENABLED (Cookie-based)")
    logger.info("Protected routes: /generate_mcqs, /generate_mcqs_from_pdf, /protected-health, /admin-panel")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )