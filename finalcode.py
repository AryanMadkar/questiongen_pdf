from flask import Flask, request, jsonify, render_template_string
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import json
import hashlib
import time
import logging
from datetime import datetime
import threading
from collections import defaultdict
from dotenv import load_dotenv
import httpx
import PyPDF2
import pdfplumber
from werkzeug.utils import secure_filename
from io import BytesIO
import tempfile
import redis
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import concurrent.futures
from typing import Dict, List, Optional, Tuple

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcq_generator.log')
    ]
)
logger = logging.getLogger(__name__)
httpx_client = httpx.Client(timeout=30.0)

app = Flask(__name__)
api_key = os.getenv('API_KEY')
json_sort_key = os.getenv("JSON_SORT_KEYS", "JSON_SORT_KEYS")

app.config[json_sort_key] = False
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", api_key)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Initialize Redis connection
try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()  # Test connection
    logger.info("Redis connection successful")
except redis.ConnectionError:
    logger.warning("Redis connection failed - using in-memory cache")
    redis_client = None

# Initialize LangChain components
chat_model = ChatGroq(
    temperature=0.2,
    model="llama3-70b-8192",
    api_key=GROQ_API_KEY,
    max_tokens=8000,
    model_kwargs={"top_p": 0.9},  # Proper way to handle top_p
    http_client=httpx_client
)

# Initialize NLP components
stop_words = set(stopwords.words('english'))

# Enhanced caching system with Redis backend
class AdvancedCache:
    def __init__(self, namespace="mcq_cache", ttl_seconds=86400):
        self.namespace = namespace
        self.ttl_seconds = ttl_seconds
        self.lock = threading.Lock()
        self.memory_cache = {}  # Fallback if Redis fails
        self.access_times = {}
    
    def _make_key(self, key):
        return f"{self.namespace}:{key}"
    
    def get(self, key):
        with self.lock:
            # Try Redis first
            if redis_client:
                try:
                    cache_key = self._make_key(key)
                    cached = redis_client.get(cache_key)
                    if cached:
                        if isinstance(cached, (bytes, bytearray)):
                            cached = cached.decode('utf-8')
                        return json.loads(cached)
                except redis.RedisError as e:
                    logger.warning(f"Redis get failed: {str(e)}")
            
            # Fallback to memory cache
            if key in self.memory_cache:
                # Check TTL for memory cache
                if time.time() - self.access_times.get(key, 0) > self.ttl_seconds:
                    del self.memory_cache[key]
                    del self.access_times[key]
                    return None
                return self.memory_cache[key]
            return None
    
    def set(self, key, value):
        with self.lock:
            # Try Redis first
            if redis_client:
                try:
                    cache_key = self._make_key(key)
                    redis_client.setex(
                        cache_key,
                        self.ttl_seconds,
                        json.dumps(value)
                    )
                    return
                except redis.RedisError as e:
                    logger.warning(f"Redis set failed: {str(e)}")
            
            # Fallback to memory cache
            self.memory_cache[key] = value
            self.access_times[key] = time.time()
    
    def clear(self):
        with self.lock:
            if redis_client:
                try:
                    keys = redis_client.keys(f"{self.namespace}:*")
                    if keys and isinstance(keys, list):
                        redis_client.delete(*keys)
                except redis.RedisError as e:
                    logger.warning(f"Redis clear failed: {str(e)}")
            self.memory_cache.clear()
            self.access_times.clear()

# Rate limiting
class RateLimiter:
    def __init__(self, max_requests=100, window_seconds=3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier):
        with self.lock:
            now = time.time()
            # Clean old requests
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if now - req_time < self.window_seconds
            ]
            
            if len(self.requests[identifier]) < self.max_requests:
                self.requests[identifier].append(now)
                return True
            return False

# Initialize systems
cache = AdvancedCache(namespace="mcq_cache", ttl_seconds=86400)  # 24h cache
rate_limiter = RateLimiter(max_requests=100, window_seconds=3600)

# File processing executor
file_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def extract_text_from_pdf(pdf_file) -> Tuple[bool, str, str]:
    """Extract text from PDF using pdfplumber with PyPDF2 fallback"""
    try:
        pdf_file.seek(0)
        content = pdf_file.read()
        
        # First try with pdfplumber for better accuracy
        try:
            with pdfplumber.open(BytesIO(content)) as pdf:
                text_content = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
                
                if text_content.strip():
                    return True, text_content.strip(), "pdfplumber"
        except Exception as plumber_error:
            logger.warning(f"pdfplumber extraction failed: {str(plumber_error)}")
        
        # Fallback to PyPDF2
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(content))
            if pdf_reader.is_encrypted:
                return False, "PDF is encrypted and cannot be processed", "PyPDF2"
            
            text_content = ""
            for page in pdf_reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
                except Exception as page_error:
                    logger.warning(f"Error extracting page: {str(page_error)}")
            
            if text_content.strip():
                return True, text_content.strip(), "PyPDF2"
        except Exception as pypdf_error:
            logger.error(f"PyPDF2 extraction failed: {str(pypdf_error)}")
        
        return False, "All PDF extraction methods failed", ""
        
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        return False, f"PDF extraction failed: {str(e)}", ""

def advanced_text_processing(text: str) -> str:
    """Clean and optimize text for processing"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E]', ' ', text)
    
    # Truncate to 50,000 characters to prevent excessive processing
    if len(text) > 50000:
        text = text[:25000] + " [...] " + text[-25000:]
    
    return text

def generate_topic_from_text(text_content: str) -> str:
    """Generate a topic summary using LangChain"""
    try:
        # Create a simple prompt to extract main topic
        topic_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at analyzing text and identifying main topics. Extract the primary subject/topic from the given text in 2-5 words. Be concise and specific."),
            ("human", "Analyze this text and provide the main topic in 2-5 words:\n\n{text}")
        ])
        
        # Summarize if text is too long
        if len(text_content) > 4000:
            sentences = sent_tokenize(text_content)
            # Take first 5 and last 5 sentences
            context = ' '.join(sentences[:5] + sentences[-5:])
        else:
            context = text_content
        
        topic_chain = topic_prompt | chat_model | StrOutputParser()
        topic = topic_chain.invoke({"text": context})
        
        # Clean and validate
        topic = re.sub(r'[^a-zA-Z0-9\s]', '', topic).strip()
        if not topic or len(topic) < 2:
            raise ValueError("Topic generation failed")
            
        return topic[:100]
        
    except Exception as e:
        logger.error(f"Topic generation error: {str(e)}")
        
        # Simple fallback
        sentences = text_content.split('.')[:3]
        first_text = '. '.join(sentences)
        words = [word for word in word_tokenize(first_text) 
                 if word.lower() not in stop_words and len(word) > 3]
        return ' '.join(words[:3]).title()[:100] if words else "Document Content"

def build_prompt_template(question_type: str, source_type: str = "text") -> ChatPromptTemplate:
    """Build prompt template with support for multiple question types"""
    base_system = (
        "You are an expert educational content creator specializing in generating high-quality "
        "assessment questions based on provided content. Always respond with valid JSON only."
    )
    
    question_types = {
        "mcq": "multiple-choice questions",
        "tf": "true/false questions",
        "fib": "fill-in-the-blank questions",
        "sa": "short answer questions",
        "academic": "academic multiple-choice questions",
        "practical": "practical multiple-choice questions",
        "conceptual": "conceptual multiple-choice questions"
    }
    
    qtype_plural = question_types.get(question_type, "multiple-choice questions")
    
    source_context = (
        "TEXT CONTENT:\n{text_content}\n\n" if source_type == "pdf" else ""
    )
    
    source_description = (
        "based on the provided document content" if source_type == "pdf" 
        else f"about \"{{topic}}\""
    )
    
    prompt_text = (
        f"Generate {{num_questions}} academically rigorous {qtype_plural} "
        f"{source_description} at {{difficulty}} level.\n\n"
        f"{source_context}"
        "Requirements:\n"
        "- Questions must be based strictly on the provided content\n"
        "- Test deep understanding, not just memorization\n"
        "- Include application, analysis, and synthesis level questions\n"
        "- For multiple-choice: options should be plausible and challenging\n"
        "- Include explanations for answers\n"
        "- Reference specific parts of the text when relevant\n\n"
        "Difficulty Guidelines:\n"
        "- Easy: Basic concepts and definitions\n"
        "- Medium: Application and analysis\n"
        "- Hard: Synthesis, evaluation, and complex reasoning\n"
        "- Expert: Advanced analysis and critical thinking\n\n"
        "Return ONLY this JSON structure:\n"
        "{{\n"
        "  \"metadata\": {{\n"
        "    \"topic\": \"{{topic}}\",\n"
        "    \"difficulty\": \"{{difficulty}}\",\n"
        "    \"total_questions\": {{num_questions}},\n"
        "    \"question_type\": \"{{question_type}}\",\n"
        "    \"generation_time\": \"{{timestamp}}\",\n"
        "    \"source\": \"{'PDF Document' if source_type == 'pdf' else 'Topic'}\",\n"
        "    \"bloom_taxonomy_levels\": [\"remember\", \"understand\", \"apply\", \"analyze\", \"evaluate\", \"create\"]\n"
        "  }},\n"
        "  \"questions\": [\n"
        "    {{\n"
        "      \"id\": 1,\n"
        "      \"question\": \"Clear, specific question text\",\n"
    )
    
    # Add type-specific fields
    if question_type in ["mcq", "academic", "practical", "conceptual"]:
        prompt_text += (
            "      \"options\": {{\n"
            "        \"A\": \"First option\",\n"
            "        \"B\": \"Second option\", \n"
            "        \"C\": \"Third option\",\n"
            "        \"D\": \"Fourth option\"\n"
            "      }},\n"
            "      \"correct_answer\": \"A\",\n"
        )
    elif question_type == "tf":
        prompt_text += (
            "      \"correct_answer\": true,\n"
        )
    elif question_type == "fib":
        prompt_text += (
            "      \"blanks\": [\"key term 1\", \"key term 2\"],\n"
            "      \"correct_answers\": {{\n"
            "        \"key term 1\": \"correct answer 1\",\n"
            "        \"key term 2\": \"correct answer 2\"\n"
            "      }},\n"
        )
    else:  # short answer
        prompt_text += (
            "      \"correct_answer\": \"Comprehensive correct answer\",\n"
        )
    
    # Common fields
    prompt_text += (
        "      \"explanation\": \"Detailed explanation referencing source\",\n"
        "      \"bloom_level\": \"analyze\",\n"
        "      \"estimated_time_seconds\": 45,\n"
        "      \"tags\": [\"concept1\", \"concept2\"],\n"
        "      \"text_reference\": \"Brief reference to relevant source text\"\n"
        "    }}\n"
        "  ]\n"
        "}}"
    )
    
    return ChatPromptTemplate.from_messages([
        ("system", base_system),
        ("human", prompt_text)
    ])

def validate_input(data: Dict, source_type: str = "text") -> Tuple[bool, str]:
    """Enhanced input validation"""
    if not data:
        return False, "No data provided"
    
    # Topic validation only required for text-based generation
    if source_type == "text" or "topic" in data:
        topic = data.get("topic", "").strip()
        if not topic or len(topic) < 2:
            return False, "Topic must be at least 2 characters long"
        if len(topic) > 200:
            return False, "Topic must be less than 200 characters"
    
    difficulty = data.get("difficulty", "medium").lower()
    valid_difficulties = ["easy", "medium", "hard", "expert"]
    if difficulty not in valid_difficulties:
        return False, f"Difficulty must be one of: {', '.join(valid_difficulties)}"
    
    num_questions = data.get("num_questions", 10)
    if not isinstance(num_questions, int) or num_questions < 1 or num_questions > 50:
        return False, "Number of questions must be between 1 and 50"
    
    question_type = data.get("question_type", "mcq").lower()
    valid_types = ["mcq", "tf", "fib", "sa", "academic", "practical", "conceptual"]
    if question_type not in valid_types:
        return False, f"Question type must be one of: {', '.join(valid_types)}"
    
    return True, ""

def allowed_file(filename):
    """Check allowed file extensions"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}

def generate_cache_key(params: Dict) -> str:
    """Generate a secure cache key from parameters"""
    content = json.dumps(params, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()

def validate_generated_content(content: str, question_type: str) -> Tuple[bool, Optional[Dict]]:
    """Validate and parse generated content with question type support"""
    try:
        # Find JSON in response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            return False, None
        
        json_data = json.loads(json_match.group())
        
        # Validate structure
        required_fields = ["metadata", "questions"]
        if not all(field in json_data for field in required_fields):
            return False, None
        
        questions = json_data.get("questions", [])
        if not questions:
            return False, None
        
        # Validate each question
        for i, q in enumerate(questions):
            # Common required fields
            common_required = ["question", "explanation", "bloom_level"]
            if not all(field in q for field in common_required):
                logger.warning(f"Question {i+1} missing common fields")
                return False, None
            
            # Type-specific validation
            if question_type in ["mcq", "academic", "practical", "conceptual"]:
                if "options" not in q or "correct_answer" not in q:
                    return False, None
                options = q.get("options", {})
                if len(options) < 2 or not all(isinstance(key, str) for key in options.keys()):
                    return False, None
                if q.get("correct_answer") not in options:
                    return False, None
                    
            elif question_type == "tf":
                if "correct_answer" not in q or not isinstance(q["correct_answer"], bool):
                    return False, None
                    
            elif question_type == "fib":
                if "blanks" not in q or "correct_answers" not in q:
                    return False, None
                if not isinstance(q["blanks"], list) or not isinstance(q["correct_answers"], dict):
                    return False, None
                    
            elif question_type == "sa":
                if "correct_answer" not in q or not isinstance(q["correct_answer"], str):
                    return False, None
        
        return True, json_data
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return False, None
    except Exception as e:
        logger.error(f"Content validation error: {e}")
        return False, None

def enhance_response(json_data: Dict, text_content: str = "") -> Dict:
    """Add enhancements to the response"""
    questions = json_data.get("questions", [])
    
    # Calculate metrics
    total_time = sum(q.get("estimated_time_seconds", 60) for q in questions)
    bloom_levels = [q.get("bloom_level", "remember") for q in questions]
    bloom_distribution = {level: bloom_levels.count(level) for level in set(bloom_levels)}
    
    # Add analytics
    json_data["analytics"] = {
        "total_estimated_time_minutes": round(total_time / 60, 1),
        "average_time_per_question": round(total_time / len(questions), 1) if questions else 0,
        "bloom_taxonomy_distribution": bloom_distribution,
        "difficulty_score": calculate_difficulty_score(json_data.get("metadata", {}).get("difficulty", "medium")),
        "quality_indicators": {
            "has_explanations": all("explanation" in q for q in questions),
            "has_bloom_levels": all("bloom_level" in q for q in questions),
            "has_tags": all("tags" in q for q in questions),
            "has_references": all("text_reference" in q for q in questions)
        }
    }
    
    return json_data

def calculate_difficulty_score(difficulty: str) -> float:
    """Calculate numerical difficulty score"""
    scores = {"easy": 0.25, "medium": 0.5, "hard": 0.75, "expert": 1.0}
    return scores.get(difficulty.lower(), 0.5)

@app.route('/')
def home():
    """API documentation page"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Advanced Question Generator API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 20px 0; border-radius: 5px; border-left: 4px solid #007bff; }
            .method { background: #28a745; color: white; padding: 4px 8px; border-radius: 3px; font-size: 12px; }
            .method.post { background: #007bff; }
            code { background: #e9ecef; padding: 2px 4px; border-radius: 3px; }
            .example { background: #f1f3f4; padding: 15px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Advanced Question Generator API</h1>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /generate_mcqs</h3>
                <p><strong>Generate questions from a topic</strong></p>
                
                <h4>Request Body (JSON):</h4>
                <div class="example">
                    <code>
                    {<br>
                    &nbsp;&nbsp;"topic": "Machine Learning",<br>
                    &nbsp;&nbsp;"difficulty": "medium",<br>
                    &nbsp;&nbsp;"num_questions": 5,<br>
                    &nbsp;&nbsp;"question_type": "academic"<br>
                    }
                    </code>
                </div>
                <p>Supported question types: mcq, tf, fib, sa, academic, practical, conceptual</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /generate_from_file</h3>
                <p><strong>Generate questions from uploaded PDF file</strong></p>
                
                <h4>Form Data:</h4>
                <div class="example">
                    <code>
                    file: [PDF file]<br>
                    num_questions: 10 (optional, default: 10)<br>
                    difficulty: "medium" (optional, default: medium)<br>
                    question_type: "mcq" (optional, default: mcq)<br>
                    Supported types: mcq, tf, fib, sa
                    </code>
                </div>
            </div>
            
            <h3>🚀 Features:</h3>
            <ul>
                <li>✅ PDF text extraction with advanced processing</li>
                <li>✅ Multiple question types (MCQ, True/False, Fill-in-Blank, Short Answer)</li>
                <li>✅ Bloom's taxonomy classification</li>
                <li>✅ Detailed explanations</li>
                <li>✅ Analytics and metrics</li>
                <li>✅ Rate limiting protection</li>
                <li>✅ Caching for improved performance</li>
            </ul>
            
            <h3>📋 Supported File Types:</h3>
            <ul>
                <li>PDF files (.pdf) - up to 32MB</li>
            </ul>
        </div>
    </body>
    </html>
    """)

@app.route('/generate_from_file', methods=['POST'])
def generate_from_file():
    """Endpoint for file-based question generation"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({"error": "Only PDF files are supported"}), 400
        
        # Get parameters
        try:
            num_questions = int(request.form.get('num_questions', 10))
        except ValueError:
            return jsonify({"error": "Invalid num_questions parameter"}), 400
            
        difficulty = request.form.get('difficulty', 'medium').lower()
        question_type = request.form.get('question_type', 'mcq').lower()
        
        # Validate parameters
        if num_questions < 1 or num_questions > 50:
            return jsonify({"error": "Number of questions must be between 1 and 50"}), 400
        
        if difficulty not in ["easy", "medium", "hard", "expert"]:
            return jsonify({"error": "Invalid difficulty level"}), 400
        
        valid_types = ["mcq", "tf", "fib", "sa"]
        if question_type not in valid_types:
            return jsonify({"error": f"Question type must be one of: {', '.join(valid_types)}"}), 400
        
        # Check rate limiting
        client_ip = request.remote_addr
        if not rate_limiter.is_allowed(client_ip):
            return jsonify({"error": "Rate limit exceeded"}), 429
        
        logger.info(f"Processing file upload: {file.filename}")
        
        # Extract text from file
        success, text_content, method = extract_text_from_pdf(file)
        if not success:
            return jsonify({
                "error": "Text extraction failed", 
                "message": text_content,
                "suggestions": [
                    "Ensure the PDF contains selectable text",
                    "Try with a different PDF file",
                    "Check if the PDF is encrypted"
                ]
            }), 400
        
        # Process text
        text_content = advanced_text_processing(text_content)
        logger.info(f"Extracted {len(text_content)} characters using {method}")
        
        # Generate topic
        topic = generate_topic_from_text(text_content)
        logger.info(f"Generated topic: {topic}")
        
        # Check cache
        cache_params = {
            "content_hash": hashlib.sha256(text_content.encode()).hexdigest(),
            "difficulty": difficulty,
            "num_questions": num_questions,
            "question_type": question_type
        }
        cache_key = generate_cache_key(cache_params)
        cached_result = cache.get(cache_key)
        
        if cached_result:
            logger.info("Cache hit for file content")
            return jsonify({**cached_result, "cached": True})
        
        # Build prompt template
        prompt_template = build_prompt_template(
            question_type=question_type,
            source_type="pdf"
        )
        timestamp = datetime.now().isoformat()
        
        # Create LangChain chain
        chain = (
            RunnablePassthrough.assign(timestamp=lambda _: timestamp)
            | prompt_template
            | chat_model
            | StrOutputParser()
        )
        
        # Generate with retry logic
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating questions (attempt {attempt + 1}) - Type: {question_type}")
                
                # Invoke LangChain
                response = chain.invoke({
                    "topic": topic,
                    "difficulty": difficulty,
                    "num_questions": num_questions,
                    "text_content": text_content[:12000],  # Limit context
                    "question_type": question_type,
                    "timestamp": timestamp
                })
                
                # Validate generated content
                is_valid_content, json_data = validate_generated_content(response, question_type)
                if is_valid_content and json_data:
                    # Enhance response
                    enhanced_data = enhance_response(json_data, text_content)
                    
                    # Add file metadata
                    enhanced_data["metadata"].update({
                        "source_file": secure_filename(file.filename),
                        "content_length": len(text_content),
                        "extraction_method": method,
                        "auto_generated_topic": topic
                    })
                    
                    # Cache the result
                    cache.set(cache_key, enhanced_data)
                    
                    logger.info(f"Successfully generated {len(enhanced_data.get('questions', []))} questions")
                    return jsonify({**enhanced_data, "cached": False})
                
                logger.warning(f"Invalid content on attempt {attempt + 1}")
                last_error = "Content validation failed"
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"Generation attempt {attempt + 1} failed: {last_error}")
                if attempt == max_retries - 1:
                    break
                time.sleep(1.5)  # Brief pause before retry
        
        return jsonify({
            "error": "Generation failed",
            "message": f"Unable to generate valid questions after {max_retries} attempts",
            "last_error": last_error,
            "debug_info": {
                "topic": topic,
                "text_length": len(text_content),
                "text_sample": text_content[:500] + "..." if len(text_content) > 500 else text_content
            }
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "details": str(e)
        }), 500

@app.route('/generate_mcqs', methods=['POST'])
def generate_mcqs():
    """Endpoint for topic-based question generation"""
    try:
        data = request.get_json()
        
        # Validate input
        is_valid, error_msg = validate_input(data, source_type="text")
        if not is_valid:
            return jsonify({"error": "Validation failed", "message": error_msg}), 400
        
        # Extract parameters
        topic = data["topic"].strip()
        difficulty = data.get("difficulty", "medium").lower()
        num_questions = data.get("num_questions", 10)
        question_type = data.get("question_type", "mcq").lower()
        
        # Check rate limiting
        client_ip = request.remote_addr
        if not rate_limiter.is_allowed(client_ip):
            return jsonify({"error": "Rate limit exceeded"}), 429
        
        # Check cache
        cache_params = {
            "topic": topic,
            "difficulty": difficulty,
            "num_questions": num_questions,
            "question_type": question_type
        }
        cache_key = generate_cache_key(cache_params)
        cached_result = cache.get(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for topic: {topic}")
            return jsonify({**cached_result, "cached": True})
        
        # Build prompt template
        prompt_template = build_prompt_template(
            question_type=question_type,
            source_type="text"
        )
        timestamp = datetime.now().isoformat()
        
        # Create LangChain chain
        chain = (
            RunnablePassthrough.assign(timestamp=lambda _: timestamp)
            | prompt_template
            | chat_model
            | StrOutputParser()
        )
        
        # Generate with retry logic
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating questions (attempt {attempt + 1}) - Topic: {topic}")
                
                # Invoke LangChain
                response = chain.invoke({
                    "topic": topic,
                    "difficulty": difficulty,
                    "num_questions": num_questions,
                    "question_type": question_type,
                    "timestamp": timestamp
                })
                
                # Validate generated content
                is_valid_content, json_data = validate_generated_content(response, question_type)
                if is_valid_content and json_data:
                    # Enhance response
                    enhanced_data = enhance_response(json_data)
                    
                    # Cache the result
                    cache.set(cache_key, enhanced_data)
                    
                    logger.info(f"Successfully generated {len(enhanced_data.get('questions', []))} questions")
                    return jsonify({**enhanced_data, "cached": False})
                
                logger.warning(f"Invalid content on attempt {attempt + 1}")
                last_error = "Content validation failed"
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"Generation attempt {attempt + 1} failed: {last_error}")
                if attempt == max_retries - 1:
                    break
                time.sleep(1.5)  # Brief pause before retry
        
        return jsonify({
            "error": "Generation failed",
            "message": f"Unable to generate valid questions after {max_retries} attempts",
            "last_error": last_error,
            "debug_info": {
                "topic": topic
            }
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "cache": "redis" if redis_client else "memory",
        "model": "llama3-70b-8192",
        "endpoints": {
            "generate_mcqs": "active",
            "generate_from_file": "active"
        }
    })

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear all cached data"""
    try:
        cache.clear()
        return jsonify({"status": "success", "message": "Cache cleared"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "File too large", "message": "Maximum file size is 32MB"}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    logger.info("Starting Advanced Question Generator API...")
    logger.info(f"Using model: llama3-70b-8192")
    logger.info(f"Redis cache: {'Enabled' if redis_client else 'Disabled'}")
    logger.info(f"Rate limiting: {rate_limiter.max_requests} requests per hour")
    logger.info("Supported endpoints: /generate_mcqs, /generate_from_file")
    
    app.run(
        debug=False,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )