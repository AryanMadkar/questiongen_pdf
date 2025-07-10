from langchain_core.prompts import ChatPromptTemplate
import re
from typing import Dict, List, Tuple, Optional

# Cached common patterns to avoid recompilation
CHAPTER_PATTERNS = [
    re.compile(r'chapter\s+\d+[:\s]+(.+)', re.IGNORECASE),
    re.compile(r'section\s+\d+[:\s]+(.+)', re.IGNORECASE),
    re.compile(r'unit\s+\d+[:\s]+(.+)', re.IGNORECASE),
    re.compile(r'lesson\s+\d+[:\s]+(.+)', re.IGNORECASE)
]

# Common stop words for topic extraction
STOP_WORDS = frozenset([
    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
    'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 
    'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 
    'a', 'an', 'it', 'they', 'we', 'you', 'he', 'she', 'him', 'her', 'us', 'them'
])

def extract_topic_from_pdf_content(text_content: str) -> str:
    """Extract topic from PDF content using optimized strategies"""
    if not text_content or len(text_content) < 20:
        return "Document Content"
    
    lines = text_content.strip().split('\n')
    
    # Strategy 1: Title detection (first 5 lines)
    for line in lines[:5]:
        line = line.strip()
        if 10 < len(line) < 80 and (line.isupper() or line.istitle()) and not line.endswith('.'):
            return line
    
    # Strategy 2: Chapter/section headers (first 15 lines)
    for line in lines[:15]:
        for pattern in CHAPTER_PATTERNS:
            match = pattern.search(line)
            if match:
                return match.group(1).strip().title()
    
    # Strategy 3: Key terms frequency (optimized)
    words = text_content.lower().split()
    word_freq = {}
    
    for word in words:
        if len(word) > 3:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word not in STOP_WORDS:
                word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
    
    if word_freq:
        top_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:2]
        return " & ".join(term[0].title() for term in top_terms)
    
    # Strategy 4: First meaningful sentence
    sentences = text_content.split('.')
    for sentence in sentences[:3]:
        sentence = sentence.strip()
        if 15 < len(sentence) < 100:
            return sentence
    
    return "Document Analysis"

def build_prompt_template_pdf(question_type: str) -> ChatPromptTemplate:
    """Build optimized prompt template for PDF content"""
    
    # Minimal JSON structure for token efficiency
    json_format = '''{
  "questions": [
    {
      "question": "Question text?",
      "options": ["A) Option1", "B) Option2", "C) Option3", "D) Option4"],
      "correct_answer": "A",
      "explanation": "Brief explanation",
      "type": "technical",
      "difficulty": "{difficulty}"
    }
  ]
}'''
    
    # Optimized base prompt
    base_prompt = """Generate {num_questions} MCQs from the text at {difficulty} level.

TEXT: {text_content}

Question Types:
- GENERAL: Factual information
- QUANTITATIVE: Math/calculations  
- VERBAL: Language/comprehension
- TECHNICAL: Subject-specific concepts
- LOGICAL: Critical thinking

Requirements:
- Use ONLY the provided text
- Mix question types appropriately
- Make options challenging but fair
- Include brief explanations

JSON format only:
""" + json_format
    
    # Question type specific additions
    type_additions = {
        "practical": "\nFocus: Real-world applications, scenarios, best practices from text.",
        "conceptual": "\nFocus: Theoretical concepts, relationships, cause-effect from text.",
        "academic": "\nFocus: Deep understanding, analysis, synthesis from text."
    }
    
    final_prompt = base_prompt + type_additions.get(question_type, type_additions["academic"])
    
    return ChatPromptTemplate.from_messages([
        ("system", "You are an expert MCQ generator. Respond with valid JSON only."),
        ("human", final_prompt)
    ])

def build_prompt_template(question_type: str) -> ChatPromptTemplate:
    """Build optimized prompt template for topic-based questions"""
    
    # Ultra-compact JSON structure
    json_format = '''{
  "questions": [
    {
      "question": "Question text?",
      "options": ["A) Option1", "B) Option2", "C) Option3", "D) Option4"],
      "correct_answer": "A",
      "explanation": "Brief reason",
      "type": "technical"
    }
  ]
}'''
    
    # Streamlined base prompt
    base_prompt = """Generate {num_questions} MCQs on "{topic}" ({difficulty} level).

Question Types - Choose appropriate mix:
• GENERAL: Broad factual knowledge
• QUANTITATIVE: Math/numerical reasoning
• VERBAL: Language/comprehension skills
• TECHNICAL: Subject-specific concepts
• LOGICAL: Critical thinking/patterns

Requirements:
- Test understanding, not memorization
- Make realistic, challenging options
- Brief but clear explanations
- Appropriate difficulty distribution

JSON only:
""" + json_format
    
    # Type-specific focus (minimal additions)
    focus_map = {
        "practical": "\nEmphasize: Real scenarios, applications, problem-solving.",
        "conceptual": "\nEmphasize: Theory, relationships, abstract thinking.",
        "academic": "\nEmphasize: Deep analysis, synthesis, critical evaluation."
    }
    
    final_prompt = base_prompt + focus_map.get(question_type, focus_map["academic"])
    
    return ChatPromptTemplate.from_messages([
        ("system", "Expert MCQ generator. Valid JSON only."),
        ("human", final_prompt)
    ])

def get_optimized_direct_prompt(topic: str, difficulty: str, num_questions: int, question_type: str = "academic") -> str:
    """Get ultra-optimized direct prompt for minimal token usage"""
    
    type_focus = {
        "practical": "real-world applications",
        "conceptual": "theoretical understanding", 
        "academic": "deep analysis"
    }
    
    return f"""Generate {num_questions} MCQs on "{topic}" ({difficulty} level, {type_focus.get(question_type, 'comprehensive')}).

JSON format:
{{
  "questions": [
    {{
      "question": "Question text?",
      "options": ["A) Option1", "B) Option2", "C) Option3", "D) Option4"],
      "correct_answer": "A",
      "explanation": "Brief reason",
      "type": "technical"
    }}
  ]
}}

Rules:
- Exactly {num_questions} unique questions
- Mix question types (general, quantitative, verbal, technical, logical)
- Challenging but fair options
- Brief explanations
- Valid JSON only"""

def get_optimized_pdf_prompt(text_content: str, difficulty: str, num_questions: int, question_type: str = "academic") -> str:
    """Get ultra-optimized PDF prompt for minimal token usage"""
    
    # Truncate text if too long (keep first 2000 chars for context)
    if len(text_content) > 2000:
        text_content = text_content[:2000] + "..."
    
    type_focus = {
        "practical": "applications from text",
        "conceptual": "concepts from text",
        "academic": "analysis of text"
    }
    
    return f"""Generate {num_questions} MCQs from this text ({difficulty} level, {type_focus.get(question_type, 'comprehensive')}).

TEXT: {text_content}

JSON format:
{{
  "questions": [
    {{
      "question": "Question based on text?",
      "options": ["A) Option1", "B) Option2", "C) Option3", "D) Option4"],
      "correct_answer": "A",
      "explanation": "Brief reason from text",
      "type": "technical"
    }}
  ]
}}

Rules:
- Use ONLY provided text
- Exactly {num_questions} questions
- Mix types: general, quantitative, verbal, technical, logical
- Valid JSON only"""

# Utility functions for enhanced performance
def validate_question_structure(question: Dict) -> bool:
    """Quick validation of question structure"""
    required_keys = {"question", "options", "correct_answer"}
    return (
        isinstance(question, dict) and
        required_keys.issubset(question.keys()) and
        isinstance(question["options"], list) and
        len(question["options"]) >= 4 and
        question["correct_answer"] in ["A", "B", "C", "D"]
    )

def extract_question_type(question_text: str) -> str:
    """Auto-detect question type from text"""
    text_lower = question_text.lower()
    
    # Quick pattern matching
    if any(word in text_lower for word in ['calculate', 'compute', 'solve', 'number', 'percent']):
        return "quantitative"
    elif any(word in text_lower for word in ['define', 'meaning', 'synonym', 'grammar']):
        return "verbal"
    elif any(word in text_lower for word in ['analyze', 'compare', 'evaluate', 'conclude']):
        return "logical"
    elif any(word in text_lower for word in ['apply', 'implement', 'use', 'practice']):
        return "practical"
    else:
        return "technical"

def optimize_question_metadata(questions: List[Dict], topic: str, difficulty: str) -> List[Dict]:
    """Add optimized metadata to questions"""
    for i, question in enumerate(questions):
        question.update({
            "id": i + 1,
            "topic": topic,
            "difficulty": difficulty,
            "type": extract_question_type(question.get("question", ""))
        })
    return questions

# Cache for compiled patterns to avoid recompilation
_compiled_patterns_cache = {}

def get_compiled_pattern(pattern: str) -> re.Pattern:
    """Get cached compiled regex pattern"""
    if pattern not in _compiled_patterns_cache:
        _compiled_patterns_cache[pattern] = re.compile(pattern, re.IGNORECASE)
    return _compiled_patterns_cache[pattern]