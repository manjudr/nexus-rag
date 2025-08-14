"""
Hybrid Educational Content Enhancer
Combines rule-based extraction with optional LangExtract integration
Works without API keys using local models and pattern matching
"""

import re
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import textwrap

try:
    import langextract as lx
    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False
    print("⚠️ LangExtract not available, using fallback methods")

@dataclass
class EnhancedEducationalMetadata:
    """Enhanced educational metadata structure"""
    learning_objectives: List[Dict]
    key_concepts: List[Dict]
    difficulty_level: str
    prerequisites: List[Dict]
    study_questions: List[Dict]
    examples: List[Dict]
    content_sections: List[Dict]
    extraction_method: str

class HybridEducationalEnhancer:
    """
    Hybrid system that enhances educational content using:
    1. Pattern-based extraction (no API keys required)
    2. LangExtract integration (when API keys available)
    3. Local LLM enhancement (using existing models)
    """
    
    def __init__(self, use_langextract: bool = True, api_key: Optional[str] = None):
        """
        Initialize the hybrid enhancer
        
        Args:
            use_langextract: Whether to attempt LangExtract usage
            api_key: Optional API key for cloud models
        """
        self.use_langextract = use_langextract and LANGEXTRACT_AVAILABLE
        
        # Check for various API key sources for LangExtract
        self.api_key = (
            api_key or 
            os.environ.get('LANGEXTRACT_API_KEY') or
            os.environ.get('GOOGLE_API_KEY') or
            os.environ.get('GEMINI_API_KEY') or
            os.environ.get('OPENAI_API_KEY')  # Use Azure OpenAI key for LangExtract
        )
        
        # Pattern-based extraction patterns
        self.patterns = {
            'learning_objectives': [
                r'learning objective[s]?[:\-]\s*(.+?)(?:\n|$)',
                r'students will\s+(.+?)(?:\n|\.)',
                r'objective[s]?[:\-]\s*(.+?)(?:\n|$)',
                r'goals?[:\-]\s*(.+?)(?:\n|$)'
            ],
            'key_concepts': [
                r'key concepts?[:\-]\s*(.+?)(?:\n|$)',
                r'important terms?[:\-]\s*(.+?)(?:\n|$)',
                r'concepts?[:\-]\s*(.+?)(?:\n|$)',
                r'vocabulary[:\-]\s*(.+?)(?:\n|$)'
            ],
            'prerequisites': [
                r'prerequisite[s]?[:\-]\s*(.+?)(?:\n|$)',
                r'students should\s+(?:know|understand|have)\s*(.+?)(?:\n|\.)',
                r'before this lesson[:\-]\s*(.+?)(?:\n|$)',
                r'background[:\-]\s*(.+?)(?:\n|$)'
            ],
            'study_questions': [
                r'(?:study )?questions?[:\-]\s*(.+?)(?:\n|$)',
                r'(?:\d+\.|\•|\-)\s*(.+\?)(?:\n|$)',
                r'review[:\-]\s*(.+?)(?:\n|$)',
                r'exercises?[:\-]\s*(.+?)(?:\n|$)'
            ],
            'examples': [
                r'examples?[:\-]\s*(.+?)(?:\n|$)',
                r'for instance[:\-]\s*(.+?)(?:\n|$)',
                r'such as[:\-]\s*(.+?)(?:\n|$)',
                r'demonstration[:\-]\s*(.+?)(?:\n|$)'
            ]
        }
    
    def enhance_educational_content(self, content: str, filename: str = "unknown") -> EnhancedEducationalMetadata:
        """
        Enhance educational content using available methods
        
        Args:
            content: Text content to enhance
            filename: Source filename for context
            
        Returns:
            Enhanced educational metadata
        """
        print(f"🔍 Enhancing content from {filename}...")
        
        # Try LangExtract first if available and configured
        if self.use_langextract and self.api_key:
            try:
                return self._enhance_with_langextract(content, filename)
            except Exception as e:
                print(f"⚠️ LangExtract failed: {str(e)}, falling back to pattern-based extraction")
        
        # Fall back to pattern-based extraction
        return self._enhance_with_patterns(content, filename)
    
    def _enhance_with_langextract(self, content: str, filename: str) -> EnhancedEducationalMetadata:
        """Enhance using LangExtract (requires API key) or Azure OpenAI direct"""
        
        print("🤖 Attempting enhanced extraction with Azure OpenAI...")
        
        # Try direct Azure OpenAI call if LangExtract fails
        try:
            from models.openai_llm import OpenAIGenerativeModel
            
            llm = OpenAIGenerativeModel('gpt-35-turbo')
            
            prompt = f"""
            You are an educational content analyzer. Extract educational metadata from the following content and return a properly formatted JSON object.

            Content:
            {content[:2000]}  # Limit content length for better processing
            
            Extract the following and return ONLY a valid JSON object with no additional text:
            {{
                "learning_objectives": ["objective1", "objective2"],
                "key_concepts": ["concept1", "concept2"],
                "prerequisites": ["prereq1", "prereq2"],
                "study_questions": ["question1", "question2"],
                "examples": ["example1", "example2"],
                "difficulty_level": "beginner"
            }}
            """
            
            response = llm.generate(prompt)
            
            # Parse the JSON response
            import json
            try:
                extracted_data = json.loads(response)
                
                # Convert to our format
                metadata = EnhancedEducationalMetadata(
                    learning_objectives=[{"text": obj, "type": "ai_extracted"} for obj in extracted_data.get("learning_objectives", [])],
                    key_concepts=[{"text": concept, "importance": "medium"} for concept in extracted_data.get("key_concepts", [])],
                    difficulty_level=extracted_data.get("difficulty_level", "intermediate"),
                    prerequisites=[{"text": prereq, "type": "ai_extracted"} for prereq in extracted_data.get("prerequisites", [])],
                    study_questions=[{"text": q, "type": "ai_extracted"} for q in extracted_data.get("study_questions", [])],
                    examples=[{"text": ex, "type": "ai_extracted"} for ex in extracted_data.get("examples", [])],
                    content_sections=[],
                    extraction_method="azure_openai_direct"
                )
                
                print(f"✅ Azure OpenAI extracted: {len(metadata.learning_objectives)} objectives, {len(metadata.key_concepts)} concepts")
                return metadata
                
            except json.JSONDecodeError:
                print("⚠️ Failed to parse Azure OpenAI JSON response, trying pattern extraction")
                
        except Exception as e:
            print(f"⚠️ Azure OpenAI direct extraction failed: {str(e)}, trying LangExtract...")
        
        # Original LangExtract attempt (keeping as fallback)
        educational_prompt = textwrap.dedent("""\
            Extract educational metadata from this content including:
            - Learning objectives and what students will learn
            - Key concepts, terms, and vocabulary
            - Prerequisites and background knowledge needed
            - Study questions and review items
            - Examples and demonstrations
            Use exact text from the content for extractions.
        """)
        
        examples = [
            lx.data.ExampleData(
                text="Learning Objective: Understand photosynthesis. Key concepts include chlorophyll and carbon dioxide.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="learning_objective",
                        extraction_text="Understand photosynthesis",
                        attributes={"type": "comprehension"}
                    ),
                    lx.data.Extraction(
                        extraction_class="key_concept",
                        extraction_text="chlorophyll",
                        attributes={"importance": "high"}
                    )
                ]
            )
        ]
        
        # Create Azure OpenAI client manually for LangExtract
        import openai
        azure_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
        deployment_name = os.environ.get('AZURE_CHAT_DEPLOYMENT', 'gpt-35-turbo')
        
        if azure_endpoint and deployment_name:
            # Try Azure OpenAI with LangExtract
            base_url = f"{azure_endpoint.rstrip('/')}/openai/deployments/{deployment_name}/"
            
            result = lx.extract(
                text_or_documents=content,
                prompt_description=educational_prompt,
                examples=examples,
                language_model_type=lx.inference.OpenAILanguageModel,  # Pass the class, not instance
                model_id=deployment_name,
                api_key=self.api_key,
                fence_output=True,  # Required for OpenAI models
                use_schema_constraints=False,  # Required for OpenAI models
                language_model_params={
                    "base_url": base_url,
                    "api-version": "2024-02-15-preview"
                }
            )
        else:
            # Fallback to regular OpenAI
            result = lx.extract(
                text_or_documents=content,
                prompt_description=educational_prompt,
                examples=examples,
                model_id="gpt-3.5-turbo",  # Use standard OpenAI model
                api_key=self.api_key,
                language_model_type=lx.inference.OpenAILanguageModel,
                fence_output=True,
                use_schema_constraints=False,
            )
        
        return self._process_langextract_result(result, filename)
    
    def _enhance_with_patterns(self, content: str, filename: str) -> EnhancedEducationalMetadata:
        """Enhance using pattern-based extraction (no API keys required)"""
        
        print("🔍 Using pattern-based extraction...")
        
        # Extract different types of educational content
        learning_objectives = self._extract_with_patterns(content, 'learning_objectives')
        key_concepts = self._extract_with_patterns(content, 'key_concepts')
        prerequisites = self._extract_with_patterns(content, 'prerequisites')
        study_questions = self._extract_with_patterns(content, 'study_questions')
        examples = self._extract_with_patterns(content, 'examples')
        
        # Extract content sections (chapters, headings)
        content_sections = self._extract_content_sections(content)
        
        # Determine difficulty level
        difficulty_level = self._determine_difficulty_pattern_based(content)
        
        # Enhance with additional analysis
        if not key_concepts:
            key_concepts = self._extract_implied_concepts(content)
        
        if not study_questions:
            study_questions = self._generate_basic_questions(content)
        
        metadata = EnhancedEducationalMetadata(
            learning_objectives=learning_objectives,
            key_concepts=key_concepts,
            difficulty_level=difficulty_level,
            prerequisites=prerequisites,
            study_questions=study_questions,
            examples=examples,
            content_sections=content_sections,
            extraction_method="pattern-based"
        )
        
        print(f"✅ Extracted: {len(learning_objectives)} objectives, {len(key_concepts)} concepts, {len(study_questions)} questions")
        
        return metadata
    
    def _extract_with_patterns(self, content: str, category: str) -> List[Dict]:
        """Extract content using regex patterns"""
        
        results = []
        patterns = self.patterns.get(category, [])
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                text = match.group(1).strip()
                if text and len(text) > 5:  # Filter out very short matches
                    results.append({
                        "text": text,
                        "source": "pattern_extraction",
                        "confidence": 0.8
                    })
        
        # Remove duplicates
        unique_results = []
        seen_texts = set()
        for result in results:
            text_lower = result["text"].lower()
            if text_lower not in seen_texts:
                seen_texts.add(text_lower)
                unique_results.append(result)
        
        return unique_results[:5]  # Limit to top 5 results
    
    def _extract_content_sections(self, content: str) -> List[Dict]:
        """Extract content sections like chapters and headings"""
        
        sections = []
        
        # Look for chapter headings
        chapter_pattern = r'^(chapter\s+\d+[:\-]?\s*(.+))$'
        for match in re.finditer(chapter_pattern, content, re.IGNORECASE | re.MULTILINE):
            sections.append({
                "type": "chapter",
                "title": match.group(2).strip(),
                "full_heading": match.group(1).strip(),
                "position": match.start()
            })
        
        # Look for section headings (lines that end with specific patterns)
        section_patterns = [
            r'^([A-Z][^.!?]*):$',  # "SECTION TITLE:"
            r'^(\d+\.\s+[^.!?]+)$',  # "1. Section Title"
            r'^([A-Z][A-Z\s]{10,})$'  # "ALL CAPS HEADINGS"
        ]
        
        for pattern in section_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                sections.append({
                    "type": "section",
                    "title": match.group(1).strip(),
                    "position": match.start()
                })
        
        return sections[:10]  # Limit to top 10 sections
    
    def _extract_implied_concepts(self, content: str) -> List[Dict]:
        """Extract key concepts by looking for important terms"""
        
        # Common educational term patterns
        concept_indicators = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:a|an|the)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+refers?\s+to',
            r'\bdefin(?:e|ition)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+means?',
        ]
        
        concepts = []
        for pattern in concept_indicators:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                concept = match.group(1).strip()
                if len(concept.split()) <= 3:  # Keep concepts to 3 words or less
                    concepts.append({
                        "text": concept,
                        "source": "implied_extraction",
                        "confidence": 0.6
                    })
        
        # Remove duplicates and return top concepts
        unique_concepts = []
        seen = set()
        for concept in concepts:
            text_lower = concept["text"].lower()
            if text_lower not in seen:
                seen.add(text_lower)
                unique_concepts.append(concept)
        
        return unique_concepts[:8]
    
    def _generate_basic_questions(self, content: str) -> List[Dict]:
        """Generate basic study questions based on content"""
        
        questions = []
        
        # Look for question-worthy statements
        if "what is" in content.lower() or "definition" in content.lower():
            questions.append({
                "text": "What are the key definitions in this content?",
                "type": "definition",
                "source": "generated"
            })
        
        if "process" in content.lower():
            questions.append({
                "text": "Describe the main process discussed in this content.",
                "type": "process",
                "source": "generated"
            })
        
        if "example" in content.lower():
            questions.append({
                "text": "What examples are provided in this content?",
                "type": "example",
                "source": "generated"
            })
        
        return questions
    
    def _determine_difficulty_pattern_based(self, content: str) -> str:
        """Determine difficulty level based on text patterns"""
        
        content_lower = content.lower()
        
        # Advanced indicators
        advanced_terms = ["complex", "advanced", "detailed", "comprehensive", "in-depth", 
                         "analyze", "evaluate", "synthesize", "critical"]
        
        # Basic indicators
        basic_terms = ["introduction", "basic", "simple", "fundamental", "overview",
                      "begin", "start", "first", "elementary"]
        
        advanced_count = sum(1 for term in advanced_terms if term in content_lower)
        basic_count = sum(1 for term in basic_terms if term in content_lower)
        
        if advanced_count > basic_count and advanced_count >= 2:
            return "advanced"
        elif basic_count > advanced_count and basic_count >= 2:
            return "basic"
        else:
            return "intermediate"
    
    def _process_langextract_result(self, result, filename: str) -> EnhancedEducationalMetadata:
        """Process LangExtract result into our metadata format"""
        
        # This would process the LangExtract result
        # For now, return a basic structure
        return EnhancedEducationalMetadata(
            learning_objectives=[],
            key_concepts=[],
            difficulty_level="intermediate",
            prerequisites=[],
            study_questions=[],
            examples=[],
            content_sections=[],
            extraction_method="langextract"
        )

def test_hybrid_enhancer():
    """Test the hybrid enhancement system"""
    
    print("🧪 Testing Hybrid Educational Enhancer...")
    
    sample_content = textwrap.dedent("""\
        Chapter 4: Photosynthesis and Plant Nutrition
        
        Learning Objectives:
        - Students will understand the process of photosynthesis
        - Learn about factors affecting plant growth
        
        Key Concepts: Chlorophyll, carbon dioxide, oxygen, glucose
        
        Photosynthesis is the process by which plants convert sunlight into energy.
        This process occurs in the chloroplasts and requires several inputs.
        
        Prerequisites: Students should understand basic plant anatomy and cell structure.
        
        Study Questions:
        1. What are the main inputs for photosynthesis?
        2. Where does photosynthesis occur in plant cells?
        3. What are the products of photosynthesis?
        
        Example: A green leaf placed in sunlight demonstrates the photosynthesis process.
    """)
    
    enhancer = HybridEducationalEnhancer(use_langextract=False)  # Test pattern-based only
    
    try:
        metadata = enhancer.enhance_educational_content(sample_content, "test_content.txt")
        
        print("📋 Enhanced Metadata:")
        print(f"  Learning Objectives: {len(metadata.learning_objectives)}")
        for obj in metadata.learning_objectives:
            print(f"    - {obj['text']}")
        
        print(f"  Key Concepts: {len(metadata.key_concepts)}")
        for concept in metadata.key_concepts:
            print(f"    - {concept['text']}")
        
        print(f"  Study Questions: {len(metadata.study_questions)}")
        for question in metadata.study_questions:
            print(f"    - {question['text']}")
        
        print(f"  Difficulty Level: {metadata.difficulty_level}")
        print(f"  Extraction Method: {metadata.extraction_method}")
        
        return metadata
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return None

if __name__ == "__main__":
    test_hybrid_enhancer()
