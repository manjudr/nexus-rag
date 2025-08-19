from tools.base import BaseTool
from vector_db.base import BaseVectorDB
from vector_db.content_discovery_db import ContentDiscoveryVectorDB
from models.embedding import EmbeddingModel
from models.llm import GenerativeModel
from rank_bm25 import BM25Okapi
from typing import List, Dict
import json
import time
import re

class ContentDiscoveryTool(BaseTool):
    """
    Updated Content Discovery Tool that works with the new database architecture
    Returns actual filenames from the database instead of generating fictional ones
    """
    
    def __init__(self, db: BaseVectorDB, embedding_model: EmbeddingModel, llm: GenerativeModel, 
                 name: str, description: str, top_k: int = 5, return_json: bool = False):
        self.db = db
        self.embedding_model = embedding_model
        self.llm = llm
        self.name = name
        self.description = description
        self.top_k = top_k
        self.return_json = return_json
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load content metadata from metadata.json"""
        try:
            import os
            metadata_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                       'data', 'educational_content', 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"⚠️ Metadata file not found at {metadata_path}, using fallback generation")
                return {}
        except Exception as e:
            print(f"⚠️ Failed to load metadata: {e}, using fallback generation")
            return {}

    def _get_metadata_for_file(self, filename: str) -> Dict:
        """Get metadata for a specific file"""
        # Try exact match first
        if filename in self.metadata:
            return self.metadata[filename]
        
        # Try case-insensitive match
        for key, value in self.metadata.items():
            if key.lower() == filename.lower():
                return value
            
        # Try partial match (without extension)
        base_filename = filename.replace('.pdf', '')
        for key, value in self.metadata.items():
            if key.replace('.pdf', '').lower() == base_filename.lower():
                return value
        
        # Try contains match (for truncated filenames)
        for key, value in self.metadata.items():
            if filename.lower() in key.lower() or key.lower() in filename.lower():
                return value
                
        return {}

    def _initialize_bm25(self):
        # Try to get all documents from any vector DB type
        try:
            if hasattr(self.db, 'get_all_documents'):
                documents_data = self.db.get_all_documents()
                if not documents_data:
                    return None, None
                
                # Handle different document formats
                if self._is_enhanced_database():
                    # Enhanced database with structured content + educational metadata
                    documents = [doc["content"] for doc in documents_data]
                    tokenized_corpus = [doc.split(" ") for doc in documents]
                    return BM25Okapi(tokenized_corpus), documents_data
                else:
                    # Basic content discovery database with structured content but no educational metadata
                    documents = [doc["content"] for doc in documents_data]
                    tokenized_corpus = [doc.split(" ") for doc in documents]
                    return BM25Okapi(tokenized_corpus), documents_data
        except Exception as e:
            # Silently return None if BM25 initialization fails
            pass
        
        return None, None

    def _extract_keywords_from_content(self, content: str, count: int = 8) -> List[str]:
        """Extract meaningful keywords from content."""
        import re
        from collections import Counter
        
        # Remove special characters and split into words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'this', 'that', 'these', 'those', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'about', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out',
            'off', 'over', 'under', 'again', 'further', 'then', 'once'
        }
        
        meaningful_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count word frequency and return top keywords
        word_counts = Counter(meaningful_words)
        keywords = [word for word, count in word_counts.most_common(count)]
        
        return keywords

    def _create_intelligent_summary(self, content: str, max_length: int = 400) -> str:
        """Create an intelligent summary that preserves sentence structure and meaning."""
        import re
        
        if not content or len(content.strip()) == 0:
            return "No content available"
        
        # Clean and normalize the content
        clean_content = re.sub(r'\s+', ' ', content.strip())
        
        # If content is already short enough, return as is
        if len(clean_content) <= max_length:
            return clean_content
        
        # Split into sentences while preserving sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', clean_content)
        
        if not sentences:
            # Fallback: if no sentences found, do intelligent word truncation
            words = clean_content.split()
            if len(' '.join(words[:max_length//6])) <= max_length:
                return ' '.join(words[:max_length//6]) + "..."
            else:
                return clean_content[:max_length-3] + "..."
        
        # Build summary by adding complete sentences
        summary = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence would exceed the limit
            potential_summary = summary + (" " if summary else "") + sentence
            
            if len(potential_summary) <= max_length:
                summary = potential_summary
            else:
                # If we have at least one sentence, stop here
                if summary:
                    break
                # If even the first sentence is too long, truncate it intelligently
                else:
                    # Find the last complete phrase or clause before the limit
                    truncated = sentence[:max_length-3]
                    # Try to break at a natural point (comma, semicolon, etc.)
                    natural_breaks = [',', ';', ':', ' and ', ' or ', ' but ', ' with ', ' in ', ' on ', ' at ']
                    best_break = 0
                    for break_point in natural_breaks:
                        last_occurrence = truncated.rfind(break_point)
                        if last_occurrence > best_break:
                            best_break = last_occurrence + len(break_point)
                    
                    if best_break > len(truncated) * 0.7:  # Only use if it's not too early
                        summary = truncated[:best_break].strip() + "..."
                    else:
                        summary = truncated + "..."
                    break
        
        # Add ellipsis if we had to truncate
        if len(clean_content) > len(summary) and not summary.endswith("..."):
            summary = summary + "..."
        
        return summary.strip()

    def _create_query_aware_summary(self, content: str, query: str, max_length: int = 400) -> str:
        """Create a summary that prioritizes content relevant to the user's query."""
        import re
        
        if not content or len(content.strip()) == 0:
            return "No content available"
        
        # Clean and normalize the content
        clean_content = re.sub(r'\s+', ' ', content.strip())
        
        # If content is already short enough, return as is
        if len(clean_content) <= max_length:
            return clean_content
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', clean_content)
        
        if not sentences:
            return self._create_intelligent_summary(content, max_length)
        
        # Extract key terms from the query
        query_terms = set()
        query_lower = query.lower()
        
        # Add exact query words
        query_words = re.findall(r'\b\w+\b', query_lower)
        query_terms.update(query_words)
        
        # Add common question patterns and their targets
        question_patterns = {
            r'what is (.+)': r'\1',
            r'define (.+)': r'\1',
            r'explain (.+)': r'\1',
            r'describe (.+)': r'\1',
            r'how does (.+) work': r'\1',
            r'tell me about (.+)': r'\1'
        }
        
        for pattern, replacement in question_patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                target_terms = re.findall(r'\b\w+\b', match.group(1))
                query_terms.update(target_terms)
        
        # First, try to find direct definitions
        definition_summary = self._find_definition_in_content(clean_content, query_terms)
        if definition_summary and len(definition_summary.strip()) > 30 and len(definition_summary) <= max_length:
            return definition_summary
        
        # Score sentences based on relevance to query
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            score = 0
            
            # Higher score for sentences that contain query terms
            for term in query_terms:
                if term in sentence_lower:
                    score += 2
            
            # Extra points for definition patterns
            definition_patterns = [
                r'is defined as', r'is described as', r'refers to', r'means',
                r'according to', r'dictionary', r'definition', r'is a',
                r'can be defined', r'is known as', r'is understood as'
            ]
            
            for pattern in definition_patterns:
                if re.search(pattern, sentence_lower):
                    score += 3
            
            # Bonus for sentences that start with the main topic
            main_topic = None
            for term in query_terms:
                if len(term) > 3 and sentence_lower.startswith(term):
                    score += 2
                    main_topic = term
                    break
            
            # Look for question-answer patterns in the content
            if '?' in sentence and any(term in sentence_lower for term in query_terms):
                score += 1
                # Check if the next sentence might be the answer
                if i + 1 < len(sentences):
                    next_sentence = sentences[i + 1].lower()
                    if any(term in next_sentence for term in query_terms):
                        score += 2
            
            sentence_scores.append((score, i, sentence))
        
        # Sort by score (descending) and original position
        sentence_scores.sort(key=lambda x: (-x[0], x[1]))
        
        # Build summary starting with highest-scoring sentences
        summary = ""
        used_indices = set()
        
        for score, idx, sentence in sentence_scores:
            if score > 0:  # Only include relevant sentences
                # Check if adding this sentence would exceed limit
                potential_summary = summary + (" " if summary else "") + sentence
                
                if len(potential_summary) <= max_length:
                    summary = potential_summary
                    used_indices.add(idx)
                else:
                    break
        
        # If we don't have enough relevant content, fill with sequential sentences
        if len(summary) < max_length * 0.7:  # If less than 70% filled
            for i, sentence in enumerate(sentences):
                if i not in used_indices:
                    potential_summary = summary + (" " if summary else "") + sentence
                    if len(potential_summary) <= max_length:
                        summary = potential_summary
                        used_indices.add(i)
                    else:
                        break
        
        # If still no good summary, fall back to intelligent summary
        if not summary.strip():
            return self._create_intelligent_summary(content, max_length)
        
        # Add ellipsis if content was truncated
        if len(clean_content) > len(summary) and not summary.endswith("..."):
            summary = summary + "..."
        
        return summary.strip()

    def _find_definition_in_content(self, content: str, query_terms: set) -> str:
        """Look for explicit definitions in the content."""
        import re
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        # Definition patterns to look for
        definition_patterns = [
            # Direct definition patterns
            r'(.+)\s+is\s+(.+)',
            r'(.+)\s+means\s+(.+)',
            r'(.+)\s+refers to\s+(.+)',
            r'(.+)\s+is defined as\s+(.+)',
            r'(.+)\s+is described as\s+(.+)',
            r'(.+)\s+can be defined as\s+(.+)',
            r'(.+)\s+is known as\s+(.+)',
            r'(.+)\s+is called\s+(.+)',
            # Authority-based definitions
            r'according to\s+.+,\s*(.+)\s+is\s+(.+)',
            r'(.+)\s+according to\s+(.+)',
            # Disease/medical specific patterns
            r'(.+)\s+is\s+a\s+(disease|virus|condition|syndrome|infection).+',
            r'(.+)\s+caused by\s+(.+)',
        ]
        
        # Look for sentences that match definition patterns
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_lower = sentence.lower()
            
            # Check if this sentence contains any query terms
            contains_query_term = any(term in sentence_lower for term in query_terms if len(term) > 2)
            
            if contains_query_term:
                for pattern in definition_patterns:
                    match = re.search(pattern, sentence_lower)
                    if match:
                        # Found a definition pattern - return this sentence and potentially the next one
                        definition = sentence
                        
                        # Look for continuation in next sentence if current one seems incomplete
                        if (i + 1 < len(sentences) and 
                            (sentence.endswith(',') or len(sentence) < 100)):
                            next_sentence = sentences[i + 1].strip()
                            if next_sentence and not next_sentence[0].isupper():
                                definition += " " + next_sentence
                        
                        return definition
        
        # Look for question-answer patterns specifically
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_lower = sentence.lower()
            
            # If we find a question about our topic, look for the answer in the next sentence(s)
            if ('?' in sentence and 
                any(term in sentence_lower for term in query_terms if len(term) > 2)):
                
                # Collect the next few sentences as the potential answer
                answer_parts = []
                for j in range(i + 1, min(i + 4, len(sentences))):  # Look at next 3 sentences max
                    next_sentence = sentences[j].strip()
                    if next_sentence:
                        answer_parts.append(next_sentence)
                        # Stop if we hit another question or if we have enough content
                        if '?' in next_sentence or len(' '.join(answer_parts)) > 300:
                            break
                
                if answer_parts:
                    return ' '.join(answer_parts)
        
        # If no explicit definition found, look for descriptive sentences
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_lower = sentence.lower()
            
            # Check if this sentence starts with the main query term and has descriptive content
            main_terms = [term for term in query_terms if len(term) > 3]
            for term in main_terms:
                if (sentence_lower.startswith(term) and 
                    ('is' in sentence_lower or 'are' in sentence_lower) and
                    len(sentence) > 20):  # Make sure it's not just a fragment
                    return sentence
        
        return ""

    def _create_llm_generated_summary(self, content: str, query: str, max_length: int = 400) -> str:
        """Use LLM to generate a query-specific summary from the content."""
        if not content or len(content.strip()) == 0:
            return "No content available"
        
        # Clean and preprocess the content more thoroughly
        import re
        clean_content = content.strip()
        
        # Remove metadata artifacts and malformed content
        clean_content = re.sub(r'FILENAME:[^\s]+\s+PAGE:\d+\s+TITLE:[^\\n]*\s+DIFFICULTY:[^\s]+\s+CONTENT:', '', clean_content, flags=re.IGNORECASE)
        clean_content = re.sub(r'FILENAME:[^\\n]*', '', clean_content, flags=re.IGNORECASE)  # Remove any FILENAME lines
        clean_content = re.sub(r'PAGE:\d+', '', clean_content, flags=re.IGNORECASE)  # Remove PAGE lines
        clean_content = re.sub(r'TITLE:[^\\n]*', '', clean_content, flags=re.IGNORECASE)  # Remove TITLE lines
        clean_content = re.sub(r'DIFFICULTY:[^\\s]*', '', clean_content, flags=re.IGNORECASE)  # Remove DIFFICULTY lines
        clean_content = re.sub(r'^[A-Z]+:[^\\n]*\\n', '', clean_content, flags=re.MULTILINE)  # Remove metadata lines
        
        # Remove common PDF artifacts and formatting issues (GENERIC patterns)
        clean_content = re.sub(r'CHAPTER \d+|Page \d+|\d+\s+[A-Z]+(?:\s+[A-Z]+)*', '', clean_content, flags=re.IGNORECASE)  # Remove chapter headers and subject titles
        clean_content = re.sub(r'\d+\.\d+\s+[A-Z][^.]*\?', '', clean_content)  # Remove section headings like "13.1 What do we Know?"
        clean_content = re.sub(r'^\s*\d+\.\s*', '', clean_content, flags=re.MULTILINE)  # Remove numbered lists
        clean_content = re.sub(r'\d{4}-\d{2}', '', clean_content)  # Remove academic years like "2020-21"
        clean_content = re.sub(r'\b[A-Z]{2,}\s+[A-Z]{2,}\b', '', clean_content)  # Remove patterns like "HIGHER PLANTS", "CELL BIOLOGY" etc.
        clean_content = re.sub(r'\s+', ' ', clean_content)  # Normalize whitespace
        clean_content = clean_content.strip()
        
        # If content is still too short or just headings, return a generic message
        if len(clean_content) < 50 or re.match(r'^[A-Z\s\d.?]+$', clean_content):
            return f"Educational content about {query.lower()} is available but requires more detailed reading."
        
        # Check if content is mostly metadata or table of contents (GENERIC detection)
        if (len(clean_content) < 150 and 
            ('FILENAME:' in content or 'PAGE:' in content or 'TITLE:' in content or 
             re.search(r'\d+\.\d+\s+[A-Z][^.]*', content) or  # Generic section pattern like "13.2 Title"
             re.search(r'Table of Contents|INDEX|BIBLIOGRAPHY', content, re.IGNORECASE))):
            return f"This appears to be a table of contents or chapter overview for {query.lower()} content."
        
        # If content is already short enough and clean, return as is
        if len(clean_content) <= max_length:
            return clean_content

        # CRITICAL FIX: Check if content is actually relevant to the query
        # Extract key terms from query and check if they appear in content
        query_terms = re.findall(r'\b\w+\b', query.lower())
        main_query_terms = [term for term in query_terms if len(term) > 3 and term not in ['what', 'how', 'why', 'when', 'where', 'tell', 'about', 'explain', 'define', 'prevent']]
        
        content_lower = clean_content.lower()
        
        # Check if the content actually contains relevant information
        relevant_terms_found = 0
        for term in main_query_terms:
            if term in content_lower:
                relevant_terms_found += 1
            else:
                # Check for fuzzy matches (typos/variations)
                if self._check_fuzzy_match(term, content_lower):
                    relevant_terms_found += 1
        
        # If content doesn't contain the main query terms (including fuzzy matches), don't hallucinate
        if relevant_terms_found == 0 and main_query_terms:
            return f"This content discusses {self._identify_content_topic(clean_content)} but does not contain information about {' '.join(main_query_terms)}."
        
        try:
            # Always try LLM approach first with better preprocessing
            if self.llm:
                # Create a very simple, focused prompt that emphasizes accuracy and handles variations
                prompt = f"""Based ONLY on the content provided below, create a brief summary that relates to: "{query}"

Content:
{clean_content[:1000]}

IMPORTANT INSTRUCTIONS:
- ONLY use information that is actually present in the content above
- If the content contains information about the topic (even with slight spelling variations), summarize it
- If the content does not contain information about the query topic, say "This content does not discuss [topic]"
- Do NOT make up or invent any information
- Write like you're explaining to a student
- Maximum 2-3 simple sentences
- No chapter references or page numbers

Summary based on the actual content:"""

                try:
                    response = self.llm.generate(prompt)
                    
                    # The response should be a string from the generate method
                    summary = response.strip() if response else ""
                    
                    # Clean up the response thoroughly
                    summary = re.sub(r'^(Summary based on the actual content:|Simple explanation:|Concise Summary:|Summary:|Answer:|Response:)\s*', '', summary, flags=re.IGNORECASE)
                    summary = re.sub(r'\*\*([^*]+)\*\*', r'\1', summary)  # Remove bold formatting
                    summary = re.sub(r'CHAPTER \d+|Page \d+', '', summary, flags=re.IGNORECASE)  # Remove generic chapter/page references
                    summary = re.sub(r'\d+\.\d+\s+', '', summary)  # Remove section numbers
                    summary = re.sub(r'\s+', ' ', summary)  # Normalize whitespace
                    summary = summary.strip()
                    
                    # Ensure proper capitalization and punctuation
                    if summary and summary[0].islower():
                        summary = summary[0].upper() + summary[1:]
                    
                    if summary and not summary.endswith(('.', '!', '?', '...')):
                        summary = summary + '.'
                    
                    # Ensure it's not too long
                    if len(summary) > max_length:
                        summary = summary[:max_length-3] + "..."
                    
                    # If we have a good summary, return it
                    if len(summary) >= 20 and len(summary) <= max_length:
                        return summary
                    
                except Exception as e:
                    pass
            
            # If LLM fails, return a basic processed version
            sentences = re.split(r'[.!?]+', clean_content)
            meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20 and not re.match(r'^[A-Z\s\d.?]+$', s.strip())]
            
            if meaningful_sentences:
                result = meaningful_sentences[0]
                if len(result) > max_length:
                    result = result[:max_length-3] + "..."
                return result
            
            return f"Content about {query.lower()} is available in the educational material."
                
        except Exception as e:
            return f"Educational information about {query.lower()} is available but needs further review."

    def _create_enhanced_summary_with_context(self, content: str, query: str, result_context: dict = None) -> str:
        """Enhanced summary generation - LLM-ONLY approach for best quality."""
        if not content or len(content.strip()) == 0:
            return "No content available"
        
        # FORCE LLM-ONLY APPROACH - No fallbacks to other methods
        # Always try LLM first and only - better content filtering and prompting
        llm_summary = self._create_llm_generated_summary(content, query)
        
        # Quality validation for LLM summary
        if (llm_summary and 
            len(llm_summary) > 30 and 
            not llm_summary.startswith("Educational content about") and
            not llm_summary.startswith("Educational information about") and
            "requires more detailed reading" not in llm_summary and
            "CHAPTER" not in llm_summary.upper() and
            not re.match(r'^\s*\d+\.\d+', llm_summary)):  # No section numbers
            return llm_summary
        
        # If LLM summary is poor, try extracting the best sentence from content
        import re
        clean_content = content.strip()
        
        # More aggressive cleaning (GENERIC patterns)
        clean_content = re.sub(r'CHAPTER \d+.*?(?=\n|\.|$)', '', clean_content, flags=re.IGNORECASE)
        clean_content = re.sub(r'\d+\.\d+\s+[A-Z][^.]*\?', '', clean_content)
        clean_content = re.sub(r'Page \d+|\d+\s+[A-Z]+(?:\s+[A-Z]+)*', '', clean_content, flags=re.IGNORECASE)  # Generic subject headers
        clean_content = re.sub(r'^\s*\d+\.\s*', '', clean_content, flags=re.MULTILINE)
        clean_content = re.sub(r'\b[A-Z]{2,}\s+[A-Z]{2,}\b', '', clean_content)  # Remove all-caps headers
        clean_content = re.sub(r'\s+', ' ', clean_content).strip()
        
        # Extract meaningful sentences
        sentences = re.split(r'[.!?]+', clean_content)
        good_sentences = []
        
        for sentence in sentences:
            s = sentence.strip()
            if (len(s) > 20 and 
                not re.match(r'^[A-Z\s\d.?]+$', s) and  # Not all caps/numbers
                not re.match(r'^\d+\.\d+', s) and  # Not section number
                not s.startswith('CHAPTER') and
                not s.startswith('Page ') and
                len([word for word in s.split() if word.isupper()]) < len(s.split()) * 0.5):  # Not mostly uppercase
                good_sentences.append(s)
        
        if good_sentences:
            # Take the first good sentence
            best_sentence = good_sentences[0]
            if len(best_sentence) > 400:
                best_sentence = best_sentence[:397] + "..."
            return best_sentence
        
        # Final fallback
        return f"Educational content about {query.lower().replace('what is ', '').replace('define ', '')} is available in the course material."

    def _create_semantic_summary(self, content: str, query: str) -> str:
        """Create summary based on semantic understanding of query intent."""
        import re
        
        # Clean content
        clean_content = re.sub(r'\s+', ' ', content.strip())
        query_lower = query.lower()
        
        # Identify query intent patterns
        if any(pattern in query_lower for pattern in ['what is', 'define', 'definition of']):
            # Definition-seeking query
            return self._extract_definition_content(clean_content, query)
        
        elif any(pattern in query_lower for pattern in ['how does', 'how to', 'explain how']):
            # Process-seeking query
            return self._extract_process_content(clean_content, query)
        
        elif any(pattern in query_lower for pattern in ['why', 'reason', 'cause']):
            # Causal explanation query
            return self._extract_causal_content(clean_content, query)
        
        else:
            # General informational query
            return self._extract_relevant_sentences(clean_content, query)

    def _extract_definition_content(self, content: str, query: str) -> str:
        """Extract definitional content from text."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        # Look for definition indicators
        definition_indicators = ['is defined as', 'refers to', 'means', 'is a', 'are a', 'according to']
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in definition_indicators):
                # Found a potential definition
                return sentence.strip()
        
        # If no explicit definition, look for descriptive sentences
        query_terms = re.findall(r'\w+', query.lower())
        main_terms = [term for term in query_terms if len(term) > 3]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in main_terms):
                if len(sentence.strip()) > 20:  # Ensure it's substantial
                    return sentence.strip()
        
        return content[:300] + "..." if len(content) > 300 else content

    def _extract_process_content(self, content: str, query: str) -> str:
        """Extract process or procedural content."""
        import re
        
        # Look for process indicators
        process_indicators = ['first', 'then', 'next', 'finally', 'step', 'process', 'method', 'procedure']
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in process_indicators):
                relevant_sentences.append(sentence.strip())
                if len(' '.join(relevant_sentences)) > 300:
                    break
        
        if relevant_sentences:
            return ' '.join(relevant_sentences)
        
        return content[:300] + "..." if len(content) > 300 else content

    def _extract_causal_content(self, content: str, query: str) -> str:
        """Extract causal explanation content."""
        import re
        
        # Look for causal indicators
        causal_indicators = ['because', 'due to', 'caused by', 'results in', 'leads to', 'reason', 'since']
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in causal_indicators):
                relevant_sentences.append(sentence.strip())
                if len(' '.join(relevant_sentences)) > 300:
                    break
        
        if relevant_sentences:
            return ' '.join(relevant_sentences)
        
        return content[:300] + "..." if len(content) > 300 else content

    def _extract_relevant_sentences(self, content: str, query: str) -> str:
        """Extract sentences most relevant to the query."""
        import re
        
        query_terms = set(re.findall(r'\w+', query.lower()))
        query_terms = {term for term in query_terms if len(term) > 2}
        
        sentences = re.split(r'(?<=[.!?])\s+', content)
        scored_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for term in query_terms if term in sentence_lower)
            if score > 0:
                scored_sentences.append((score, sentence.strip()))
        
        # Sort by relevance score
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # Combine top sentences
        result_sentences = []
        total_length = 0
        for score, sentence in scored_sentences:
            if total_length + len(sentence) <= 350:
                result_sentences.append(sentence)
                total_length += len(sentence)
            else:
                break
        
        if result_sentences:
            return ' '.join(result_sentences)
        
        return content[:300] + "..." if len(content) > 300 else content

    def _is_enhanced_database(self) -> bool:
        """Check if the database supports enhanced content discovery features."""
        # Check if it's the enhanced version specifically, not just the base ContentDiscoveryVectorDB
        from vector_db.enhanced_content_discovery_db import EnhancedContentDiscoveryVectorDB
        return isinstance(self.db, EnhancedContentDiscoveryVectorDB)
    
    def _generate_simple_response(self, query: str, results: list) -> str:
        """Generate a simple LLM response for basic databases."""
        # Extract text content from results
        context_chunks = []
        for result in results:
            if isinstance(result, dict) and "content" in result:
                context_chunks.append(result["content"])
            elif isinstance(result, tuple):
                context_chunks.append(result[0])  # Text is first element
            else:
                context_chunks.append(str(result))
        
        if not context_chunks:
            return f"I couldn't find any relevant information to answer your question about '{query}'."

        # Limit context to avoid overwhelming the model
        context = "\n\n".join(context_chunks[:3])  # Use only top 3 chunks
        
        prompt = f"""Based on the following information, answer the question.

Information:
{context}

Question: {query}

Answer the question using the information provided above. Be specific and detailed."""
        
        try:
            response = self.llm.generate(prompt)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _get_content_recommendations(self, query: str, results: List[Dict]) -> Dict:
        """Generate content recommendations with enhanced educational metadata."""
        recommendations = []
        seen_files = set()
        
        for i, result in enumerate(results[:self.top_k]):
            filename = result["filename"]
            page = result["page"]
            
            # Check if this is enhanced database result with educational metadata
            if self._is_enhanced_database() and 'learning_objectives' in result:
                # Use enhanced educational metadata
                title = result.get('title', filename.replace('.pdf', '').replace('_', ' ').title())
                difficulty = result.get('difficulty', 'unknown')
                learning_objectives = result.get('learning_objectives', [])
                key_concepts = result.get('key_concepts', [])
                clean_content = result.get('clean_content', result.get('content', ''))
                
                # Generate LLM-based summary from clean content
                summary = self._create_llm_generated_summary(clean_content, query)
                
                # QUALITY CHECK: Skip results with poor quality summaries (metadata artifacts)
                if self._is_poor_quality_summary(summary):
                    continue  # Skip this result entirely
                
                # Use key concepts as keywords, fallback to content extraction
                keywords = key_concepts if key_concepts else self._extract_keywords_from_content(clean_content)
                
                # Get metadata for this file
                file_metadata = self._get_metadata_for_file(filename)
                
                # Use metadata if available, otherwise generate from key concepts or filename
                if file_metadata:
                    course_id = file_metadata.get('course_id', 'unknown_course_id')
                    course_title = file_metadata.get('course_title', filename.replace('.pdf', '').replace('_', ' ').title())
                    content_title = file_metadata.get('content_title', title)
                    course = file_metadata.get('course', 'General Studies')
                    author = file_metadata.get('author', 'Document Author')
                    subject = file_metadata.get('subject', 'General')
                    grade = file_metadata.get('grade', 'General')
                    board = file_metadata.get('board', 'General')
                else:
                    # Fallback to dynamic generation
                    course_id = f"generated_{filename.replace('.pdf', '').replace(' ', '_').lower()}"
                    if key_concepts:
                        course_title = f"{key_concepts[0].title()} Studies"
                        course = f"{key_concepts[0].title()} Studies"
                    else:
                        course_title = f"{filename.replace('.pdf', '').replace('_', ' ').title()} Studies"
                        course = f"{filename.replace('.pdf', '').replace('_', ' ').title()} Studies"
                    content_title = title
                    author = "Document Author"
                    subject = "General"
                    grade = "General" 
                    board = "General"
                
                file_key = f"{filename}_page_{page}"
                if file_key not in seen_files:
                    seen_files.add(file_key)
                    
                    recommendation = {
                        "filename": filename,  # ACTUAL filename from database
                        "title": title,
                        "content_title": content_title,
                        "author": author,
                        "course_id": course_id,
                        "course_title": course_title,
                        "course": course,
                        "subject": subject,
                        "grade": grade,
                        "board": board,
                        "page_number": page,
                        "section": f"Page {page}",
                        "keywords": keywords[:5],  # Limit to top 5
                        "summary": summary,
                        "difficulty_level": difficulty,
                        "learning_objectives": learning_objectives[:3],  # Top 3 objectives
                        "key_concepts": key_concepts[:5],  # Top 5 concepts
                        "relevance_score": round(1.0 - (i * 0.1), 2)  # Decreasing relevance
                    }
                    recommendations.append(recommendation)
            else:
                # Fallback to basic metadata extraction for non-enhanced databases
                content = result["content"]
                
                # Extract keywords from content
                keywords = self._extract_keywords_from_content(content)
                
                # Extract keywords that match the query
                query_lower = query.lower()
                query_words = set(query_lower.split())
                matched_keywords = [kw for kw in keywords if any(qw in kw for qw in query_words)]
                
                # If no query-specific keywords found, use general keywords from content
                if not matched_keywords:
                    matched_keywords = keywords[:5]  # Top 5 keywords
                
                file_key = f"{filename}_page_{page}"
                if file_key not in seen_files:
                    seen_files.add(file_key)
                    
                    summary = self._create_llm_generated_summary(content, query)
                    
                    # QUALITY CHECK: Skip results with poor quality summaries (metadata artifacts)
                    if self._is_poor_quality_summary(summary):
                        continue  # Skip this result entirely
                    
                    # Generate title from filename
                    title = filename.replace('.pdf', '').replace('_', ' ').title()
                    
                    # Get metadata for this file
                    file_metadata = self._get_metadata_for_file(filename)
                    
                    # Use metadata if available, otherwise generate from keywords
                    if file_metadata:
                        course_id = file_metadata.get('course_id', 'unknown_course_id')
                        course_title = file_metadata.get('course_title', title)
                        content_title = file_metadata.get('content_title', title)
                        course = file_metadata.get('course', 'General Studies')
                        author = file_metadata.get('author', 'Document Author')
                        subject = file_metadata.get('subject', 'General')
                        grade = file_metadata.get('grade', 'General')
                        board = file_metadata.get('board', 'General')
                    else:
                        # Fallback to dynamic generation
                        course_id = f"generated_{filename.replace('.pdf', '').replace(' ', '_').lower()}"
                        course_title = f"{matched_keywords[0].title()} Studies" if matched_keywords else "General Studies"
                        content_title = title
                        course = f"{matched_keywords[0].title()} Studies" if matched_keywords else "General Studies"
                        author = "Document Author"
                        subject = "General"
                        grade = "General"
                        board = "General"
                    
                    recommendation = {
                        "filename": filename,  # ACTUAL filename from database
                        "title": title,
                        "content_title": content_title,
                        "author": author,
                        "course_id": course_id,
                        "course_title": course_title,
                        "course": course,
                        "subject": subject,
                        "grade": grade,
                        "board": board,
                        "page_number": page,
                        "section": f"Page {page}",
                        "keywords": matched_keywords,
                        "summary": summary,
                        "relevance_score": round(1.0 - (i * 0.1), 2)  # Decreasing relevance
                    }
                    recommendations.append(recommendation)
        
        return {
            "total_recommendations": len(recommendations),
            "query": query,
            "recommendations": recommendations,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def _create_text_response(self, recommendations_data: Dict) -> str:
        """Create human-readable text response with enhanced educational metadata"""
        query = recommendations_data["query"]
        recommendations = recommendations_data["recommendations"]
        total = recommendations_data["total_recommendations"]
        
        if not recommendations:
            return f"I couldn't find any relevant content for '{query}'. Please try a different search term."
        
        response = f"🔍 **Content Discovery Results for: '{query}'**\n\n"
        response += f"Found {total} relevant educational resources:\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            response += f"**{i}. {rec['title']}**\n"
            response += f"   📄 File: {rec['filename']} (Page {rec['page_number']})\n"  # REAL filename!
            response += f"   📚 Course: {rec['course']}\n"
            
            # Show enhanced educational metadata if available
            if 'difficulty_level' in rec and rec['difficulty_level'] != 'unknown':
                response += f"   📊 Difficulty: {rec['difficulty_level'].title()}\n"
            
            if 'learning_objectives' in rec and rec['learning_objectives']:
                objectives_text = ', '.join(rec['learning_objectives'][:2])  # Show top 2
                response += f"   🎯 Learning Objectives: {objectives_text}\n"
            
            if 'key_concepts' in rec and rec['key_concepts']:
                concepts_text = ', '.join(rec['key_concepts'][:3])  # Show top 3
                response += f"   🧠 Key Concepts: {concepts_text}\n"
            
            response += f"   🏷️  Keywords: {', '.join(rec['keywords'][:5])}\n"
            response += f"   📝 Summary: {rec['summary']}\n"
            response += f"   ⭐ Relevance: {rec['relevance_score']}\n\n"
        
        return response

    def run(self, query: str):
        """Run content discovery with actual database filenames"""
        try:
            bm25, documents_data = self._initialize_bm25()

            query_embedding = self.embedding_model.create_embedding(query)
            
            # Use enhanced search if available, otherwise fallback to basic search
            if self._is_enhanced_database() and hasattr(self.db, 'search_enhanced'):
                vector_results = self.db.search_enhanced(query_embedding, top_k=self.top_k)
            else:
                vector_results = self.db.search(query_embedding, top_k=self.top_k)

            # Use vector results directly (they already have the right format)
            final_results = vector_results
            
            # Optionally enhance with BM25 if available
            if bm25 and documents_data:
                tokenized_query = query.split(" ")
                bm25_scores = bm25.get_scores(tokenized_query)
                
                # Get top BM25 results
                top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:self.top_k]
                bm25_results = [documents_data[i] for i in top_bm25_indices]
                
                # Combine results (prioritize vector search)
                combined_results = vector_results.copy()
                for bm25_result in bm25_results:
                    if bm25_result not in combined_results:
                        combined_results.append(bm25_result)
                
                final_results = combined_results[:self.top_k]

            # Check relevance scores - filter out results with poor similarity
            RELEVANCE_THRESHOLD = 0.3  # Higher threshold for better content matching
            relevant_results = []
            
            for result in final_results:
                # Check if result has a score/distance field
                score = result.get('score', 0)
                distance = result.get('distance', None)
                
                # Use score if available (higher is better), otherwise convert distance (lower is better)
                if score > 0:
                    similarity = score  # Score is already similarity (0-1, higher is better)
                elif distance is not None:
                    similarity = 1.0 - distance if distance <= 1.0 else max(0.0, 1.0 - (distance / 2.0))
                else:
                    similarity = 0.0
                
                # CONTENT RELEVANCE CHECK: Analyze if content actually contains query-related information
                content = result.get('content', '') or result.get('clean_content', '')
                if self._is_content_relevant_to_query(content, query):
                    # Only include results above threshold AND with relevant content
                    if similarity >= RELEVANCE_THRESHOLD:
                        relevant_results.append(result)
                else:
                    # Skip content that doesn't actually relate to the query
                    continue
            
            if not relevant_results:
                error_msg = f"No relevant content found for '{query}'. The available content doesn't match your query well enough."
                
                if self.return_json:
                    error_response = {
                        "query": query,
                        "status": "error",
                        "error_code": "low_relevance",
                        "error_message": error_msg,
                        "suggestion": "Try a different query or check if relevant content has been loaded",
                        "threshold": RELEVANCE_THRESHOLD
                    }
                    return json.dumps(error_response, indent=2)
                return error_msg
            
            # Use filtered relevant results
            # Use the relevant results instead of all results
            final_results = relevant_results

            # Generate recommendations with ACTUAL filenames
            if self._is_enhanced_database():
                recommendations_data = self._get_content_recommendations(query, final_results)
            else:
                # Fallback for basic database - generate simple LLM response
                return self._generate_simple_response(query, final_results)

            if self.return_json:
                # Return structured JSON response
                response = {
                    "query": recommendations_data["query"],
                    "status": "success",
                    "total_results": recommendations_data["total_recommendations"],
                    "recommendations": recommendations_data["recommendations"],
                    "message": "Content discovery completed successfully"
                }
                
                return json.dumps(response, indent=2)
            else:
                # Return human-readable text
                return self._create_text_response(recommendations_data)

        except Exception as e:
            error_msg = f"Error in content discovery: {str(e)}"
            
            if self.return_json:
                error_response = {
                    "query": query,
                    "status": "error",
                    "error_code": "processing_error",
                    "error_message": error_msg
                }
                return json.dumps(error_response, indent=2)
            
            return error_msg

    def _identify_content_topic(self, content: str) -> str:
        """Identify what the content is actually about based on keywords - GENERIC approach."""
        import re
        
        content_lower = content.lower()
        
        # Extract most frequent meaningful words (dynamic topic detection)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content_lower)
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'this', 'that', 'these', 'those', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'about', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out',
            'off', 'over', 'under', 'again', 'further', 'then', 'once', 'also', 'such',
            'very', 'more', 'most', 'some', 'any', 'many', 'much', 'each', 'every'
        }
        
        meaningful_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        if not meaningful_words:
            return "general content"
        
        # Count word frequency and get top terms
        from collections import Counter
        word_counts = Counter(meaningful_words)
        top_words = [word for word, count in word_counts.most_common(5)]
        
        # Use top words to describe the topic
        if len(top_words) >= 2:
            return f"{top_words[0]} and {top_words[1]} studies"
        elif len(top_words) >= 1:
            return f"{top_words[0]} studies"
        else:
            return "general content"
    
    def _is_content_relevant_to_query(self, content: str, query: str) -> bool:
        """Check if content actually contains information relevant to the query BEFORE calling LLM."""
        import re
        
        if not content or not query:
            return False
        
        # Clean and normalize content and query
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Extract meaningful terms from query (exclude common question words)
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'to', 'and', 'or', 'of', 'in', 'on', 'at', 'for', 'with', 'by'}
        query_terms = re.findall(r'\b\w{3,}\b', query_lower)
        meaningful_query_terms = [term for term in query_terms if term not in stop_words]
        
        if not meaningful_query_terms:
            return False
        
        # Check for direct term matches and fuzzy matches
        direct_matches = 0
        fuzzy_matches = 0
        
        for term in meaningful_query_terms:
            # Exact match
            if term in content_lower:
                direct_matches += 1
            else:
                # Fuzzy matching for common typos and variations
                fuzzy_match_found = self._check_fuzzy_match(term, content_lower)
                if fuzzy_match_found:
                    fuzzy_matches += 1
        
        total_matches = direct_matches + fuzzy_matches
        
        # If most terms are present (including fuzzy matches), it's likely relevant
        if total_matches >= len(meaningful_query_terms) * 0.5:  # At least 50% of terms match
            return True
        
        # Check for semantic relevance using keyword categories
        query_categories = self._categorize_query(query_lower)
        content_categories = self._categorize_content(content_lower)
        
        # If query and content share categories, they're related
        if query_categories.intersection(content_categories):
            return True
        
        # Final check: Dynamic semantic relationship detection
        # Instead of hardcoded topic indicators, use semantic word analysis
        for main_term in meaningful_query_terms:
            # Look for semantic relationships using word patterns
            semantic_patterns = [
                rf'\b{main_term}\w*\b',  # Variations of the term (e.g., prevent -> prevention)
                rf'\b\w*{main_term}\w*\b',  # Words containing the term
            ]
            
            for pattern in semantic_patterns:
                if re.search(pattern, content_lower):
                    return True
            
            # Look for definitional patterns around the term
            definition_patterns = [
                rf'{main_term}\s+(is|are|means|refers)',
                rf'(definition|meaning)\s+.*{main_term}',
                rf'{main_term}.*\b(process|method|technique|approach|system)\b'
            ]
            
            for pattern in definition_patterns:
                if re.search(pattern, content_lower):
                    return True
        
        return False
    
    def _check_fuzzy_match(self, query_term: str, content: str) -> bool:
        """Generic fuzzy matching based purely on algorithmic similarity - no hardcoded patterns."""
        import re
        
        if len(query_term) < 4:  # Skip very short terms
            return False
        
        # Extract all meaningful words from content (4+ characters)
        words_in_content = re.findall(r'\b\w{4,}\b', content.lower())
        
        for word in words_in_content:
            # Check similarity using multiple generic approaches
            if self._is_similar_word(query_term.lower(), word):
                return True
        
        return False
    
    def _is_similar_word(self, term1: str, term2: str) -> bool:
        """Generic word similarity detection using multiple algorithms."""
        # Skip if length difference is too large
        if abs(len(term1) - len(term2)) > 3:
            return False
        
        # 1. Edit distance approach - allows for typos
        if self._simple_edit_distance(term1, term2) <= 2:
            return True
        
        # 2. Character overlap approach - handles letter swaps/omissions
        if self._character_overlap_similarity(term1, term2) >= 0.8:
            return True
        
        # 3. Longest common subsequence - handles missing/extra letters
        if self._lcs_similarity(term1, term2) >= 0.75:
            return True
        
        # 4. Phonetic similarity - handles sound-alike words
        if len(term1) >= 5 and len(term2) >= 5:
            if self._phonetic_similarity(term1, term2) >= 0.8:
                return True
        
        return False
    
    def _character_overlap_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity based on character overlap."""
        from collections import Counter
        
        chars1 = Counter(s1.lower())
        chars2 = Counter(s2.lower())
        
        # Calculate intersection and union
        intersection = sum((chars1 & chars2).values())
        union = sum((chars1 | chars2).values())
        
        return intersection / union if union > 0 else 0.0
    
    def _lcs_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity using Longest Common Subsequence."""
        def lcs_length(x, y):
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        
        lcs_len = lcs_length(s1.lower(), s2.lower())
        max_len = max(len(s1), len(s2))
        return lcs_len / max_len if max_len > 0 else 0.0
    
    def _phonetic_similarity(self, s1: str, s2: str) -> float:
        """Basic phonetic similarity - maps similar sounds."""
        # Simple character substitution rules for common sound-alike patterns
        def normalize_phonetic(word):
            word = word.lower()
            # Common phonetic substitutions
            substitutions = [
                ('ph', 'f'), ('gh', 'f'), ('c', 'k'), ('ck', 'k'),
                ('qu', 'kw'), ('x', 'ks'), ('z', 's'), ('th', 't'),
                ('sh', 's'), ('ch', 's'), ('tion', 'shun'), ('sion', 'shun')
            ]
            for old, new in substitutions:
                word = word.replace(old, new)
            return word
        
        norm1 = normalize_phonetic(s1)
        norm2 = normalize_phonetic(s2)
        
        # Use edit distance on normalized forms
        distance = self._simple_edit_distance(norm1, norm2)
        max_len = max(len(norm1), len(norm2))
        return 1.0 - (distance / max_len) if max_len > 0 else 0.0
    
    def _simple_edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings - optimized for fuzzy matching."""
        if not s1 or not s2:
            return max(len(s1), len(s2))
        
        if s1 == s2:
            return 0
        
        # Optimize for very different lengths
        if abs(len(s1) - len(s2)) > 3:
            return 999
        
        # Ensure s1 is shorter for memory optimization
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        
        # Use single array for space optimization
        distances = list(range(len(s1) + 1))
        
        for i2, c2 in enumerate(s2):
            new_distances = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    new_distances.append(distances[i1])
                else:
                    # Cost: insertion, deletion, substitution
                    new_distances.append(1 + min(
                        distances[i1],      # substitution
                        distances[i1 + 1],  # insertion
                        new_distances[-1]   # deletion
                    ))
            distances = new_distances
        
        return distances[-1]
    
    def _categorize_query(self, query: str) -> set:
        """Categorize query into topic areas - GENERIC approach using dynamic keyword extraction."""
        import re
        
        categories = set()
        
        # Extract meaningful terms from query
        query_terms = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        
        # Remove common question words
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'tell', 'explain', 'define', 'describe'}
        meaningful_terms = [term for term in query_terms if term not in stop_words]
        
        # Use the meaningful terms as categories
        for term in meaningful_terms:
            categories.add(term)
        
        # GENERIC APPROACH: Use word frequency and linguistic patterns instead of hardcoded categories
        # Extract word stems and patterns to identify subject areas dynamically
        word_stems = self._extract_word_stems(meaningful_terms)
        
        # Add significant word stems as categories
        for stem in word_stems:
            if len(stem) > 3:  # Only meaningful stems
                categories.add(stem)
        
        return categories if categories else {'general'}
    
    def _extract_word_stems(self, words: list) -> list:
        """Extract meaningful word stems using linguistic patterns."""
        stems = []
        
        for word in words:
            if len(word) < 4:
                stems.append(word)
                continue
            
            # Simple stemming by removing common suffixes
            stem = word
            suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'ion', 'tion', 'sion', 'ment', 'ness', 'ity', 'al', 'ic', 'ical']
            
            for suffix in sorted(suffixes, key=len, reverse=True):  # Try longer suffixes first
                if stem.endswith(suffix) and len(stem) > len(suffix) + 2:
                    stem = stem[:-len(suffix)]
                    break
            
            stems.append(stem)
        
        return stems
    
    def _categorize_content(self, content: str) -> set:
        """Categorize content into topic areas - GENERIC approach using dynamic analysis."""
        import re
        from collections import Counter
        
        categories = set()
        content_lower = content.lower()
        
        # Extract meaningful words (nouns and adjectives typically)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content_lower)
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'this', 'that', 'these', 'those', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'about', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out',
            'off', 'over', 'under', 'again', 'further', 'then', 'once', 'also', 'such',
            'very', 'more', 'most', 'some', 'any', 'many', 'much', 'each', 'every',
            'page', 'chapter', 'section', 'content', 'text', 'document', 'file'
        }
        
        meaningful_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        if not meaningful_words:
            return {'general'}
        
        # Count word frequency and get top terms as categories
        word_counts = Counter(meaningful_words)
        top_words = [word for word, count in word_counts.most_common(10) if count > 1]
        
        # Add frequent words as categories
        for word in top_words[:5]:  # Top 5 most frequent words
            categories.add(word)
        
        # GENERIC APPROACH: Dynamic domain detection based on word patterns and frequencies
        # Instead of hardcoded domains, detect patterns dynamically
        
        # Group words by semantic similarity (without predefined categories)
        word_clusters = self._cluster_words_by_patterns(meaningful_words)
        
        # Add the most significant word clusters as categories
        for cluster_name, cluster_words in word_clusters.items():
            if len(cluster_words) >= 2:  # Only if cluster has multiple related words
                categories.add(cluster_name)
        
        return categories if categories else {'general'}
    
    def _cluster_words_by_patterns(self, words: list) -> dict:
        """Dynamically cluster words by common patterns - no hardcoded domains."""
        from collections import defaultdict
        
        clusters = defaultdict(list)
        
        # Dynamic pattern detection based on word endings and common linguistic patterns
        for word in words:
            if len(word) < 4:
                continue
            
            # Cluster by common word endings (linguistic approach)
            if word.endswith(('tion', 'sion')):
                clusters['process_concepts'].append(word)
            elif word.endswith(('ment', 'ence', 'ance')):
                clusters['state_concepts'].append(word)
            elif word.endswith(('ing', 'ling')):
                clusters['action_concepts'].append(word)
            elif word.endswith(('ism', 'ology', 'ics')):
                clusters['field_concepts'].append(word)
            elif word.endswith(('able', 'ible')):
                clusters['quality_concepts'].append(word)
            else:
                # Cluster by word length and patterns for root concepts
                if len(word) >= 6:
                    clusters['complex_concepts'].append(word)
                else:
                    clusters['basic_concepts'].append(word)
        
        # Convert to meaningful cluster names using the most frequent word in each cluster
        final_clusters = {}
        for cluster_type, word_list in clusters.items():
            if len(word_list) >= 2:  # Only keep clusters with multiple words
                # Use the most frequent or longest word as the cluster representative
                representative_word = max(word_list, key=lambda w: (word_list.count(w), len(w)))
                final_clusters[representative_word] = word_list
        
        return final_clusters
    
    def _is_poor_quality_summary(self, summary: str) -> bool:
        """Check if a summary contains metadata artifacts or is of poor quality."""
        if not summary:
            return True
        
        # Generic metadata indicators (not content-specific)
        metadata_indicators = [
            'FILENAME:', 'PAGE:', 'TITLE:', 'DIFFICULTY:', 'CONTENT:', 'CHAPTER',
            'Table of Contents', 'INDEX:', 'BIBLIOGRAPHY:', 'REFERENCES:'
        ]
        
        for indicator in metadata_indicators:
            if indicator in summary:
                return True
        
        # Check for section numbering patterns (generic)
        import re
        if re.search(r'\b\d+\.\d+\s+[A-Z][^.]*\b', summary):  # Pattern like "13.2 Title"
            return True
        
        # Check if summary is mostly just numbers and short words (table of contents style)
        if (len(summary) < 100 and 
            re.search(r'^\s*\d+\..*\d+\..*\d+\.', summary)):  # Multiple numbered sections
            return True
        
        # Check if summary is just fragmented text
        words = summary.split()
        if len(words) < 10:  # Too short to be meaningful
            return True
        
        # Check for incomplete sentences (more than 50% of "words" are single characters or numbers)
        short_fragments = [word for word in words if len(word) <= 2]
        if len(short_fragments) > len(words) * 0.5:
            return True
        
        # Check if it's mostly uppercase (often indicates headers/metadata)
        uppercase_words = [word for word in words if word.isupper() and len(word) > 2]
        if len(uppercase_words) > len(words) * 0.4:  # More than 40% uppercase
            return True
        
        # Additional checks for better quality detection (more balanced)
        # Check for repeated section numbers or table of contents patterns
        section_numbers = re.findall(r'\b\d+\.\d*\b', summary)
        if len(section_numbers) > 5:  # Too many section numbers indicates TOC (increased threshold)
            return True
        
        # Check if summary contains filename artifacts (generic pattern)
        if re.search(r'\w+_\w+_[\w]+\.pdf', summary, re.IGNORECASE):  # Any courseware/document pattern
            return True
        
        # Check if summary starts with filename pattern or broken text
        if re.match(r'^[a-z]{1,3}\s+[a-z_]+\.pdf', summary, re.IGNORECASE):
            return True
        
        # Check for garbled text patterns (more specific)
        garbled_patterns = [
            r'^[a-z]{1,2}\s+[a-z_]+\.pdf',  # Broken filename start
            r'\b[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\b',  # Single letter fragments
        ]
        
        for pattern in garbled_patterns:
            if re.search(pattern, summary, re.IGNORECASE):
                return True
        
        return False
