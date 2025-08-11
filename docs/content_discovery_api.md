# Content Discovery API Documentation

## Overview
The Content Discovery Agent now supports structured JSON responses, making it easy to integrate with modern UIs, mobile apps, and web applications.

## Usage

### Command Line
```bash
# Get JSON response
python app.py --query "I want to learn about photosynthesis" --json

# Get text response (backward compatibility)
python app.py --query "I want to learn about photosynthesis"
```

### API Response Schema

#### Success Response
```json
{
  "query": "string",                    // Original user query
  "status": "success",                  // Response status
  "total_results": 5,                   // Number of recommendations found
  "recommendations": [                  // Array of content recommendations
    {
      "filename": "biology_plants.pdf",
      "title": "Plant Biology and Growth",
      "author": "Dr. Sarah Johnson",
      "course": "Biology",
      "page_number": 4,
      "section": "Chapter 4: Photosynthesis Process",
      "summary": "Brief content summary...",
      "relevance_score": 1.0,            // 0.0 to 1.0 relevance
      "keywords": ["photosynthesis", "chlorophyll"]  // Matched keywords
    }
  ],
  "message": "Helpful tip for user",     // User guidance message
  "processing_time_ms": 108.22          // Performance metric
}
```

#### Error Response
```json
{
  "query": "string",
  "status": "error",
  "error_code": "NO_CONTENT_FOUND",     // Machine-readable error code
  "error_message": "Human readable error message",
  "total_results": 0,
  "recommendations": []
}
```

## Field Descriptions

### ContentRecommendation Object
| Field | Type | Description |
|-------|------|-------------|
| `filename` | string | PDF filename (e.g., "biology_plants.pdf") |
| `title` | string | Full title of the educational content |
| `author` | string | Content author name |
| `course` | string | Subject area (Biology, Physics, Mathematics, etc.) |
| `page_number` | integer | Page where content is found |
| `section` | string | Chapter or section name |
| `summary` | string | Brief content preview (200 chars max) |
| `relevance_score` | float | Relevance to query (1.0 = most relevant) |
| `keywords` | array[string] | Keywords that matched the query |

## UI Integration Examples

### React/Vue Component
```javascript
const ContentRecommendations = ({ query }) => {
  const [results, setResults] = useState(null);
  
  const searchContent = async (query) => {
    const response = await fetch(`/api/search?q=${query}&format=json`);
    const data = await response.json();
    setResults(data);
  };
  
  return (
    <div className="content-recommendations">
      {results?.recommendations.map((item, index) => (
        <div key={index} className="recommendation-card">
          <h3>{item.title}</h3>
          <p><strong>Course:</strong> {item.course}</p>
          <p><strong>Author:</strong> {item.author}</p>
          <p><strong>Page:</strong> {item.page_number}</p>
          <p><strong>Section:</strong> {item.section}</p>
          <div className="keywords">
            {item.keywords?.map(keyword => (
              <span key={keyword} className="keyword-tag">{keyword}</span>
            ))}
          </div>
          <p className="summary">{item.summary}</p>
          <div className="relevance">
            Relevance: {(item.relevance_score * 100).toFixed(0)}%
          </div>
        </div>
      ))}
    </div>
  );
};
```

### Mobile App (React Native)
```javascript
const ContentDiscoveryScreen = () => {
  const renderRecommendation = ({ item }) => (
    <View style={styles.card}>
      <Text style={styles.title}>{item.title}</Text>
      <Text style={styles.metadata}>
        📖 {item.course} • 👨‍🏫 {item.author} • 📄 Page {item.page_number}
      </Text>
      <Text style={styles.section}>{item.section}</Text>
      <Text style={styles.summary}>{item.summary}</Text>
      <View style={styles.keywords}>
        {item.keywords?.map(keyword => (
          <Text key={keyword} style={styles.keyword}>{keyword}</Text>
        ))}
      </View>
      <ProgressBar progress={item.relevance_score} />
    </View>
  );
  
  return (
    <FlatList
      data={searchResults?.recommendations || []}
      renderItem={renderRecommendation}
      keyExtractor={(item, index) => index.toString()}
    />
  );
};
```

## Error Codes

| Code | Description |
|------|-------------|
| `NO_CONTENT_FOUND` | No matching educational content found |
| `PROCESSING_ERROR` | Internal error during search processing |

## Performance Metrics

The API includes `processing_time_ms` to help with:
- Performance monitoring
- User experience optimization
- System scaling decisions

## Example Queries

### Biology Topics
```bash
python app.py --query "I want to learn about photosynthesis" --json
python app.py --query "How do plants grow?" --json
python app.py --query "What is plant structure?" --json
```

### Physics Topics
```bash
python app.py --query "Explain Newton's laws of motion" --json
python app.py --query "What are forces in physics?" --json
```

### Mathematics Topics
```bash
python app.py --query "Which course covers algebra?" --json
python app.py --query "How to solve quadratic equations?" --json
```

## Integration Tips

1. **Error Handling**: Always check the `status` field before processing recommendations
2. **Relevance Filtering**: Consider filtering results below a certain relevance threshold
3. **Keyword Highlighting**: Use the `keywords` array to highlight matching terms in the UI
4. **Performance**: The `processing_time_ms` can help optimize user experience
5. **Pagination**: Currently returns top 5 results; implement client-side pagination if needed

## Future Enhancements

- Real metadata storage and retrieval
- Advanced filtering by course, grade, board
- Content similarity recommendations
- User preference learning
- Multi-language support
