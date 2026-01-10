# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Install dependencies:**
```bash
uv sync
```

**Run the application:**
```bash
./run.sh
# OR manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

**Access points:**
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Environment Setup

Create a `.env` file in the root directory:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Architecture Overview

This is a Retrieval-Augmented Generation (RAG) system for answering questions about course materials. The system uses:
- **ChromaDB** for vector storage (dual-collection approach)
- **Anthropic's Claude** with tool-calling for AI generation
- **Sentence Transformers** for embeddings
- **FastAPI** for the backend API
- Vanilla JavaScript frontend

### Core Data Flow

1. **Document Processing** (`document_processor.py`):
   - Parses course documents with expected format: Course Title → Course Link → Course Instructor → Lesson sections
   - Extracts structured Course/Lesson metadata
   - Chunks lesson content with overlap (default: 800 char chunks, 100 char overlap)
   - First chunk of each lesson gets prefixed with "Lesson {N} content:"

2. **Vector Storage** (`vector_store.py`):
   - **Two ChromaDB collections**:
     - `course_catalog`: Stores course metadata (title, instructor, lessons JSON)
     - `course_content`: Stores actual lesson content chunks
   - Search workflow: query → optional course name resolution via semantic search → filtered content search
   - Course titles are used as unique IDs

3. **AI Generation** (`ai_generator.py`):
   - Uses Claude with tool-calling capability
   - System prompt instructs: one search per query maximum, no meta-commentary, brief/concise responses
   - Two-step process: initial LLM call → tool execution → final LLM call with tool results

4. **Tool System** (`search_tools.py`):
   - `CourseSearchTool`: Enables Claude to search with semantic course name matching
   - Tool accepts: query (required), course_name (optional), lesson_number (optional)
   - Tracks sources for UI display via `last_sources` attribute

5. **RAG Orchestration** (`rag_system.py`):
   - Main coordinator that wires together all components
   - Handles document ingestion (single files or folders)
   - Prevents duplicate course loading by checking existing titles
   - Query flow: user query → AI with tools → response + sources

6. **Session Management** (`session_manager.py`):
   - Maintains conversation history per session
   - Configurable history depth (default: 2 exchanges)

### Configuration (`config.py`)

Key settings:
- `ANTHROPIC_MODEL`: "claude-sonnet-4-20250514"
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2"
- `CHUNK_SIZE`: 800 characters
- `CHUNK_OVERLAP`: 100 characters
- `MAX_RESULTS`: 5 search results
- `MAX_HISTORY`: 2 conversation exchanges
- `CHROMA_PATH`: "./chroma_db"

### Course Document Format

Expected structure for files in `docs/`:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [lesson title]
Lesson Link: [url]
[lesson content...]

Lesson 1: [lesson title]
Lesson Link: [url]
[lesson content...]
```

### API Endpoints (`app.py`)

- `POST /api/query`: Submit questions (returns answer + sources + session_id)
- `GET /api/courses`: Get course statistics (total courses + titles list)
- Startup: Auto-loads documents from `../docs` folder

### Frontend (`frontend/`)

Single-page application with vanilla JS that interfaces with the API endpoints.

## Important Implementation Notes

1. **Duplicate Prevention**: The system checks `course_catalog` for existing titles before adding new courses to avoid re-processing
2. **Tool Calling Pattern**: Claude decides when to use the search tool based on query content (general knowledge vs course-specific)
3. **Source Tracking**: Sources flow from tool execution → ToolManager → RAGSystem → API response → UI
4. **Chunk Context**: First chunks get lesson number prefix to maintain context in vector search
5. **Semantic Course Matching**: Partial course names work (e.g., "MCP" matches "Introduction to MCP") via vector similarity search on the catalog collection
