# Vibe Journey – Agentic Travel Itinerary Generator

Vibe Journey is an **Agentic RAG (Retrieval-Augmented Generation) system** that generates detailed, personalized travel itineraries using:

- **CrewAI multi-agent orchestration**
- **Gemini Flash LLM**
- **Serper real-time search tools**
- **A Flask web API backend**

The system analyzes group interests, performs live venue/event research, and builds a day-by-day, time-aware itinerary formatted in clean Markdown.

---

## 🚀 Features

### ✅ Multi-Agent Architecture
The system uses three specialized agents:

- **Real-Time Event & Activity Researcher**  
  Finds events, venues, restaurants, sports facilities (with emphasis on indoor pickleball, cuisine subtypes, etc.)

- **Local Travel Expert**  
  Evaluates search results and selects the best venues with logistics, hours, cost, etc.

- **Itinerary Coordinator**  
  Creates a time-aware, structured daily itinerary that satisfies all required interests.

### ✅ Intelligent Interest Aggregation
Interest matching is case-insensitive and includes:

- Priority interests  
- Common vs. unique interests  
- Ordered preference lists used to guide planning

### ✅ Flask API Endpoint
Frontend or external services can request itineraries via a simple JSON POST request.

---

## 🧱 Project Structure
├── app.py # Flask app exposing /generate API  
├── crew_engine.py # Main CrewAI multi-agent engine  
├── pyproject.toml # Dependency management  
└── templates/  
└── index.html  


---

## 📦 Installation & Setup

This project uses `pyproject.toml` for dependency management.

### 1. Create & Activate Virtual Environment

```bash
python3 -m venv venv
```

### 2. Install Dependencies (Using pyproject.toml)
```bash 
source venv/bin/activate
pip install .
```

### 3. Environment Variables

Create a file named .env in the project root:
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
SERPER_API_KEY="YOUR_SERPER_API_KEY"

## 🧠 How It Works (Core Logic Summary)
![User input](images\Inputs_vibe_journey.png "User Input")

## 🖥️ Running the Application

With your environment active:

```bash
python app.py
```
The local server starts at:
```
http://127.0.0.1:5000
```