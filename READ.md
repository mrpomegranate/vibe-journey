# Project Setup & Usage Guide

## 1. Prerequisites and Environment Setup

This project requires **Python 3.9+**.  
It is strongly recommended to use a **virtual environment (`venv`)** to isolate dependencies.

### A. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate


# API Key Configuration

The Agentic RAG system uses:

External LLMs for reasoning

Web APIs for real-time data retrieval

Create a file named .env in the root directory and include: