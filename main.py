from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import List, Optional
import logging
from huggingface_hub import InferenceClient
import json
import requests
from fastapi.middleware.cors import CORSMiddleware

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Get frontend URLs from environment variable and split into list
FRONTEND_URLS = os.getenv("FRONTEND_URLS", "http://localhost:3000").split(",")

app = FastAPI()

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"]
)

# Initialize the client
client = InferenceClient(
    model=os.getenv("ENDPOINT_URL"), 
    api_key=os.getenv("HUGGING_FACE_TOKEN")
)

# Core system message that defines the AI assistant's personality and expertise
SYSTEM_MESSAGE = """You are a helpful AI assistant specialized in hematology and oncology.
Speak like a human. Only provide detailed medical information when appropriate while otherwise being conversational.
You are an expert of Hemonc.org the largest freely available medical wiki of interventions, regimens, and general information relevant to the fields of hematology and oncology.
You provide accurate, evidence-based information while being clear and professional.
Be clear and detailed while also being concise."""

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

async def stream_processor(stream):
    """
    Processes the streaming response from the LLM and yields formatted SSE data.
    Each chunk is wrapped in a JSON structure for frontend consumption.
    """
    try:
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                yield f"data: {json.dumps({'content': content})}\n\n"
    except Exception as e:
        logger.error(f"Error in stream processing: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint that handles message streaming from the LLM.
    Includes system message for context and maintains conversation history.
    """
    try:
        if SYSTEM_MESSAGE:
            messages =  [{"role": "system", "content": SYSTEM_MESSAGE}] + \
                        [{"role": m.role, "content": m.content} for m in request.messages]
        else:
            messages = [{"role": m.role, "content": m.content} for m in request.messages]
            
        logger.info(f"Sending messages: {messages}")
        
        stream = client.chat.completions.create(
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            top_p=0.95,
            stream=True
        )
        
        return StreamingResponse(
            stream_processor(stream),
            media_type="text/event-stream",
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST',
            }
        )
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/model-status")
async def check_model_status():
    try:
        # Send a test chat completion request to start the model
        test_messages = [
            {"role": "user", "content": "test"}
        ]
        
        response = client.chat.completions.create(
            messages=test_messages,
            temperature=0.7,
            max_tokens=500,
            top_p=0.95,
            stream=False  # Don't stream for status check
        )
        
        return {"status": True, "estimated_time": "0"}
            
    except Exception as e:
        logger.error(f"Error checking model status: {str(e)}")
        return {
            "status": False,
            "estimated_time": "3-5 minutes"
        }
