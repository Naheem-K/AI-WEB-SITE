import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Gemini client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

client = genai.Client(api_key=api_key)

app = FastAPI(title="AI Song Recommendation API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class RecommendationRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    num_songs: int = Field(default=5, ge=1, le=10)
    include_reason: bool = Field(default=False)

class RecommendationResponse(BaseModel):
    recommendation: str
    songs: list[dict] = []
    status: str = "success"

# Serve frontend
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def home():
    """Serve the main frontend page"""
    return FileResponse("frontend/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "song-recommendation-api"}

@app.post("/api/ai/recommend", response_model=RecommendationResponse)
async def recommend(payload: RecommendationRequest):
    """
    Generate AI-powered song recommendations based on user input
    """
    try:
        # Build the prompt
        reason_instruction = ""
        if payload.include_reason:
            reason_instruction = "\nFor each song, include a brief reason (1 sentence) why it matches the request."
        
        prompt = f"""
You are a professional music recommendation engine with deep knowledge of all music genres, artists, and songs.

User request:
"{payload.question}"

Return exactly {payload.num_songs} song recommendations that best match this request.
{reason_instruction}

Format your response EXACTLY as follows:
1. Song Name – Artist{' – [Brief reason]' if payload.include_reason else ''}
2. Song Name – Artist{' – [Brief reason]' if payload.include_reason else ''}
... (continue for all {payload.num_songs} songs)

Important:
- Provide diverse recommendations across different eras/styles when appropriate
- Ensure all songs actually exist and artist names are correct
- Prioritize quality and relevance over obscurity
- Use the exact format specified above
"""

        # Call Gemini API
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        recommendation_text = response.text
        
        # Parse songs from response (basic parsing)
        songs = []
        lines = recommendation_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                song_text = line.lstrip('0123456789.-) ').strip()
                if '–' in song_text or '-' in song_text:
                    parts = song_text.replace('–', '-').split('-', 1)
                    if len(parts) == 2:
                        song_name = parts[0].strip()
                        rest = parts[1].strip()
                        
                        # Check if reason is included
                        if payload.include_reason and '–' in rest:
                            artist_reason = rest.split('–', 1)
                            artist = artist_reason[0].strip()
                            reason = artist_reason[1].strip() if len(artist_reason) > 1 else ""
                        else:
                            artist = rest
                            reason = ""
                        
                        songs.append({
                            "song": song_name,
                            "artist": artist,
                            "reason": reason if payload.include_reason else None
                        })
        
        logger.info(f"Generated {len(songs)} recommendations for query: {payload.question[:50]}...")
        
        return RecommendationResponse(
            recommendation=recommendation_text,
            songs=songs,
            status="success"
        )
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )

@app.post("/api/ai/song-info")
async def get_song_info(payload: dict):
    """
    Get detailed information about a specific song
    """
    song = payload.get("song")
    artist = payload.get("artist")
    
    if not song or not artist:
        raise HTTPException(status_code=400, detail="Song and artist are required")
    
    try:
        prompt = f"""
Provide detailed information about the song "{song}" by {artist}.

Include:
- Release year
- Album name
- Genre(s)
- Brief description (2-3 sentences about the song's style, themes, or significance)
- Similar songs (3 recommendations)

Format as a readable paragraph or structured text.
"""
        
        response = client.models.generate_content(
            model="gemini-2.5 -flash",
            contents=prompt
        )
        
        return {"info": response.text, "status": "success"}
    
    except Exception as e:
        logger.error(f"Error getting song info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get song information: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)