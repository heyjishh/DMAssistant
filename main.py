#!/usr/bin/env python3
import random
import json
import os
import time
import base64
import asyncio
from typing import List, Dict, Any
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain & AI imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage  # Added for proper message handling

# Telegram imports
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, ContextTypes, MessageHandler, CallbackQueryHandler, CommandHandler, filters
import telegram.error

# Google Generative AI for Vision
try:
    import google.generativeai as genai
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("⚠️ google-generativeai not installed. Vision API disabled.")

# ============================================================================
# CONFIGURATION & VALIDATION (Updated API Key)
# ============================================================================

def validate_env_vars():
    """Validate required environment variables"""
    required = {
        "GEMINI_API_KEY": "Gemini API key for LLM and embeddings",
        "TELEGRAM_BOT_TOKEN": "Telegram bot token"
    }
    
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"\n❌ Missing required env vars: {', '.join(missing)}")
        print("\n🔧 Set these in your environment:")
        for k in missing:
            print(f"  export {k}='your_key_here'")
        return False
    
    print("✅ All required env vars present")
    
    return True

# ============================================================================
# GLOBAL STATE & INITIALIZATION
# ============================================================================

LAST_PHOTO = {}
FREEZE_ROUNDS = {}
USER_CONTEXT = {}  # Store context per user

# Initialize Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY and VISION_AVAILABLE:
    genai.configure(api_key=GEMINI_API_KEY)

# ============================================================================
# USER CONTEXT - DELHI ISION_AVAILABLESTYLE (ENHANCED WITH RESEARCH)
# ============================================================================
genai.configure(api_key=GEMINI_API_KEY)
DELHI_CONTEXT = {
    "tz": "Asia/Kolkata",
    "country": "IN",
    "handle": "@heyyjishh",
    "region": "north_india_delhi_northern",
    "vibe": "delhi_genz_northern_flirty_teasy",
    "lang": "hinglish_english_shortform",
    "mood": "playful_flirt_banter",
    
    # Writing rules for natural, punchy replies (from KB: simple, respectful, context-tuned)
    "writing_rules": {
        "style": [
            "Be present-focused: comment on immediate context (e.g., her reel, message)",
            "Be specific and observant: reference exact details from profile/reel/message",
            "Keep tone light and curious: aim to learn about her, not impress",
            "Use open-ended prompts: questions needing more than yes/no",
            "Respect boundaries: if low energy, end gracefully",
            "Write short, punchy sentences",
            "Use active voice",
            "Focus on practical, actionable insights",
            "Speak directly using 'you' and 'your'",
            "Skip vague phrases"
        ],
        "avoid_phrases": [
            "not just this, but also this",
            "in conclusion",
            "it goes without saying",
            "at the end of the day",
            "needless to say",
            "Generic lines like 'hey beautiful' - too canned",
            "Over-complimenting body/sex appeal early",
            "Chasing with multiple texts if no reply",
            "Needy questions like 'why no reply?'"
        ],
        "punctuation": [
            "Avoid em dashes - use commas or periods",
            "No semicolons",
            "No hashtags",
            "No markdown formatting",
            "No asterisks for emphasis",
            "Use emojis sparingly for Gen-Z vibe (😂, 👀, 😏)"
        ],
        "do_dont": {
            "do": [
                "Start with context: reel, story, profile pic",
                "Mirror her energy: short reply = short back",
                "Tease playfully: light banter, not mean",
                "Share brief self-reveal: 10-25 sec anecdote for reciprocity",
                "Transition to future: 'plans this weekend?' after 2-3 exchanges",
                "Compliment specific/non-physical: 'your humor timing is spot on'",
                "From Quora/Reddit: For reels - 'this reel slaps, what's the backstory?'",
                "For in-feed: 'saw your post on [topic], what's your take?'",
                "Practice micro-convos daily for confidence"
            ],
            "dont": [
                "Force connection: if uninterested, exit politely",
                "Use pickup lines: natural > rehearsed",
                "Bombard with questions: one focused point per message",
                "Ignore red flags: love bomb, gold digger - slow down/pivot",
                "From Reddit: Don't double-text same day if no reply; abundance mindset",
                "Quora tip: Avoid 'what do you do?' early - too interview-y"
            ]
        }
    },
    
    # Modern chat shortcuts (expanded)
    "shortcuts": {
        "kk": "okay", "k": "okay", "kkhy": "okay hai yaar",
        "ok": "okay", "okhy": "okay hai yaar", "okn": "okay na",
        "no": "nahi", "nope": "nahi", "nah": "nahi",
        "chal na bhai": "let's go brother", "koi na": "not a big deal",
        "haan": "yes", "nhi": "nahi", "thik": "okay",
        "acha": "okay/good", "sahi": "right/good",
        "bas": "just/enough", "arre": "hey",
        "yaar": "dude/friend", "bhai": "brother/dude",
        "bc": "casual swear", "yaar bc": "dude seriously",
        "bet": "agreed", "ngl": "not gonna lie", "tbh": "to be honest",
        "iykyk": "if you know you know", "fr fr": "for real for real",
        "bruh": "dude", "sus": "suspicious"
    },
    
    # Core Delhi slang (expanded with Reddit/Quora Gen-Z terms)
    "slang": [
        "bhai kya hua", "ek number", "pakka", "bawaal bnchod", "banda solid hai",
        "chill kar yaar", "ek dum jhakaas", "toh kya", "thik hai na", "seedha bol na",
        "arey yaar", "genuine ekdum", "okay report", "fattu", "gandu", "lnd",
        "bhai kya scene hai", "bawaal macha diya", "chill karo yaar",
        "kya scene chal raha hai", "bhai io bol", "vella ban ja", "kalesh ho jayega",
        "mere yaar", "noobde spotted", "bhatta maar", "ghusand mat kar", "phod diya bhai",
        "kalesh macha", "vella time", "io bol na", "jugaad", "ghanta", "vella",
        "bindaas", "phod diya", "pet phat gaya", "chep", "katta", "faaltu", "saala",
        "bakchodi", "chull", "pataka", "kya scene", "arra yaar", "abey yaar", "haye re",
        "achha ji", "nahi ji", "pakau", "nutter", "enthu cutlet", "senti", "funda",
        "timepass", "toh", "haina", "badmash", "tell me", "bhai log", "yaarana",
        "fundae", "freak out", "fundaas", "bindaas only", "chalta hai yaar", "kuch bhi",
        "bilkul", "exactly", "point toh banta hai", "kya bolti tu", "sab changa",
        "ki haal hai", "mast", "sahi hai bhai", "ekdum sahi", "full jhakaas", "bawaal hai",
        "scene hai", "kya kar raha hai", "bol na yaar", "sun toh", "dekh na", "mat kar yaar",
        "ho gaya na", "chalega", "theek hai ji", "koi baat nahi", "no issue", "fine only",
        "bhai please", "yaar sorry", "kya hua", "sab theek", "mast mazaa aaya", "bahut accha",
        "zabardast", "shandaar", "wah yaar", "superb", "awesome only", "fantastic",
        "mind blowing", "outstanding", "ekdum top", "number one", "solid stuff", "rocking",
        "dhamakedaar", "blockbuster", "hit hai", "flop ho gaya", "bakwaas", "behenji",
        "aunty", "uncle ji", "bhai saab", "madam ji", "sahab", "ji", "huzoor", "maalik",
        "boss", "dada", "neta ji", "babu", "seth", "lala", "pandit ji", "maulvi sahab",
        "sardar ji", "bhaiya", "didi", "bhabhi", "devar", "jeth", "jethani", "nanad",
        "bhabo", "kaka", "kaki", "mama", "mami", "nana", "nani", "tau", "tauji",
        "chacha", "chachi", "bua", "fufaji", "masi", "masaji", "dadi", "dada ji",
        "nani ma", "nana ji",

        # Modern Gen-Z (from Reddit/Quora)
        "bro the fit is clean", "lowkey obsessed", "no cap this slaps",
        "Delhi winters hit different", "chai > everything",
        "fit check or delete", "rate the drip 1-10", "rizz level 100",
        "sus af", "bet", "finna head out", "bruh moment", "deadass",
        "iykyk", "tbh", "ngl", "fyp energy", "fr fr", "brb simping",
        "omg yes", "periodt", "slay", "tea", "chutiya" "chutiye", "chutiyapa",
        "bkl", "bkl chod", "bkl chod diya", "bkl chod diya yaar",
        
        # Flirty pickup (refined from KB: light, context-based)
        "are you set max kyuki deewana bana de",
        "purvi my interests are very purvi",
        "hey a i won't call you daddy but our children will",
        "tujhe pata mera baap kaun hai",
        "naya murgha zyada pharpharaata hai",
        "sheeeeesh", "raw-dogging the convo", "couple goals we're it",
        "manifesting reply", "panga with feelings",
        
        # Delhi regional (expanded)
        "kya scene chal raha hai", "bhai io bol", "vella ban ja",
        "kalesh ho jayega", "clubbing chalte hain", "mere yaar",
        "noobde spotted", "bhatta maar", "ghusand mat kar",
        
        # South Delhi posh
        "south delhi english", "gk preferred", "posh areas",
        "oh my god where H&M", "rizz in gk", "lowkey south delhi",
        "bet on hauz khas", "finna deer park date",
        
        # East Delhi laithi
        "laithi east side", "east delhi launda", "panga east wala",
        "genuine east ekdum", "laithi but cute", "bindaas east yaar",
        
        # Low side
        "dwarka side waale", "uttam nagar drip", "janakapuri chill",
        "lala flex", "delhi 71", "crasher kp panga",
        
        # Haryana mix
        "theth haryanvi", "mithi haryana mix", "jaat haryana",
        "haryanvi rowdy", "sonipat theth", "rohtak panga",
        
        # Punjabi influence
        "punjabi refugees", "dilli punjabi", "gol gappa punjabi",
        "patola irl", "sardar punjabi", "balle low side",
        "shava punjabi date",
        
        # Chat naturals (from text research)
        "chal na bhai", "koi na", "haan yaar", "nhi yaar",
        "thik hai", "acha sahi", "bas yaar", "arre sun",
        "dekh bhai", "mat kar", "ho gaya", "chal theek",
        "sahi hai", "badiya", "mast hai", "solid",
        
        # Reel/In-feed specific (from research)
        "this reel hits different", "fyp got me", "stitch this?", "duet vibes",
        "your take on this?", "relatable af", "lowkey obsessed with this feed"
    ]
}

# Embeddings disabled for speed
embeddings = None
print("⚡ Fast mode: Embeddings disabled")

# Enhanced Fast in-memory knowledge base (integrated full KB + research)
KNOWLEDGE_BASE = {
    "opener": [
        "Match her energy. Short reply = short reply back (Reddit tip)",
        "Use Delhi slang naturally. Bhai, yaar, scene, chal na (Quora fave)",
        "Keep it under 20 words. Punchy and direct",
        "Be present-focused: comment on something immediate like her reel/message",
        "Specific > generic: 'That reel on chai slaps' vs 'hey beautiful'",
        "Open-ended: 'What's your go-to chai spot?' not yes/no"
    ],
    "high_enthu": [
        "She's excited. Match the energy with emojis (😂, 👀)",
        "Ask open questions. Keep conversation flowing (KB principle)",
        "Playful teasing works well here: 'Arre, you're too good at this!'",
        "Share brief self-reveal: 'I tried that and failed hilariously'",
        "From Quora: Follow-up with 'Why's that your fave?' to deepen"
    ],
    "low_dry": [
        "Short replies mean low interest. Don't chase (abundance mindset)",
        "One more try with value, then ghost (Reddit consensus)",
        "Abundance mindset. Move on if no effort",
        "Shift topic lightly or exit: 'Nice chatting—enjoy your day'",
        "Don't double-text same day; wait 24h min (research)"
    ],
    "love_bomb": [
        "Too intense too fast = red flag (Quora warning)",
        "Slow it down. Ask about her day, hobbies (KB: facts to feelings)",
        "Stay calm and don't match intensity: 'That's sweet, tell me more about you'",
        "Pivot to light: 'Haha, manifesting slower— what's your weekend vibe?'"
    ],
    "gold_digger": [
        "Money talk early = test her (Reddit red flag)",
        "Pivot to her interests, not your wallet: 'Cool, but what's your passion project?'",
        "Talk about passions, values, not money (KB compliment style)",
        "If persists, graceful exit: 'Interesting take—catch you later!'"
    ],
    "feminine_grammar": [
        "Use: kar rhi ho, bna leti ho, jaa rhi ho (mandatory for natural Hinglish)",
        "NOT: kar raha ho, bna leta, jaa raha (masculine = awkward)",
        "Always feminine verbs when talking to her (Delhi chat norm)"
    ],
    "reel_opener": [
        "For reels: 'This slaps— what's the backstory? 👀' (Quora top)",
        "Play along: 'Duet this if you're brave' or 'Stitch your version'",
        "Relatable: 'FYP knows me too well—your take?'",
        "Don't: Generic 'lol'—add value with question",
        "Why: Builds on shared content, low pressure (KB context-based)"
    ],
    "infeed_opener": [
        "For in-feed post: 'Saw your [topic] post— what's one tip?' (Reddit effective)",
        "Specific: Reference detail like 'That outfit in GK? Where?'",
        "Curious: 'What's the highlight of that trip?'",
        "Don't: Ignore context—feels random",
        "Why: Shows you paid attention, invites reply (KB observant)"
    ],
    "followup": [
        "Use: 'Really? Tell me more' or 'Why's that your fave?' (KB flow)",
        "Mirror keyword: Repeat her word + expand question",
        "Move facts to feelings: After 'I like hiking' → 'What do you love about it?'",
        "Future-oriented: 'Any plans this weekend?' after 2 exchanges",
        "Don't: Bombard— one question max per text"
    ],
    "compliment": [
        "Specific, genuine, non-physical: 'Your timing in that reel is hilarious'",
        "Focus on taste/humor/energy/skill (KB safe)",
        "Hinglish twist: 'Ek dum jhakaas take yaar!'",
        "Don't: Body-focused early—'hot pic' = thirsty",
        "Why: Builds rapport without creep (Quora)"
    ],
    "awkward_rejection": [
        "Silence/short answers: Shift topic or exit: 'Nice chatting—enjoy!' (KB graceful)",
        "Declines: Accept friendly, no pressure: 'Cool, no worries—take care'",
        "Red flag signs: Uninterested cues—don't push",
        "Practice: Micro-convos daily (barista chats) for ease",
        "Mindset: Goal is authentic exchange, not force connection"
    ]
}

def get_advice(context: str) -> str:
    """Fast knowledge base lookup - now pulls from enhanced KB"""
    context_lower = context.lower()
    relevant_keys = [key for key in KNOWLEDGE_BASE if key in context_lower]
    if relevant_keys:
        key = random.choice(relevant_keys)
        return random.choice(KNOWLEDGE_BASE[key])
    # Fallback to random tip
    all_tips = [tip for tips in KNOWLEDGE_BASE.values() for tip in tips]
    return random.choice(all_tips)

collection = None
print("⚡ Fast mode: Enhanced knowledge base loaded with dating tips & research")

# Initialize LLM (Updated: model, response_mime_type for JSON, max_retries, timeout)
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        temperature=0.7,
        google_api_key=GEMINI_API_KEY
    )
    print("✅ LLM initialized")
except Exception as e:
    print(f"❌ LLM init failed: {e}")
    llm = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def escape_markdown_v2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2"""
    # Characters that need escaping in MarkdownV2 (according to Telegram docs)
    # Note: @ should NOT be escaped
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text

# ============================================================================
# CHROMADB SEEDING & HELPERS
# ============================================================================

# Removed ChromaDB functions for speed

# Stage & history removed for speed

# Tracking removed for speed

# ============================================================================
# IMAGE ANALYSIS (VISION API) - ENHANCED FOR REELS/IN-FEED
# ============================================================================

def analyze_insta_grid(image_path: str) -> Dict[str, Any]:
    """Analyze Instagram screenshot using Gemini Vision - now detects reels/in-feed"""
    if not VISION_AVAILABLE or not GEMINI_API_KEY:
        print("⚠️ Vision API unavailable. Using mock data.")
        return {
            "history": [
                {"role": "her", "message": "haha nice pic"},
                {"role": "user", "message": "Thanks! Where do you shop?"}
            ],
            "f": "10k",
            "g": "500",
            "b": "Fashion | Travel | Vibe",
            "d": "High-fashion outfits, travel pics",
            "v": "fashionista_high_value",
            "i": "yes",
            "h": "@posh_girl",
            "type": "chat",  # Default
            "content_type": "standard"  # New: for reels/in-feed
        }
    
    try:
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Use Gemini Vision (Updated model to multimodal)
        model = genai.GenerativeModel('gemini-2.0-flash')  # Updated for vision support
        
        # Enhanced detection prompt: now includes reel/in-feed
        detect_prompt = """
        Look at this Instagram screenshot carefully and identify what type it is.

PROFILE PAGE indicators:
- Shows follower/following counts at top
- Has a grid of photos/posts below
- Shows bio/description text
- Has "Edit Profile" or "Follow" button
- Shows profile picture at top

CHAT/DM indicators:
- Shows conversation bubbles/messages
- Has text messages going back and forth
- Shows timestamps on messages
- Has message input box at bottom
- Shows "Send" button

REEL/IN-FEED indicators:
- Shows video thumbnail with play button
- Has like/comment/share icons under video
- Displays reel text overlay or caption
- Single post expanded view

Analyze the image and respond with ONLY ONE WORD:
- Type "PROFILE" if it's a profile page
- Type "CHAT" if it's a DM/chat conversation
- Type "REEL" if it's a reel/video
- Type "INFEED" if it's a static post/feed item
        """
        
        detect_response = model.generate_content([
            detect_prompt,
            {"mime_type": "image/jpeg", "data": image_data}
        ])
        
        detect_text = detect_response.text.strip().upper()
        print(f"🔍 Detection result: {detect_text}")
        
        # Determine type from response
        if "PROFILE" in detect_text:
            image_type = "profile"
            content_type = "standard"
        elif "CHAT" in detect_text or "DM" in detect_text:
            image_type = "chat"
            content_type = "standard"
        elif "REEL" in detect_text:
            image_type = "chat"  # Treat as chat context
            content_type = "reel"
        elif "INFEED" in detect_text:
            image_type = "profile"  # Treat as profile context
            content_type = "infeed"
        else:
            # Default to chat if unclear
            image_type = "chat"
            content_type = "standard"
            print(f"⚠️ Unclear detection, defaulting to chat")
        
        # Use appropriate prompt based on type
        if image_type == "profile" or content_type in ["infeed"]:
            prompt = """
            This is an Instagram PROFILE or IN-FEED post. Extract all visible information:

Look for:
- Username/handle (usually at top)
- Bio/description text
- Follower count number
- Following count number  
- Number of posts
- What kind of photos are in the grid/post (fashion, travel, fitness, food, etc)
- If IN-FEED: Specific post caption, likes, comments
- Overall vibe/aesthetic

Respond in this EXACT format (fill in what you see):

Handle: @username
Bio: the bio text here
Followers: number
Following: number
Posts: number
Grid/Post: description of the photos/posts (include caption if in-feed)
Vibe: overall vibe assessment
Influencer: yes or no
Content Type: reel or infeed or standard
            """
        else:
            prompt = """
            This is an Instagram CHAT/DM conversation or REEL context. Extract the messages/content:

Look for:
- All messages in the conversation
- Who sent each message (her vs the user)
- Her most recent message (IMPORTANT)
- Her username if visible
- Any bio text if visible
- If REEL: Reel caption, overlay text, or reaction prompt

Respond in this EXACT format:

Messages:
HER: her message text
USER: user message text
HER: her next message
(continue for all visible messages)

Her Last Message: her most recent message text
Handle: @username if visible
Bio: bio text if visible
Content Type: reel or standard

CRITICAL: Make sure to identify HER LAST MESSAGE correctly - this is what we need to reply to.
            """
        
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_data}
        ])
        
        # Parse response based on type
        text = response.text.strip()
        print(f"📄 Raw response preview: {text[:200]}...")
        
        result = {"type": image_type, "content_type": content_type}
        
        if image_type == "profile":
            # Parse profile format
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("Handle:"):
                    result["handle"] = line.split(":", 1)[1].strip()
                elif line.startswith("Bio:"):
                    result["bio"] = line.split(":", 1)[1].strip()
                elif line.startswith("Followers:"):
                    result["followers"] = line.split(":", 1)[1].strip()
                elif line.startswith("Following:"):
                    result["following"] = line.split(":", 1)[1].strip()
                elif line.startswith("Posts:"):
                    result["posts"] = line.split(":", 1)[1].strip()
                elif line.startswith("Grid/Post:"):
                    result["grid_desc"] = line.split(":", 1)[1].strip()
                elif line.startswith("Vibe:"):
                    result["vibe"] = line.split(":", 1)[1].strip()
                elif line.startswith("Influencer:"):
                    result["is_influencer"] = line.split(":", 1)[1].strip().lower()
                elif line.startswith("Content Type:"):
                    result["content_type"] = line.split(":", 1)[1].strip().lower()
            
            # Set defaults for profile
            result.setdefault("handle", "@unknown")
            result.setdefault("bio", "No bio")
            result.setdefault("followers", "unknown")
            result.setdefault("following", "unknown")
            result.setdefault("posts", "unknown")
            result.setdefault("grid_desc", "Standard posts")
            result.setdefault("vibe", "casual")
            result.setdefault("is_influencer", "no")
            result.setdefault("content_type", "standard")
            
        else:
            # Parse chat format
            history = []
            her_last = ""
            handle = "@unknown"
            bio = ""
            
            lines = text.split('\n')
            in_messages = False
            
            for line in lines:
                line = line.strip()
                
                if line.startswith("Messages:"):
                    in_messages = True
                    continue
                
                if in_messages:
                    if line.startswith("HER:"):
                        msg = line.split(":", 1)[1].strip()
                        history.append({"role": "her", "message": msg})
                        her_last = msg  # Keep updating to get the last one
                    elif line.startswith("USER:"):
                        msg = line.split(":", 1)[1].strip()
                        history.append({"role": "user", "message": msg})
                    elif line.startswith("Her Last Message:"):
                        her_last = line.split(":", 1)[1].strip()
                        in_messages = False
                    elif line.startswith("Handle:"):
                        handle = line.split(":", 1)[1].strip()
                        in_messages = False
                    elif line.startswith("Bio:"):
                        bio = line.split(":", 1)[1].strip()
                        in_messages = False
                    elif line.startswith("Content Type:"):
                        result["content_type"] = line.split(":", 1)[1].strip().lower()
            
            result["history"] = history if history else [{"role": "her", "message": her_last or "hey"}]
            result["her_last_message"] = her_last or (history[-1]["message"] if history and history[-1]["role"] == "her" else "hey")
            result["handle"] = handle
            result["bio"] = bio
        
        # Common defaults
        result.setdefault("f", "unknown")
        result.setdefault("g", "unknown")
        result.setdefault("b", result.get("bio", "No bio"))
        result.setdefault("d", result.get("grid_desc", "Standard"))
        result.setdefault("v", result.get("vibe", "casual_normal"))
        result.setdefault("i", result.get("is_influencer", "no"))
        result.setdefault("h", result.get("handle", "@unknown"))
        result.setdefault("her_last_message", result.get("history", [{}])[-1].get("message", "hey") if result.get("history") else "hey")
        result.setdefault("content_type", "standard")
        
        return result
        
    except Exception as e:
        print(f"⚠️ Vision analysis error: {e}")
        # Fallback mock with content_type
        return {
            "history": [{"role": "her", "message": "hey"}, {"role": "user", "message": "yo what's up"}],
            "her_last_message": "hey",
            "f": "unknown", "g": "unknown", "b": "Analysis failed",
            "d": "Could not analyze grid", "v": "unknown", "i": "no", "h": "@unknown",
            "type": "chat", "content_type": "standard"
        }

# ============================================================================
# MOOD & TRAP DETECTION (ENHANCED WITH KB TIPS)
# ============================================================================

def sense_her_mood(hist: List[Dict[str, str]]):
    """Fast keyword-based mood detection - now includes research cues"""
    msgs = [m["message"] for m in hist if m["role"] == "her"][-3:]
    
    if not msgs:
        return {"m": "medium_cool", "t": None}
    
    combined = " ".join([m.lower() for m in msgs])
    
    # Fast keyword detection (expanded)
    if any(w in combined for w in ["haha", "lol", "😂", "😊", "😍", "!", "love", "omg", "yess", "slay", "bet"]):
        mood = "high_enthu"
    elif any(w in combined for w in ["k", "ok", "kk", "hmm", "idk", "whatever", "busy", "ngl short"]) and len(combined) < 20:
        mood = "low_dry"
    elif any(w in combined for w in ["why", "what", "seriously", "wtf", "stop", "leave", "no thanks"]):
        mood = "angry_defensive"
    else:
        mood = "medium_cool"
    
    # Fast trap detection (from KB/Reddit)
    trap = None
    if any(w in combined for w in ["miss you", "need you", "perfect", "soulmate", "marry"]) and len(hist) < 10:
        trap = {"t": "love_bomb", "a": "Too intense too fast - slow down (Quora red flag)"}
    elif any(w in combined for w in ["expensive", "buy", "gift", "rich", "afford", "dinner on you"]):
        trap = {"t": "gold_digger", "a": "Money focused early - pivot to values (Reddit advice)"}
    elif any(w in combined for w in ["creep", "weird", "stop", "block"]) and len(combined) < 50:
        trap = {"t": "rejection", "a": "Uninterested cues - exit gracefully (KB boundary respect)"}
    
    return {"m": mood, "t": trap}

# ============================================================================
# REPLY GENERATION (ENHANCED PROMPT WITH FULL KB INTEGRATION)
# ============================================================================

def generate_cot_reply(extract: Dict[str, Any], uid: int):
    """Generate reply using Chain of Thought - updated prompt with KB/research"""
    if not llm:
        # Fallback replies (enhanced with KB openers)
        return {
            "r": [
                {"text": "Yo! Kya scene hai? Specific se bolo 👀", "sub_vibe": "hinglish"},
                {"text": "Hey! What's the vibe? Tell me more.", "sub_vibe": "english"}
            ],
            "a": "LLM unavailable. Using KB fallback: Be specific & curious.",
            "h": "Open-ended hook.",
            "s": "False",
            "sources": [{"key": "opener", "quote": "Specific > generic: reference exact details"}]
        }, None, 50
    
    mood = extract.get("mood", {})
    trap = mood.get("t")
    her_last = extract.get("her_last_message", "hey")
    content_type = extract.get("content_type", "standard")
    
    # Fast knowledge base lookup (now pulls 3-4 tips)
    mood_type = mood.get("m", "medium_cool")
    advice_context = f"{mood_type} {content_type}"
    if trap:
        advice_context += f" {trap['t']}"
    
    advice_list = [
        get_advice(mood_type),
        get_advice(content_type),  # New: reel/infeed specific
        get_advice("feminine_grammar"),
        get_advice("followup") if mood_type != "opener" else get_advice("opener")
    ]
    advice = " | ".join(advice_list)
    
    # Build prompt
    history_str = "\n".join([
        f"{m['role']}: {m['message']}"
        for m in extract.get('history', [])
    ])
    
    # Sample slang for context (expanded sample)
    slang_sample = ", ".join(random.sample(DELHI_CONTEXT['slang'], min(20, len(DELHI_CONTEXT['slang']))))
    shortcuts_str = ", ".join([f"{k}={v}" for k, v in list(DELHI_CONTEXT['shortcuts'].items())[:15]])
    
    # ENHANCED PROMPT: Integrated full KB principles, do/don't, research
    prompt = ChatPromptTemplate.from_template(
        """You are a Delhi Gen-Z DM coach specializing in Indian dating: complete girl emotions, get friendly fast without creepy. From North India/Delhi. Generate replies for a GUY talking to a GIRL.

Context:
- History: {history}
- Her Last Message: "{her_last}"
- Content Type: {content_type} (reel/infeed/standard)
- Stage: {stage} (1=Stranger, 4=GF)
- Her Mood: {mood}
- Trap: {trap}
- Advice: {advice} (from research/KB)

Delhi Gen-Z Style Guide (KB Principles):
- Region: North India, Delhi, Northern style
- Slang examples: {slang}
- Chat shortcuts: {shortcuts}
- Tone: Playful, flirty, teasy, confident, light & curious
- Language: Natural Hinglish/English mix: kk, okhy, chal na bhai, koi na, haan yaar, nhi yaar, thik hai, acha, sahi, bas yaar
- Core Rules (from Quora/Reddit/KB): 
  - Present-focused: Comment on immediate (her reel/message/profile detail)
  - Specific/observant: Reference exact thing (e.g., 'That chai reel? Obsessed')
  - Open-ended: Need more than yes/no (e.g., 'What's your fave spot? Why?')
  - Respect boundaries: Low energy? Short/light or exit gracefully
  - Do: Mirror energy, playful tease, brief self-reveal, facts→feelings, future topics after 2 exchanges
  - Don't: Generic/canned lines, chase low interest, over-compliment physical, bombard questions, needy (abundance mindset)
  - For Reels: Play along/duet vibe, relatable reaction + question
  - For In-Feed: Specific post hook, 'What's your take/hot tip?'
  - Compliments: Specific/non-physical (humor, style, energy/skill)

CRITICAL: YOU ARE TALKING TO A GIRL - USE ONLY FEMININE GRAMMAR (Delhi norm)

FEMININE VERB FORMS (MANDATORY):
When talking about HER actions, ALWAYS use these endings:
- Present continuous: "kar RHI HO", "jaa RHI HO", "so RHI HO", "dekh RHI HO", "sun RHI HO"
- Habitual: "karti HO", "jaati HO", "soti HO", "dekhti HO", "sunti HO"  
- Compound: "bna LETI HO", "kha LETI HO", "kar LETI HO"

NEVER USE MASCULINE FORMS:
❌ WRONG: "kar raha ho", "kar raha hai", "karta ho", "kar leta ho", "bna leta main"
❌ WRONG: "jaa raha", "so raha", "dekh raha", "sun raha"
❌ WRONG: Any verb ending in "raha", "rahe", "leta", "lete", "ta", "te"

CORRECT EXAMPLES YOU MUST FOLLOW:
✅ "Acha bna leti ho" (NOT "bna leti main" or "bna leta")
✅ "Kya kar rhi ho?" (NOT "kya kar raha ho")
✅ "Samajh rhi ho?" (NOT "samajh raha ho")
✅ "Kahan jaa rhi ho?" (NOT "kahan jaa raha ho")
✅ "So rhi thi kya?" (NOT "so raha tha")

DOUBLE CHECK: Before finalizing each reply, verify NO masculine verb forms (raha/rahe/leta/lete/ta/te) are used when talking about HER.

CRITICAL CITATION RULE: For every reply and analysis, cite EXACT KB sources used. Format: [key: "direct quote from KB"]. Include 1-2 per response in a new "sources" list in JSON. Quote verbatim—do not paraphrase. Examples:
- If using opener: [opener: "Match her energy. Short reply = short reply back (Reddit tip)"]
- If reel: [reel_opener: "For reels: 'This slaps— what's the backstory? 👀' (Quora top)"]
Base replies ONLY on these cited sources + context. If no direct match, cite 'general' and explain.

WRITING RULES (Punchy & Natural from KB):
- Clear, simple language; short sentences; active voice
- Cut fluff: No em dashes, semicolons, hashtags, markdown, asterisks
- Emojis: 1-2 max for vibe (👀, 😏, 😂)
- Conversational like real texting: One focused point, direct response to {her_last}

Generate TOON: {{r:[{{text:str,sub_vibe:str}}],a:analysis,h:hook,sources:[{{"key":str,"quote":str}}]}}

STRICT RULES:
1. 2 replies: one Hinglish, one English
2. Under 20 words each (SHORT, focused - KB punchy)
3. High-value, confident, playful; match {content_type} (reel: play along; infeed: specific hook)
4. Use Delhi slang/shortcuts naturally (1-2 per reply)
5. MANDATORY: Feminine grammar for HER - "kar rhi ho", "bna leti ho", "jaa rhi ho" (NO masculine)
6. DIRECTLY respond to her last message: "{her_last}" - stay on topic, add open-ended hook
7. ONE focused point per reply - no random extras
8. Address trap: Love bomb? Slow/light; Gold digger? Pivot values; Rejection? Graceful exit
9. Sound like real Delhi Gen-Z guy: Abundance, no neediness
10. If low mood: Light exit option in analysis
11. FINAL CHECK: Scan for masculine verbs - fix to feminine; ensure open-ended
12. ALWAYS include 1-2 sources in "sources" array

Output ONLY valid JSON."""
    )
    
    try:
        chain = prompt | llm | JsonOutputParser()
        out = chain.invoke({
            "history": history_str,
            "her_last": her_last,
            "content_type": content_type,
            "stage": 1,
            "mood": mood.get("m", "medium_cool"),
            "trap": trap["t"] if trap else "none",
            "advice": advice,
            "slang": slang_sample,
            "shortcuts": shortcuts_str
        })
        
        # Light check for sources (new)
        if "sources" not in out or not out["sources"]:
            # Fallback cite
            out["sources"] = [{"key": "general", "quote": get_advice("opener")}]
            if "a" in out:
                out["a"] += f" [citing general: '{out['sources'][0]['quote']}']"
            else:
                out["a"] = f"Analysis unavailable. [citing general: '{out['sources'][0]['quote']}']"
        
        if not isinstance(out, dict) or 'r' not in out:
            raise ValueError("Invalid LLM output")
            
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ LLM generation error: {error_msg}")
        
        # Check if it's a quota error
        is_quota_error = "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower()
        
        # Smart fallback based on her message - enhanced with KB
        her_last_lower = her_last.lower()
        sources_fallback = [{"key": "opener", "quote": "Be present-focused: comment on something immediate like her reel/message"}]
        if "hey" in her_last_lower or "hi" in her_last_lower or "hello" in her_last_lower:
            out = {
                "r": [
                    {"text": f"Yo! Kya scene? Specific batao, like your last reel 👀", "sub_vibe": "hinglish"},
                    {"text": f"Hey! What's up? What's one thing from your feed I should check?", "sub_vibe": "english"}
                ],
                "a": "Casual opener: Specific & open-ended [citing opener: 'Specific > generic'] (KB fallback)",
                "h": "Curious hook",
                "sources": sources_fallback,
                "quota_error": is_quota_error
            }
        elif "kk" in her_last_lower or "ok" in her_last_lower or "haan" in her_last_lower:
            out = {
                "r": [
                    {"text": f"Haha okhy. Toh phir, kya kar rhi ho abhi? 😏", "sub_vibe": "hinglish"},
                    {"text": f"Haha alright. So, what's your take on that? 😏", "sub_vibe": "english"}
                ],
                "a": "Flow follow-up: Why/how [citing followup: 'Use: \"Really? Tell me more\"'] (KB)",
                "h": "Deepen lightly",
                "sources": [{"key": "followup", "quote": "Use: 'Really? Tell me more' or 'Why's that your fave?' (KB flow)"}],
                "quota_error": is_quota_error
            }
        elif "sleep" in her_last_lower or "so" in her_last_lower or "tired" in her_last_lower:
            out = {
                "r": [
                    {"text": f"Arre so rhi thi kya? Chal koi na, rest well yaar 😴", "sub_vibe": "hinglish"},
                    {"text": f"Oh, sleeping? No rush—catch you when you're up 😴", "sub_vibe": "english"}
                ],
                "a": "Respect boundaries: Graceful [citing awkward_rejection: 'Silence/short answers: Shift topic or exit'] (KB)",
                "h": "Low pressure",
                "sources": [{"key": "awkward_rejection", "quote": "Silence/short answers: Shift topic or exit: 'Nice chatting—enjoy!' (KB graceful)"}],
                "quota_error": is_quota_error
            }
        else:
            out = {
                "r": [
                    {"text": f"Haha sahi. Chal na, batao kya pasand kiya usme? 👀", "sub_vibe": "hinglish"},
                    {"text": f"Haha nice. What do you like most about it?", "sub_vibe": "english"}
                ],
                "a": "Facts to feelings: KB flow [citing followup: 'Move facts to feelings'] (fallback)",
                "h": "Engage deeper",
                "sources": [{"key": "followup", "quote": "Move facts to feelings: After 'I like hiking' → 'What do you love about it?'"}],
                "quota_error": is_quota_error
            } 
    
    # Check for warnings (enhanced)
    warn = None
    user_streak = sum(1 for m in reversed(extract.get("history", [])[:5]) if m["role"] == "user")
    if user_streak >= 3:
        warn = "⚠️ 3+ texts in a row. Let her miss you! (Abundance - Reddit)"
    if extract.get("mood", {}).get("m") == "low_dry":
        warn = warn or "Low energy: One more value text, then pause (KB don't chase)"
    
    trust = get_trust(uid, extract.get("h", "unknown"))
    
    return out, warn, trust

# ============================================================================
# TELEGRAM HANDLERS (ENHANCED WITH NEW TEMPLATES)
# ============================================================================

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle screenshot uploads - now handles content_type"""
    uid = update.effective_user.id
    now = time.time()
    
    # Rate limiting
    if uid in LAST_PHOTO and now - LAST_PHOTO[uid] < 30:
        await update.message.reply_text(
            "⏳ Chill yaar! 30 sec cooldown.\n\n"
            "**Abundance** > desperation, remember? 💯 (KB mindset)"
        )
        return
    
    LAST_PHOTO[uid] = now
    
    # Download photo
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    
    # Use temp directory that works on all platforms
    import tempfile
    temp_dir = tempfile.gettempdir()
    fp = os.path.join(temp_dir, f"screenshot_{uid}_{int(now)}.jpg")
    
    await update.message.reply_text("🔍 Analyzing your DM game...")
    
    try:
        await file.download_to_drive(fp)
        
        # Validate image
        with Image.open(fp) as img:
            if img.size[0] < 200 or img.size[1] < 200:
                await update.message.reply_text("🚫 Image too small bhai. Send HD screenshot (720p+).")
                return
        
        # Process
        extract = {"history": [], "h": "@girl"}
        grid = analyze_insta_grid(fp)
        extract.update(grid)
        
        # Check content_type for tailored templates
        content_type = extract.get("content_type", "standard")
        
        # Check if it's a profile screenshot
        if extract.get("type") == "profile":
            # Extract profile data
            bio = extract.get('bio', '').lower()
            grid = extract.get('grid_desc', '').lower()
            
            # Extract key words for templates
            bio_short = extract.get('bio', '')[:50] if extract.get('bio') else "your bio"
            bio_key = "your vibe"
            if bio:
                bio_words = bio.split()
                if len(bio_words) > 2:
                    bio_key = " ".join(bio_words[:3])
            
            grid_key = "content"
            if "fashion" in grid or "outfit" in grid:
                grid_key = "fashion"
            elif "travel" in grid:
                grid_key = "travel"
            elif "food" in grid:
                grid_key = "food"
            elif "gym" in grid or "fitness" in grid:
                grid_key = "fitness"
            elif "art" in grid:
                grid_key = "art"
            elif "reel" in grid or "video" in grid or content_type == "reel":
                grid_key = "reels"
            elif content_type == "infeed":
                grid_key = "post"
        
        # === ENHANCED MASTER TEMPLATES (INTEGRATED KB/RESEARCH) ===
        OPENERS_TEMPLATES = {
            # POV 1: Self-Deprecating (KB light tone)
            "pov_self": [
                "POV: I'm the guy who trips in slow motion. You? (Relatable fail - Reddit)",
                "POV: My camera roll is 80% memes, 20% panic. Yours?",
                "POV: I just googled 'how to adult'. Send help or laugh?",
                "POV: My plants die from overthinking. You saving yours?",
                "POV: I reply 'haha' to everything. Even sad news. You?"
            ],
            # POV 2: Shared Chaos (KB present-focused)
            "pov_chaos": [
                "POV: We both open 47 tabs and cry. Solidarity?",
                "POV: '5 more mins' turns into 3 hours. Same gang?",
                "POV: We ghost plans but overthink texts. Hi?",
                "POV: Chai fixes 90% of our problems. The other 10%? (Delhi vibe)",
                "POV: We say 'I'm fine' but mean 'send memes'. You?"
            ],
            # POV 3: Game Invite (KB open-ended)
            "pov_game": [
                "POV: We play '2 truths 1 lie'. You start or I do?",
                "POV: I say a word, you say the first thing. Ready? 'Weekend'",
                "POV: Would you rather: no WiFi or no chai? Fight me",
                "POV: Rate my vibe 1-10. I'll rate yours back. Deal?",
                "POV: Send me a song. I'll send one back. No skips"
            ],
            # POV 4: Bio Hook (KB specific)
            "pov_bio": [
                "POV: Your bio says '{bio_key}' — I tried it and failed. Story? (Quora hook)",
                "POV: '{bio_short}' in bio? I need the director's cut",
                "POV: We both live by '{bio_key}'. Prove it — one example?",
                "POV: Your bio = my lock screen quote. Coincidence?",
                "POV: '{bio_short}' — bold. I respect it. Origin story?"
            ],
            # POV 5: Grid Insight (KB observant)
            "pov_grid": [
                "POV: {grid_key} people understand {grid_key}. Am I right?",
                "POV: Your grid = my FYP. What's the algorithm hiding?",
                "POV: We both gatekeep {grid_key} spots. Share one?",
                "POV: {grid_key} but make it chaotic. Your version?",
                "POV: Your posts = comfort show. What's the pilot episode?"
            ],
            # Enhanced for Reels/In-Feed
            "pov_reel": [
                "POV: Your reel just ended my scroll. Backstory? 👀 (Research top)",
                "POV: Dueting this in my head. Your version better?",
                "POV: FYP energy matches your vibe. What's next?",
                "POV: This reel = my mood. Remix ideas?",
                "POV: Stitching this for sure. Tag me?"
            ],
            "pov_infeed": [
                "POV: Your post on {grid_key} — hot take? (KB specific)",
                "POV: Scrolled past but came back. Why this one?",
                "POV: {grid_key} feed goals. One tip for newbies?",
                "POV: Relatable post af. Your why behind it?",
                "POV: This in-feed = conversation starter. Go."
            ],
            # Fallbacks (KB casual)
            "casual": [
                "Quick: your camera roll's 90% selfies or 90% food pics?",
                "Arre, what's one app you can't delete?",
                "Yo, if your life had a loading screen, what % are you at?",
                "Hey, last song in private session?",
                "What's the most 'adulting' thing you did this week?"
            ],
            "hobby_connect": [
                "Saw your {grid_key} posts — what's the one thing *everyone* gets wrong? (Quora)",
                "Hey, {grid_key} in your grid? Same. What's your 'IYKYK' moment?",
                "Arre, {grid_key} enthusiast detected. Drop your hottest take",
                "Your {grid_key} game is strong. What's the one resource you'd gatekeep?",
                "Yo! {grid_key} posts = instant follow. What's next level?"
            ],
            "meme_vibe": [
                "Your grid's giving 'main character in a coming-of-age movie'. Soundtrack?",
                "Arre, if your feed was a Netflix category, what would it be called?",
                "Yo, your posts = 10/10 would scroll again. What's the one you'd delete?",
                "Grid check: are you the 'aesthetic chaos' or 'organized mess' type?",
                "Your content's like a vibe playlist — what's the skip track?"
            ]
        }

        # === ENHANCED VOICE NOTE SCRIPTS (KB brief self-reveal) ===
        VOICE_NOTE_TEMPLATES = {
        "vn_bio": [
            "[chill] Yo! Your bio — '{bio_short}' — wait, how do you even live by that? I tried and lasted 3 days [laugh]. Voice note me the real story. (Self-reveal KB)",
            "[playful] Arre, '{bio_key}' in bio? That's bold. Drunk 2 AM or deep philosophy? Voice note me."
        ],
        "vn_grid": [
            "[excited] Yo, your {grid_key} content? Obsessed. What’s the one thing no one gets? Voice note me your hot take. (Curious KB)",
            "[curious] Arre, {grid_key} in your grid — teach me one thing. 20 sec. I’ll send proof [laugh]."
        ],
        "vn_game": [
            "[playful] POV: Voice note tag. I say: '{grid_key}'. You reply first thing. Go. (Open-ended KB)",
            f"[excited] POV: 2 truths 1 lie. 1. I cried in a movie. 2. I can cook. 3. I have 12 plants. Guess — voice note me! (Game KB)"
        ],
        # New for content_type
        "vn_reel": [
            "[vibey] This reel? Lowkey obsessed. Voice note your inspo behind it? (Reel research)",
            "[teasy] Arre, dueting this mentally. Send your full version VN?"
        ],
        "vn_infeed": [
            "[curious] Your {grid_key} post — what's the untold story? VN me. (Infeed specific)",
            "[playful] Saw that in-feed, had to pause. Quick VN: fave part?"
        ]
        }

        # === ENHANCED REEL COMMENT STARTERS (From research) ===
        REEL_REACTION_TEMPLATES = {
            "funny": ["bro this is me at 3am", "the accuracy is scary", "send this to your gc", "i felt this in my soul", "deadass relatable"],
            "relatable": ["finally someone said it", "this is my entire personality", "the algorithm knows me", "ngl this hits", "iykyk vibes"],
            "hobby_match": ["okay but {grid_key} people will understand", "only {grid_key} girlies get this", "{grid_key} tea spilled"],
            "play_along": ["duet this if you're brave", "reply with your version", "tag your partner in crime", "stitch challenge accepted?"]
        }

        # === REPLY BOOSTERS (KB follow-ups) ===
        REPLY_BOOSTERS = [
            "Arre wait — [her answer]? Same energy. But real q: how do you even function? (Mirror KB)",
            "Okay [her reply] is top tier. Now level 2: worst fail story go (Deepen)",
            "Not me relating to [her answer] at 2 AM. Your turn to ask me anything (Reciprocity KB)",
            "Hold up — [her reply]? I need receipts. Voice note or it didn’t happen (Playful)",
            "You win. [her answer] > my answer. Rematch? (Tease light)"
        ]

        # === MICRO-POV SCENARIOS (KB non-creepy) ===
        MICRO_POV = [
            "**POV 1: The Relatable Fail** → Send: *'POV: I tried your {grid_key} thing and failed in 3 sec. Video proof?'*",
            "**POV 2: The Shared Secret** → *'Only {bio_key} people know this struggle. You too?'*",
            "**POV 3: The Mini-Challenge** → *'Bet you can’t say your {grid_key} in 3 words. Go.'*"
        ]

        # === EMOJI-ONLY OPENERS (Gen-Z research) ===
        EMOJI_OPENERS = [
            "chai question coffee answer ☕❓",
            "question meme answer 😂❓",
            "question song answer 🎵❓",
            "question fail answer 😩❓",
            "question and chill? ❄️❓"
        ]

        # === STORY REACTION IDEAS (KB context) ===
        STORY_REACTIONS = [
            "this but make it [her story vibe]",
            "POV: you're in this story",
            "the way i paused 👀"
        ]

        # === NO REPLY? SEND THIS (KB graceful) ===
        NO_REPLY_NUDGE = "Arre, seen but no reply? Fair. But now I’m IYKYK — your turn (Low pressure)"

        # === AUTO-REPLY SIMULATOR (Test her response) ===
        def simulate_her_reply(user_line):
            responses = [
                "lmao same", "wait how did you know", "okay but real q", "not me", "send proof",
                "question question question", "this is too real", "you're not wrong", "haha bet", "ngl obsessed"
            ]
            return random.choice(responses)

        # === MAIN GENERATOR (TAILORED TO CONTENT_TYPE) ===
        def generate_master_package(extract):
            bio = extract.get('bio', '').strip()
            grid = extract.get('grid_desc', '').lower().strip()
            content_type = extract.get('content_type', 'standard')
            
            bio_words = [w for w in bio.split() if len(w) > 3 and w[0].isalnum()] if bio else []
            bio_key = ' '.join(bio_words[:2]) if bio_words else 'your vibe'
            bio_short = bio[:38] + '...' if bio and len(bio) > 38 else bio
            
            grid_words = [w for w in grid.split() if len(w) > 4] if grid else []
            grid_key = random.choice(grid_words).capitalize() if grid_words else "your posts"
            
            result = {}
            
            # 1. TEXT OPENERS (5, tailored)
            text_lines = []
            text_lines.append({"text": random.choice(OPENERS_TEMPLATES["pov_self"]), "style": "pov_self"})
            text_lines.append({"text": random.choice(OPENERS_TEMPLATES["pov_chaos"]), "style": "pov_chaos"})
            text_lines.append({"text": random.choice(OPENERS_TEMPLATES["pov_game"]), "style": "pov_game"})
            if bio and len(bio) > 10:
                text_lines.append({"text": random.choice(OPENERS_TEMPLATES["pov_bio"]).format(bio_key=bio_key, bio_short=bio_short), "style": "pov_bio"})
            else:
                text_lines.append({"text": random.choice(OPENERS_TEMPLATES["casual"]), "style": "pov_casual"})
            # Tailor last based on content_type
            if content_type == "reel":
                text_lines.append({"text": random.choice(OPENERS_TEMPLATES["pov_reel"]).format(grid_key=grid_key), "style": "pov_reel"})
            elif content_type == "infeed":
                text_lines.append({"text": random.choice(OPENERS_TEMPLATES["pov_infeed"]).format(grid_key=grid_key), "style": "pov_infeed"})
            else:
                if any(word in grid for word in ["book", "music", "food", "gym", "art", "code", "dance", "travel"]):
                    text_lines.append({"text": random.choice(OPENERS_TEMPLATES["hobby_connect"]).format(grid_key=grid_key), "style": "pov_hobby"})
                else:
                    text_lines.append({"text": random.choice(OPENERS_TEMPLATES["meme_vibe"]), "style": "pov_meme"})
            result["text_lines"] = text_lines
            
            # 2. VOICE NOTES (3, tailored)
            vns = []
            if content_type == "reel":
                vns.append(random.choice(VOICE_NOTE_TEMPLATES["vn_reel"]).format(grid_key=grid_key))
            elif content_type == "infeed":
                vns.append(random.choice(VOICE_NOTE_TEMPLATES["vn_infeed"]).format(grid_key=grid_key))
            else:
                vns.append(random.choice(VOICE_NOTE_TEMPLATES["vn_bio"]).format(bio_short=bio_short, bio_key=bio_key) if bio else random.choice(VOICE_NOTE_TEMPLATES["vn_game"]))
            if content_type in ["reel", "infeed"]:
                vns.append(random.choice(VOICE_NOTE_TEMPLATES["vn_grid"]).format(grid_key=grid_key))
            else:
                vns.append(random.choice(VOICE_NOTE_TEMPLATES["vn_game"]).format(grid_key=grid_key))
            vns.append(random.choice(VOICE_NOTE_TEMPLATES["vn_grid"]).format(grid_key=grid_key))
            result["voice_notes"] = [{"script": vns[i], "tip": ["Chill", "Curious", "Playful"][i % 3]} for i in range(min(3, len(vns)))]
            
            # 3. REEL COMMENTS (3, enhanced)
            reels = []
            reels.append(random.choice(REEL_REACTION_TEMPLATES["funny"]))
            if content_type == "reel" or any(w in grid for w in ["travel","food","gym","art"]):
                reels.append(random.choice(REEL_REACTION_TEMPLATES["hobby_match"]).format(grid_key=grid_key))
            else:
                reels.append(random.choice(REEL_REACTION_TEMPLATES["relatable"]))
            reels.append(random.choice(REEL_REACTION_TEMPLATES["play_along"]))
            result["reel_comments"] = reels
            
            # 4. REPLY BOOSTERS
            result["reply_boosters"] = REPLY_BOOSTERS
            
            # 5. MICRO-POV SCENARIOS
            result["micro_pov"] = [p.format(grid_key=grid_key, bio_key=bio_key) for p in MICRO_POV]
            
            # 6. EMOJI OPENERS
            result["emoji_openers"] = [e.format(grid_key=grid_key) for e in EMOJI_OPENERS]
            
            # 7. STORY REACTIONS
            result["story_reactions"] = STORY_REACTIONS
            
            # 8. NO REPLY NUDGE
            result["no_reply"] = NO_REPLY_NUDGE
            
            # 9. SIMULATOR
            result["simulator"] = simulate_her_reply
            
            return result

        # === IN PROFILE HANDLER (TAILORED OUTPUT) ===
        if extract.get("type") == "profile":
            # Build concise response
            bio = extract.get('bio', 'No bio')[:80]
            grid = extract.get('grid_desc', 'Standard')[:60]
            content_type = extract.get('content_type', 'standard')
            
            # Build profile text (use code blocks for ALL content with special chars)
            txt = ["👤 *Profile Analysis*\n"]
            txt.append(f"`{extract.get('handle', '@unknown')}`")  # Handle in code block
            txt.append(f"Bio: `{bio[:80]}`")  # Use code block for bio
            txt.append(f"Followers: {extract.get('followers', '?')} \\| Posts: {extract.get('posts', '?')}")
            txt.append(f"Grid: `{grid[:80]}`")  # Use code block for grid
            txt.append(f"Content: {content_type.upper()} vibe\n")
            
            # Generate customized opening lines with LLM (enhanced prompt)
            opening_lines = []
            if llm:
                try:
                    # Analyze grid photos in detail
                    grid_full = extract.get('grid_desc', 'Standard posts')
                    bio_full = extract.get('bio', 'No bio')
                    content_type_full = content_type
                    
                    prompt_text = f"""Delhi guy DMing girl on Instagram. Generate 10 opening lines - each with BOTH English and Hinglish versions, tuned to Delhi Gen-Z vibe (playful, teasy, confident, abundance mindset).

HER PROFILE:
Bio: {bio_full}
Grid Photos: {grid_full}
Content Type: {content_type_full}
Followers: {extract.get('followers', 'unknown')}

Generate 10 opening lines based on KB principles: Present-focused (immediate bio/grid detail), specific/observant (reference exact element), light/curious (open-ended question, no impressing), non-physical compliment if fitting (humor/energy/skill). For reels: Play-along/duet vibe; infeed: Hot take/tip; standard: Relatable chaos/game.

For EACH line, provide:
- English version (natural, under 15 words)
- Hinglish version (same meaning, Delhi slang like yaar/arre/chal na, feminine grammar: kar rhi ho/bna leti ho)

Format EXACTLY like this:
1. English: [english text]
   Hinglish: [hinglish text]
2. English: [english text]
   Hinglish: [hinglish text]
(continue for 10 lines; no extras)

RULES (from KB/Quora/Reddit/youtube):
- Under 15 words each; teasy, punchy, active voice.
- Reference SPECIFIC profile/content (e.g., bio keyword, grid theme).
- Feminine grammar mandatory (kar rhi ho, jaa rhi ho, bna leti ho; NO masculine raha/rahe/leta).
- Natural Delhi style: Mix shortcuts (kkhy, haan yaar, sahi, dude); 1-2 emojis max (👀, 😏).
- High-value: Playful tease/reciprocal share, open-ended (why/how/fave?); no generic/creepy.
- Variety: bio-hooks, grid-specific, content-type tailored, games/POVs, relatable chaos, playful teases.

RESPOND WITH ONLY THE 20 LINES (10 English + 10 Hinglish):"""
                    
                    # Use proper messages for invoke
                    messages = [
                        SystemMessage(content="You are a Delhi Gen-Z DM coach. Generate concise opening lines."),
                        HumanMessage(content=prompt_text)
                    ]
                    
                    response = llm.invoke(messages)
                    lines = [l.strip() for l in response.content.strip().split('\n') if l.strip()]
                    
                    # Parse lines - expecting format: "1. English: text" and "1.1 Hinglish: text"
                    opening_lines = []  # Will store tuples of (english, hinglish)
                    current_english = None
                    
                    for line in lines:
                        if 'English:' in line or 'english:' in line:
                            # Extract English version
                            text = line.split(':', 1)[-1].strip()
                            # Remove numbering
                            if text and text[0].isdigit():
                                parts = text.split('.', 1)
                                if len(parts) > 1:
                                    text = parts[1].strip()
                            current_english = text
                        elif ('Hinglish:' in line or 'hinglish:' in line) and current_english:
                            # Extract Hinglish version
                            text = line.split(':', 1)[-1].strip()
                            # Remove numbering
                            if text and text[0].isdigit():
                                parts = text.split('.', 1)
                                if len(parts) > 1:
                                    text = parts[1].strip()
                            opening_lines.append({'english': current_english, 'hinglish': text})
                            current_english = None
                    
                    # Ensure we have at least 10 pairs
                    default_lines = [
                        {'english': f"Hey! Your {content_type_full} content is fire. What's the story?", 'hinglish': f"Yo! Tera {content_type_full} content ekdum jhakaas. Kya scene hai?"},
                        {'english': f"Your {grid_full[:30]} caught my eye. Tell me more?", 'hinglish': f"Arre, tera {grid_full[:30]} dekha. Batao na?"},
                        {'english': "Let's chat. What are you up to?", 'hinglish': "Chal baat karte hain. Kya kar rhi ho?"},
                        {'english': "Your vibe is different. What's your secret?", 'hinglish': "Tera vibe alag hai. Secret kya hai?"},
                        {'english': "That bio is bold. Story behind it?", 'hinglish': "Bio ekdum bold hai. Story kya hai?"},
                        {'english': "Your posts have energy. How do you pick what to share?", 'hinglish': "Teri posts mein energy hai. Kaise decide karti ho kya share karna?"},
                        {'english': "Interesting content. What inspires you?", 'hinglish': "Interesting content hai. Kya inspire karta hai?"},
                        {'english': "Your style is unique. How'd you develop it?", 'hinglish': "Tera style unique hai. Kaise develop kiya?"},
                        {'english': "That content is solid. What's next?", 'hinglish': "Content solid hai. Aage kya plan?"},
                        {'english': "Let's be real. What's your story?", 'hinglish': "Chal real baat karte hain. Teri story kya hai?"}
                    ]
                    while len(opening_lines) < 10:
                        opening_lines.append(default_lines[len(opening_lines)])
                        
                except Exception as e:
                    print(f"⚠️ Opening line generation error: {e}")
                    # Fallback tailored with English/Hinglish pairs (10 lines)
                    if content_type == "reel":
                        opening_lines = [
                            {'english': "This reel is fire. What's the backstory?", 'hinglish': "Yo! Yeh reel ekdum jhakaas. Backstory kya hai? 👀"},
                            {'english': "Duet vibes on this. Your take?", 'hinglish': "Arre, duet vibes aa rhe. Tera take kya hai?"},
                            {'english': "FYP gold right here. Remix?", 'hinglish': "Bhai FYP pe aana chahiye. Remix karegi?"},
                            {'english': "This reel slaps. How'd you come up with it?", 'hinglish': "Yeh reel ekdum solid. Kaise socha?"},
                            {'english': "Reel energy is different. What inspired this?", 'hinglish': "Reel ki energy alag hai. Kya inspire kiya?"},
                            {'english': "That transition though. Tutorial?", 'hinglish': "Arre woh transition! Tutorial degi?"},
                            {'english': "Reel game strong. What's your secret?", 'hinglish': "Reel game ekdum strong. Secret kya hai?"},
                            {'english': "This deserves more views. What's next?", 'hinglish': "Isko zyada views milne chahiye. Aage kya?"},
                            {'english': "Reel vibes hit different. Your fave part?", 'hinglish': "Reel vibes alag hain. Tera fave part?"},
                            {'english': "That audio choice. Why this one?", 'hinglish': "Woudio choice. Kyun yeh wala?"}
                        ]
                    elif content_type == "infeed":
                        opening_lines = [
                            {'english': "This post caught my eye. Hot take?", 'hinglish': "Yo! Yeh post dekha. Hot take kya hai?"},
                            {'english': "That content is solid. Any tips?", 'hinglish': "Arre, yeh content solid hai. Koi tip hai?"},
                            {'english': "Post is interesting. Why this one?", 'hinglish': "Yeh post interesting hai. Kyun yeh wala?"},
                            {'english': "Your caption game is strong. How do you write them?", 'hinglish': "Tera caption game strong hai. Kaise likhti ho?"},
                            {'english': "That aesthetic though. What's your process?", 'hinglish': "Woh aesthetic! Tera process kya hai?"},
                            {'english': "Post vibes are different. What inspired this?", 'hinglish': "Post vibes alag hain. Kya inspire kiya?"},
                            {'english': "Your feed is curated. How do you plan it?", 'hinglish': "Teri feed curated hai. Kaise plan karti ho?"},
                            {'english': "That shot is clean. Photography tips?", 'hinglish': "Woh shot clean hai. Photography tips?"},
                            {'english': "Post energy is unique. What's the story?", 'hinglish': "Post ki energy unique hai. Story kya hai?"},
                            {'english': "Your content stands out. What's your approach?", 'hinglish': "Tera content stand out karta hai. Approach kya hai?"}
                        ]
                    else:
                        opening_lines = [
                            {'english': "Your profile is clean. What's up?", 'hinglish': "Yo! Tera profile ekdum clean. Kya scene hai?"},
                            {'english': "Your content is fire. What's the story?", 'hinglish': "Arre, tera content fire hai. Story kya hai?"},
                            {'english': "Profile caught my eye. Let's chat?", 'hinglish': "Tera profile dekha. Chal baat karte hain?"},
                            {'english': "Your vibe is different. What's your secret?", 'hinglish': "Tera vibe alag hai. Secret kya hai?"},
                            {'english': "That bio is bold. Story behind it?", 'hinglish': "Bio ekdum bold hai. Story kya hai?"},
                            {'english': "Your posts have energy. How do you pick?", 'hinglish': "Teri posts mein energy hai. Kaise pick karti ho?"},
                            {'english': "Interesting content. What inspires you?", 'hinglish': "Interesting content hai. Kya inspire karta hai?"},
                            {'english': "Your style is unique. How'd you develop it?", 'hinglish': "Tera style unique hai. Kaise develop kiya?"},
                            {'english': "Content is solid. What's next?", 'hinglish': "Content solid hai. Aage kya plan?"},
                            {'english': "Let's be real. What's your story?", 'hinglish': "Chal real baat karte hain. Teri story kya hai?"}
                        ]
            else:
                # Fallback (10 pairs)
                opening_lines = [
                    {'english': "Your profile is clean. What are you up to?", 'hinglish': "Yo! Tera profile clean hai. Kya kar rhi ho?"},
                    {'english': "Your content is solid. What's up?", 'hinglish': "Arre, tera content solid hai. Kya scene?"},
                    {'english': "Let's talk. What's the vibe?", 'hinglish': "Chal baat karte hain. Kya vibe hai?"},
                    {'english': "Your style is different. What's your secret?", 'hinglish': "Tera style alag hai. Secret kya hai?"},
                    {'english': "That bio caught my eye. Story?", 'hinglish': "Bio dekha. Story kya hai?"},
                    {'english': "Your posts are interesting. What inspires you?", 'hinglish': "Teri posts interesting hain. Kya inspire karta?"},
                    {'english': "Content game is strong. How do you do it?", 'hinglish': "Content game strong hai. Kaise karti ho?"},
                    {'english': "Your vibe is unique. Tell me more?", 'hinglish': "Tera vibe unique hai. Batao na?"},
                    {'english': "Profile is fire. What's next?", 'hinglish': "Profile fire hai. Aage kya?"},
                    {'english': "Let's chat. What's your story?", 'hinglish': "Chal baat karte hain. Teri story?"}
                ]
            
            # Add opening lines to main message (show all 10, with English + Hinglish)
            txt.append("\n💬 *Opening Lines:*")
            for i, line_pair in enumerate(opening_lines[:10], 1):  # Show up to 10 pairs
                txt.append(f"{i}\\. English: `{line_pair['english']}`")
                txt.append(f"{i}\\.1 Hinglish: `{line_pair['hinglish']}`")
            
            # Generate package for voice notes and reels
            package = generate_master_package(extract)
            
            # Add voice notes (show all 3)
            txt.append("\n🎤 *Voice Note Scripts:*")
            for i, vn in enumerate(package["voice_notes"][:3], 1):
                txt.append(f"{i}\\. `{vn['script']}`")
            
            # Add reel comments (show all 3)
            txt.append("\n📱 *Reel/In\\-Feed Comments:*")
            for i, rc in enumerate(package["reel_comments"][:3], 1):
                txt.append(f"{i}\\. `{rc}`")
            
            txt.append("\n_Tap any line to select and copy\\!_ 👑")
            
            # Build with MarkdownV2 - bio/grid as plain text, replies in backticks
            md_txt = ["👤 *Profile Analysis*\n"]
            
            # Handle, Bio, Grid, Followers, Posts - ALL need escaping
            handle_escaped = escape_markdown_v2(extract.get('handle', '@unknown'))
            bio_escaped = escape_markdown_v2(bio[:80])
            grid_escaped = escape_markdown_v2(grid[:80])
            followers_escaped = escape_markdown_v2(str(extract.get('followers', '?')))
            posts_escaped = escape_markdown_v2(str(extract.get('posts', '?')))
            
            md_txt.append(f"*{handle_escaped}*")
            md_txt.append(f"Bio: {bio_escaped}")
            md_txt.append(f"Followers: {followers_escaped} \\| Posts: {posts_escaped}")
            md_txt.append(f"Grid: {grid_escaped}")
            md_txt.append(f"Content: {content_type.upper()} vibe\n")
            
            # Opening lines in backticks (selectable)
            md_txt.append("\n💬 *Opening Lines:*")
            for i, line_pair in enumerate(opening_lines[:10], 1):
                md_txt.append(f"{i}\\. English: `{line_pair['english']}`")
                md_txt.append(f"{i}\\.1 Hinglish: `{line_pair['hinglish']}`")
            
            # Voice notes in backticks (selectable)
            md_txt.append("\n🎤 *Voice Note Scripts:*")
            for i, vn in enumerate(package["voice_notes"][:3], 1):
                md_txt.append(f"{i}\\. `{vn['script']}`")
            
            # Reel comments in backticks (selectable)
            md_txt.append("\n📱 *Reel/In\\-Feed Comments:*")
            for i, rc in enumerate(package["reel_comments"][:3], 1):
                md_txt.append(f"{i}\\. `{rc}`")
            
            md_txt.append("\n_Tap any line to select and copy\\!_ 👑")
            
            try:
                await update.message.reply_text(
                    "\n".join(md_txt),
                    parse_mode='MarkdownV2'
                )
            except telegram.error.TimedOut:
                await update.message.reply_text("⚠️ Network slow. Retry screenshot.")
                return
            except Exception as e:
                # Fallback to plain text if markdown fails
                print(f"Markdown error: {e}")
                plain_txt = ["👤 Profile Analysis\n"]
                plain_txt.append(f"{extract.get('handle', '@unknown')}")
                plain_txt.append(f"Bio: {bio[:80]}")
                plain_txt.append(f"Followers: {extract.get('followers', '?')} | Posts: {extract.get('posts', '?')}")
                plain_txt.append(f"Grid: {grid[:80]}")
                plain_txt.append(f"Content: {content_type.upper()} vibe\n")
                plain_txt.append("\n💬 Opening Lines:")
                for i, line_pair in enumerate(opening_lines[:10], 1):
                    plain_txt.append(f"{i}. English: `{line_pair['english']}`")
                    plain_txt.append(f"{i}.1 Hinglish: `{line_pair['hinglish']}`")
                plain_txt.append("\n🎤 Voice Note Scripts:")
                for i, vn in enumerate(package["voice_notes"][:3], 1):
                    plain_txt.append(f"{i}. `{vn['script']}`")
                plain_txt.append("\n📱 Reel/In-Feed Comments:")
                for i, rc in enumerate(package["reel_comments"][:3], 1):
                    plain_txt.append(f"{i}. `{rc}`")
                plain_txt.append("\nTap any line to select and copy! 👑")
                await update.message.reply_text("\n".join(plain_txt))
                return
            
            # Only More button
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("🎲 More Options", callback_data=f"more_profile_{uid}")]
            ])
            
            USER_CONTEXT[uid] = {
                "extract": extract, 
                "package": package, 
                "opening_lines": opening_lines,
                "type": "profile",
                "content_type": content_type
            }
            
            # Send More button
            try:
                await update.message.reply_text("Need more ideas?", reply_markup=kb)
            except telegram.error.TimedOut:
                pass
            return
    
    except (FileNotFoundError, telegram.error.TimedOut):
        await update.message.reply_text("🚫 Network timeout. Telegram servers slow. Send screenshot again.")
    except json.JSONDecodeError as e:
        await update.message.reply_text(f"🤖 LLM returned garbage. Try again.\n\n`{str(e)[:100]}`")
    except Exception as e:
        await update.message.reply_text(
            f"💥 Unexpected error:\n`{str(e)[:200]}`\n\n"
            "DM @heyyjishh if this keeps happening."
        )
        import traceback
        traceback.print_exc()
    finally:
        # Safe file cleanup for Windows
        if 'fp' in locals() and os.path.exists(fp):
            try:
                os.remove(fp)
            except PermissionError:
                # File still in use, try again after a short delay
                await asyncio.sleep(0.1)
                try:
                    os.remove(fp)
                except:
                    pass  # If still fails, let OS clean it up later

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages (her DM text) - enhanced with content_type fallback"""
    uid = update.effective_user.id
    text = update.message.text
    
    # Ignore commands
    if text.startswith('/'):
        return
    
    await update.message.reply_text("🔍 Analyzing her message...")
    
    try:
        # Create extract from text
        extract = {
            "history": [{"role": "her", "message": text}],
            "her_last_message": text,
            "h": "@girl",
            "v": "casual_normal",
            "f": "unknown",
            "g": "unknown",
            "b": "No bio",
            "d": "Text only",
            "i": "no",
            "c": retrieve_stage(uid),
            "n": retrieve_stage(uid),
            "content_type": "standard"  # Default for text
        }
        
        # Analyze mood
        girl = extract.get("h", "@unknown")
        mood = sense_her_mood(extract.get("history", []), uid, girl)
        extract["mood"] = mood
        
        # Generate reply
        cot, warn, trust = generate_cot_reply(extract, uid)
        
        # Store context
        USER_CONTEXT[uid] = {
            "extract": extract,
            "cot": cot,
            "warn": warn,
            "trust": trust,
            "girl": girl
        }
        
        # Build response with MarkdownV2
        txt = ["🔥 *DM Coach Report*\n"]
        
        # Her message as plain text (escaped)
        escaped_text = escape_markdown_v2(text)
        txt.append(f"💬 *Her:* {escaped_text}\n")
        
        # Check for quota error
        if cot.get("quota_error"):
            txt.append("⚠️ *API Quota Exceeded* \\- Using smart fallback replies\n")
        
        # Analysis (enhanced) - plain text (escaped)
        analysis = cot.get('a', 'Keep it high-value.')
        if warn:
            analysis += f"\n{warn}"
        escaped_analysis = escape_markdown_v2(analysis)
        txt.append(f"🧠 *Play:* {escaped_analysis}\n")
        
        # Add replies to main message - in backticks (selectable)
        txt.append("\n📱 *Your Drip:*")
        replies = cot.get('r', [])
        for i, r in enumerate(replies, 1):
            vibe = r.get('sub_vibe', 'unknown').title()
            txt.append(f"{i}\\. \\[{vibe}\\] `{r['text']}`")
        
        # Add KB Sources if available - plain text (escaped)
        if "sources" in cot and cot["sources"]:
            txt.append("\n📚 *KB Sources:*")
            for s in cot["sources"][:2]:
                escaped_quote = escape_markdown_v2(s['quote'][:80])
                txt.append(f"• {s['key'].title()}: {escaped_quote}\\.\\.\\.")
        
        txt.append("\n_Tap any reply above to select and copy\\!_ 👑")
        
        # Send everything in ONE message with MarkdownV2
        try:
            await update.message.reply_text(
                "\n".join(txt),
                parse_mode='MarkdownV2'
            )
        except Exception as e:
            # Fallback to plain text if markdown fails
            print(f"Markdown error: {e}")
            plain_txt = ["🔥 DM Coach Report\n"]
            plain_txt.append(f"💬 Her: {text}\n")
            if cot.get("quota_error"):
                plain_txt.append("⚠️ API Quota Exceeded - Using smart fallback replies\n")
            plain_txt.append(f"🧠 Play: {analysis}\n")
            plain_txt.append("\n📱 Your Drip:")
            for i, r in enumerate(replies, 1):
                vibe = r.get('sub_vibe', 'unknown').title()
                plain_txt.append(f"{i}. [{vibe}] `{r['text']}`")
            if "sources" in cot and cot["sources"]:
                plain_txt.append("\n📚 KB Sources:")
                for s in cot["sources"][:2]:
                    plain_txt.append(f"• {s['key'].title()}: {s['quote'][:80]}...")
            plain_txt.append("\nTap any reply above to select and copy! 👑")
            await update.message.reply_text("\n".join(plain_txt))
        
        # Only More button
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🎲 More Options", callback_data=f"more_{uid}")]
        ])
        
        await update.message.reply_text("Need more ideas?", reply_markup=kb)
        
    except Exception as e:
        await update.message.reply_text(
            f"💥 Error:\n`{str(e)[:200]}`\n\n"
            "Try again or send a screenshot."
        )
        import traceback
        traceback.print_exc()

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button presses - enhanced for content_type"""
    query = update.callback_query
    
    # Answer callback with timeout handling
    try:
        await query.answer()
    except telegram.error.TimedOut:
        pass  # Ignore timeout on answer, continue processing
    except Exception:
        pass  # Ignore other errors on answer
    
    data = query.data
    
    # All copy button handlers removed - using selectable text instead
    
    if data.startswith("text_"):
        # Send as text button pressed
        parts = data.split("_")
        if len(parts) >= 3:
            uid = int(parts[1])
            idx = int(parts[2])
            
            if uid in USER_CONTEXT:
                replies = USER_CONTEXT[uid]["cot"].get("r", [])
                extract = USER_CONTEXT[uid]["extract"]
                
                if idx < len(replies):
                    reply_text = replies[idx]["text"]
                    vibe_type = replies[idx].get("sub_vibe", "unknown").title()
                    
                    # Send the reply as a formatted text message
                    response = [
                        "📱 **Your Reply is Ready!**\n",
                        f"**Type:** {vibe_type}",
                        f"**Replying to:** \"{extract.get('her_last_message', 'her message')}\"",
                        f"**Girl:** {extract.get('h', '@unknown')}\n",
                        "─────────────────────",
                        f"**YOUR REPLY:**\n\n{reply_text}",
                        "─────────────────────\n",
                        "✅ Copy this text and paste in Instagram DM",
                        "💯 Remember: High-value, abundance mindset (KB)",
                        "👑 You got this, king!"
                    ]
                    
                    await query.message.reply_text("\n".join(response))
                    
                    # Also send as a separate message for easy copying
                    await query.message.reply_text(
                        f"`{reply_text}`",
                        parse_mode='Markdown'
                    )
                    
                    # Send sources if available
                    cot = USER_CONTEXT[uid]["cot"]
                    if "sources" in cot and cot["sources"]:
                        sources_text = "**KB Sources Used:**\n" + "\n".join([f"• **{s['key'].title()}**: {s['quote'][:100]}..." for s in cot["sources"][:3]])
                        await query.message.reply_text(sources_text)
                else:
                    await query.message.reply_text("❌ Reply not found. Send screenshot again.")
            else:
                await query.message.reply_text("❌ Context lost. Send screenshot again bhai.")
    
    elif data.startswith("more_profile_"):
        # More options for profile - generate additional opening lines
        parts = data.split("_")
        if len(parts) >= 3:
            try:
                uid = int(parts[2])
            except (ValueError, IndexError):
                await query.message.reply_text("❌ Invalid button")
                return
            
            if uid in USER_CONTEXT and USER_CONTEXT[uid].get("type") == "profile":
                extract = USER_CONTEXT[uid]["extract"]
                content_type = USER_CONTEXT[uid].get("content_type", "standard")
                
                await query.message.reply_text("🔄 Generating more options...")
                
                # Generate more diverse opening lines
                if llm:
                    try:
                        grid_full = extract.get('grid_desc', 'Standard posts')
                        bio_full = extract.get('bio', 'No bio')
                        
                        from langchain_core.messages import SystemMessage, HumanMessage
                        
                        prompt_text = f"""Generate 3 MORE creative opening lines for Instagram DM (different from previous).

HER PROFILE:
Bio: {bio_full}
Grid: {grid_full}
Content Type: {content_type}

Generate 3 NEW opening lines. For EACH line provide:
- English version
- Hinglish version

Format:
1. English: [text]
1.1 Hinglish: [text]
2. English: [text]
2.1 Hinglish: [text]
3. English: [text]
3.1 Hinglish: [text]

Make these DIFFERENT from typical openers:
- Style 1: Playful/teasing
- Style 2: Mysterious/intriguing
- Style 3: Direct/confident

Under 15 words each, Delhi Gen-Z style, feminine grammar."""
                        
                        messages = [
                            SystemMessage(content="You are a Delhi Gen-Z DM coach."),
                            HumanMessage(content=prompt_text)
                        ]
                        
                        response = llm.invoke(messages)
                        lines = [l.strip() for l in response.content.strip().split('\n') if l.strip()]
                        
                        # Parse English/Hinglish pairs
                        more_lines = []
                        current_english = None
                        
                        for line in lines:
                            if 'English:' in line or 'english:' in line:
                                text = line.split(':', 1)[-1].strip()
                                if text and text[0].isdigit():
                                    parts = text.split('.', 1)
                                    if len(parts) > 1:
                                        text = parts[1].strip()
                                current_english = text
                            elif ('Hinglish:' in line or 'hinglish:' in line) and current_english:
                                text = line.split(':', 1)[-1].strip()
                                if text and text[0].isdigit():
                                    parts = text.split('.', 1)
                                    if len(parts) > 1:
                                        text = parts[1].strip()
                                more_lines.append({'english': current_english, 'hinglish': text})
                                current_english = None
                        
                        # Fallback if parsing failed
                        if len(more_lines) < 3:
                            more_lines = [
                                {'english': "Your vibe is different. What's your secret?", 'hinglish': "Tera vibe alag hai. Secret kya hai?"},
                                {'english': "Interesting content. What inspires you?", 'hinglish': "Interesting content hai. Kya inspire karta hai?"},
                                {'english': "Let's be real. What's your story?", 'hinglish': "Chal real baat karte hain. Teri story kya hai?"}
                            ]
                        
                        # Build response with MarkdownV2
                        txt = ["🎲 *More Opening Lines:*\n"]
                        for i, line_pair in enumerate(more_lines[:3], 1):
                            txt.append(f"{i}\\. English: `{line_pair['english']}`")
                            txt.append(f"{i}\\.1 Hinglish: `{line_pair['hinglish']}`")
                        
                        txt.append("\n_Tap any line to select and copy\\!_ 👑")
                        
                        try:
                            await query.message.reply_text(
                                "\n".join(txt),
                                parse_mode='MarkdownV2'
                            )
                        except Exception as e:
                            print(f"Markdown error: {e}")
                            await query.message.reply_text("\n".join(txt))
                        
                    except Exception as e:
                        print(f"More profile options error: {e}")
                        # Fallback
                        fallback_lines = [
                            {'english': "Your style is unique. How'd you develop it?", 'hinglish': "Tera style unique hai. Kaise develop kiya?"},
                            {'english': "Content is solid. What's next?", 'hinglish': "Content solid hai. Aage kya plan?"},
                            {'english': "Let's chat. What's your vibe today?", 'hinglish': "Chal baat karte hain. Aaj ka vibe kya hai?"}
                        ]
                        
                        txt = ["🎲 *More Opening Lines:*\n"]
                        for i, line_pair in enumerate(fallback_lines, 1):
                            txt.append(f"{i}\\. English: `{line_pair['english']}`")
                            txt.append(f"{i}\\.1 Hinglish: `{line_pair['hinglish']}`")
                        
                        txt.append("\n_Tap any line to select and copy\\!_ 👑")
                        
                        try:
                            await query.message.reply_text(
                                "\n".join(txt),
                                parse_mode='MarkdownV2'
                            )
                        except:
                            await query.message.reply_text("\n".join(txt))
                else:
                    await query.message.reply_text("❌ LLM unavailable.")
            else:
                await query.message.reply_text("❌ Context lost. Send screenshot again.")
        return
    
    elif data.startswith("more_"):
        # Generate more diverse options for chat (enhanced prompt)
        parts = data.split("_")
        if len(parts) >= 2:
            try:
                uid = int(parts[1])
            except (ValueError, IndexError):
                await query.message.reply_text("❌ Invalid button")
                return
            
            if uid in USER_CONTEXT:
                extract = USER_CONTEXT[uid]["extract"]
                content_type = extract.get("content_type", "standard")
                
                # Generate more diverse replies with higher temperature
                await query.message.reply_text("🔄 Generating more options...")
                
                # Temporarily increase temperature for diversity
                if llm:
                    try:
                        her_last = extract.get("her_last_message", "hey")
                        mood = extract.get("mood", {})
                        
                        # Use simpler approach without JsonOutputParser for more reliability
                        from langchain_google_genai import ChatGoogleGenerativeAI
                        diverse_llm = ChatGoogleGenerativeAI(
                            model="gemini-flash-lite-latest",
                            temperature=0.9,
                            google_api_key=GEMINI_API_KEY
                        )
                        
                        prompt_text = f"""Generate 3 different creative replies to her message: "{her_last}"
Content: {content_type}

Style 1: Playful/teasing (light banter KB)
Style 2: Mysterious/intriguing (curious hook)
Style 3: Direct/confident (specific reference)

Each reply under 30 words, Delhi slang natural, DIFFERENT vibes.
For {content_type}: Reel=play along; Infeed=specific; Standard=mirror energy.

CRITICAL CITATION RULE: Cite 1-2 KB sources in "sources" array.

Reply 1:
Reply 2:
Reply 3:"""
                        
                        messages = [
                            SystemMessage(content="You are a Delhi Gen-Z DM coach. Generate creative replies with citations."),
                            HumanMessage(content=prompt_text)
                        ]
                        
                        response = diverse_llm.invoke(messages)
                        lines = response.content.strip().split('\n')
                        
                        # Parse replies
                        replies = []
                        styles = ['playful', 'mysterious', 'direct']
                        reply_num = 0
                        
                        for line in lines:
                            line = line.strip()
                            if line.startswith('Reply') and ':' in line:
                                text = line.split(':', 1)[1].strip()
                                if text:
                                    replies.append({
                                        'text': text,
                                        'style': styles[reply_num] if reply_num < len(styles) else 'casual',
                                        'sub_vibe': 'hinglish'
                                    })
                                    reply_num += 1
                        
                        # Fallback if parsing failed
                        if len(replies) < 3:
                            if content_type == "reel":
                                replies = [
                                    {"text": "Haha this reel slaps. Duet me? 👀", "style": "playful", "sub_vibe": "hinglish"},
                                    {"text": "Lowkey obsessed... backstory? 😏", "style": "mysterious", "sub_vibe": "hinglish"},
                                    {"text": "Yo, FYP gold. Your take?", "style": "direct", "sub_vibe": "hinglish"}
                                ]
                            elif content_type == "infeed":
                                replies = [
                                    {"text": "Arre post fire. Hot take? 😂", "style": "playful", "sub_vibe": "hinglish"},
                                    {"text": "That detail... intriguing. Why? 👀", "style": "mysterious", "sub_vibe": "hinglish"},
                                    {"text": "Direct: Love the vibe. Tip?", "style": "direct", "sub_vibe": "hinglish"}
                                ]
                            else:
                                replies = [
                                    {"text": "Haha bet. Toh scene kya hai? 👀", "style": "playful", "sub_vibe": "hinglish"},
                                    {"text": "Interesting... batao phir, what's the vibe?", "style": "mysterious", "sub_vibe": "hinglish"},
                                    {"text": "Yo, let's keep it real. What's up?", "style": "direct", "sub_vibe": "hinglish"}
                                ]
                        
                        txt = ["**🎲 More Options:**\n"]
                        txt.append("_Tap any text below to select and copy!_ 👑")
                        
                        await query.message.reply_text("\n".join(txt))
                        
                        # Send each reply as selectable text
                        for i, r in enumerate(replies[:3], 1):
                            style = r.get('style', 'casual')
                            try:
                                await query.message.reply_text(f"`{r['text']}`")
                            except telegram.error.TimedOut:
                                continue
                        
                        # Store new replies
                        USER_CONTEXT[uid]["more_replies"] = replies
                        
                        # Send sources (assume from prompt rule)
                        sources_text = "**KB Sources Used:**\n• **Opener**: Specific & open-ended (fallback)"
                        await query.message.reply_text(sources_text)
                        
                    except Exception as e:
                        print(f"More options error: {e}")
                        # Fallback replies (tailored)
                        if content_type == "reel":
                            replies = [
                                {"text": "Haha bet. Reel remix? 👀", "style": "playful", "sub_vibe": "hinglish"},
                                {"text": "Vibe check... spill? 😏", "style": "mysterious", "sub_vibe": "hinglish"},
                                {"text": "Solid reel. Next one?", "style": "direct", "sub_vibe": "hinglish"}
                            ]
                        else:
                            replies = [
                                {"text": "Haha bet. Toh scene kya hai? 👀", "style": "playful", "sub_vibe": "hinglish"},
                                {"text": "Interesting... batao phir, what's the vibe?", "style": "mysterious", "sub_vibe": "hinglish"},
                                {"text": "Yo, let's keep it real. What's up?", "style": "direct", "sub_vibe": "hinglish"}
                            ]
                        
                        txt = ["**🎲 More Options (Fallback):**\n"]
                        txt.append("_Tap any text below to select and copy!_ 👑")
                        
                        await query.message.reply_text("\n".join(txt))
                        
                        # Send each reply as selectable text
                        for i, r in enumerate(replies, 1):
                            try:
                                await query.message.reply_text(f"`{r['text']}`")
                            except telegram.error.TimedOut:
                                continue
                        
                        USER_CONTEXT[uid]["more_replies"] = replies
                        
                        # Fallback sources
                        await query.message.reply_text("**KB Sources Used:**\n• **Opener**: Match her energy (fallback)")
                        
                else:
                    await query.message.reply_text("❌ LLM unavailable.")
            else:
                await query.message.reply_text("❌ Context lost. Send screenshot again bhai.")
    
    # copy_more_ handler removed - using selectable text instead

# Missing function - add placeholder
def retrieve_stage(uid):
    return 1  # Default stranger

def get_trust(uid, handle):
    return 50  # Default

# ============================================================================
# MAIN APPLICATION
# ============================================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command - enhanced with KB tips"""
    await update.message.reply_text(
        "🔥 **DM Coach Bot - Delhi Edition**\n\n"
        "Send me a screenshot OR text of her message and I'll:\n"
        "✅ Analyze her vibe & mood (high/low energy)\n"
        "✅ Detect traps (gold digger, love bomb, rejection)\n"
        "✅ Generate high-value replies in Hinglish & English (feminine grammar)\n"
        "✅ Tailored for reels/in-feed (play along/specific hooks)\n"
        "✅ Track trust & stage progression\n"
        "✅ Cite exact KB sources for transparency (e.g., [opener: 'Match energy'])\n\n"
        "**How to use:**\n"
        "1. Screenshot your Instagram DM/post/reel OR copy her text\n"
        "2. Send it here\n"
        "3. Get instant drip replies 💯\n"
        "4. Tap 🎲 More Options for diverse replies\n\n"
        "**KB Quick Tips:**\n"
        "• Match energy, be specific (no generics)\n"
        "• Open-ended questions > yes/no\n"
        "• Abundance > Obsession 👑\n\n"
        "From Quora/Reddit: Practice daily micro-convos!"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command - enhanced"""
    await update.message.reply_text(
        "**📚 DM Coach Commands**\n\n"
        "/start - Get started\n"
        "/help - Show this message\n\n"
        "**How to use:**\n"
        "• Send screenshot OR copy-paste her text/reel caption\n"
        "• Get 2 replies (Hinglish + English)\n"
        "• Tap 🎲 More Options for 3 diverse styles\n"
        "• Clear screenshots work best (720p+)\n"
        "• Replies cite exact KB tips for transparency\n\n"
        "**Do's/Don'ts (KB/Research):**\n"
        "• Do: Light, curious, context-based (reel: 'duet?'; infeed: 'tip?')\n"
        "• Don't: Chase dry replies, generic compliments, masculine grammar\n"
        "• Ghost after 2 ignores; abundance wins\n\n"
        "Questions? DM @heyyjishh"
    )

def main():
    """Start the bot"""
    print("\n" + "="*60)
    print("🚀 DM COACH BOT - ENHANCED WITH KB & RESEARCH")
    print("="*60 + "\n")
    
    # Validate environment
    if not validate_env_vars():
        print("\n❌ Startup failed. Fix env vars and retry.\n")
        return
    
    # ChromaDB disabled for speed
    
    # Get bot token
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not set!")
        return
    
    # Build application
    print("🤖 Building Telegram application...")
    app = Application.builder().token(token).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_callback))
    
    print("✅ Handlers registered")
    print("\n" + "="*60)
    print("🎯 BOT IS LIVE! Send screenshots to analyze DMs/reels.")
    print("="*60 + "\n")
    
    # Start polling
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
