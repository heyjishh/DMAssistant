#!/usr/bin/env python3
"""
DM Coach Bot - Fixed & Production Ready
Delhi Gen-Z Instagram DM Analysis Bot with Auto-Reply Feature
"""

import random
import json
import os
import time
import base64
from typing import List, Dict, Any
from PIL import Image

# LangChain & AI imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Telegram imports
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, ContextTypes, MessageHandler, CallbackQueryHandler, CommandHandler, filters
import telegram.error

# ChromaDB imports
import chromadb
from chromadb.config import Settings

# Google Generative AI for Vision
try:
    import google.generativeai as genai
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("⚠️ google-generativeai not installed. Vision API disabled.")

# ============================================================================
# CONFIGURATION & VALIDATION
# ============================================================================

def validate_env_vars():
    """Validate required environment variables"""
    required = {
        "GEMINI_API_KEY": "Gemini API key for LLM and embeddings",
        "TELEGRAM_BOT_TOKEN": "Telegram bot token"
    }
    optional = {
        "CHROMA_API_KEY": "ChromaDB API key (optional for local)",
        "CHROMA_TENANT": "ChromaDB tenant",
        "CHROMA_DATABASE": "ChromaDB database"
    }
    
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"\n❌ Missing required env vars: {', '.join(missing)}")
        print("\n🔧 Set these in your environment:")
        for k in missing:
            print(f"  export {k}='your_key_here'")
        return False
    
    print("✅ All required env vars present")
    
    missing_optional = [k for k in optional if not os.getenv(k)]
    if missing_optional:
        print(f"⚠️ Optional env vars not set: {', '.join(missing_optional)}")
    
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

# Initialize embeddings
try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GEMINI_API_KEY
    )
    print("✅ Embeddings initialized")
except Exception as e:
    print(f"❌ Embeddings init failed: {e}")
    embeddings = None

# Initialize ChromaDB
try:
    chroma_api_key = os.getenv("CHROMA_API_KEY", "ck-8YrdvWKNapbQ7ByG8qjF9iyfVwRjzZBDqRCyT3giyFts")
    chroma_tenant = os.getenv("CHROMA_TENANT", "6debe819-654b-405c-90e5-ce5359ec38ec")
    chroma_database = os.getenv("CHROMA_DATABASE", "Production")
    
    chroma_client = chromadb.CloudClient(
        api_key=chroma_api_key,
        tenant=chroma_tenant,
        database=chroma_database
    )
    collection = chroma_client.get_or_create_collection(name="dm_coach")
    print("✅ ChromaDB connected")
except Exception as e:
    print(f"⚠️ ChromaDB connection failed: {e}")
    print("💡 Trying local ChromaDB...")
    try:
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(name="dm_coach")
        print("✅ Local ChromaDB initialized")
    except Exception as e2:
        print(f"❌ ChromaDB failed completely: {e2}")
        collection = None

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.7,
        google_api_key=GEMINI_API_KEY
    )
    print("✅ LLM initialized")
except Exception as e:
    print(f"❌ LLM init failed: {e}")
    llm = None

# ============================================================================
# CHROMADB SEEDING & HELPERS
# ============================================================================

def init_collection():
    """Seed ChromaDB with base advice if empty"""
    if not collection or not embeddings:
        print("⚠️ ChromaDB or embeddings unavailable. Skipping seed.")
        return
    
    try:
        result = collection.query(query_texts=["test"], n_results=1)
        if not result["ids"] or not result["ids"][0]:
            print("📦 Seeding ChromaDB with base advice...")
            seed_examples = [
                {"id": "advice_1", "text": "Options > obsession. Abundance mindset always wins.", "metadata": {"category": "mindset"}},
                {"id": "advice_2", "text": "Ghost after 2 ignores. Let her miss you, king.", "metadata": {"category": "ghost"}},
                {"id": "advice_3", "text": "Match her energy. Short reply = short reply back.", "metadata": {"category": "mirror"}},
                {"id": "advice_4", "text": "Never double text unless she's earned it with effort.", "metadata": {"category": "value"}},
                {"id": "advice_5", "text": "Gold digger? Neutral pivot. Talk about her interests, not money.", "metadata": {"category": "trap"}},
                {"id": "advice_6", "text": "Love bomb early? Slow down. Ask about her day, hobbies.", "metadata": {"category": "trap"}},
                {"id": "advice_7", "text": "Emotion test? Stay cool. Light tease, then change topic.", "metadata": {"category": "trap"}},
                {"id": "advice_8", "text": "High trust = personal questions work. Low trust = stay surface.", "metadata": {"category": "escalation"}},
            ]
            
            for ex in seed_examples:
                try:
                    embed = embeddings.embed_query(ex["text"])
                    collection.upsert(
                        ids=[ex["id"]],
                        embeddings=[embed],
                        metadatas=[ex["metadata"]],
                        documents=[ex["text"]]
                    )
                except Exception as e:
                    print(f"⚠️ Failed to seed {ex['id']}: {e}")
            
            print("✅ ChromaDB seeded successfully!")
        else:
            print("✅ ChromaDB already has data")
    except Exception as e:
        print(f"⚠️ ChromaDB init check error: {e}")

def safe_collection_query(query_text: str, n_results: int = 3) -> List[str]:
    """Safely query ChromaDB with fallback"""
    if not collection or not embeddings:
        return ["Keep it high-value and confident.", "Match her energy.", "Abundance mindset."]
    
    try:
        embed = embeddings.embed_query(query_text)
        result = collection.query(query_embeddings=[embed], n_results=n_results)
        
        if result["metadatas"] and result["metadatas"][0]:
            return [m.get("text", "Stay cool.") for m in result["metadatas"][0]]
        return ["Keep it high-value.", "Match her vibe."]
    except Exception as e:
        print(f"⚠️ Query error: {e}")
        return ["Be confident.", "Stay high-value."]

# ============================================================================
# STAGE & HISTORY MANAGEMENT
# ============================================================================

def retrieve_stage(uid):
    """Retrieve user's current stage"""
    if not collection:
        return 1
    
    key = f"stage_u{uid}"
    try:
        result = collection.query(query_texts=[key], n_results=1)
        if result["metadatas"] and result["metadatas"][0]:
            return result["metadatas"][0][0].get("s", 1)
    except:
        pass
    return 1

def store_stage(uid, stage):
    """Store user's stage"""
    if not collection or not embeddings:
        return
    
    key = f"stage_u{uid}"
    try:
        embed = embeddings.embed_query(key)
        collection.upsert(
            ids=[key],
            embeddings=[embed],
            metadatas=[{"s": stage}]
        )
    except Exception as e:
        print(f"⚠️ Store stage error: {e}")

def store_history(uid, history, metadata):
    """Store conversation history"""
    if not collection or not embeddings:
        return
    
    key = f"history_u{uid}"
    try:
        embed = embeddings.embed_query(key)
        collection.upsert(
            ids=[key],
            embeddings=[embed],
            metadatas=[metadata],
            documents=[json.dumps(history)]
        )
    except Exception as e:
        print(f"⚠️ Store history error: {e}")

# ============================================================================
# IGNORE & TRUST TRACKING
# ============================================================================

def track_ignore_count(user_id, girl_handle, replied=False):
    """Track how many times she's ignored user"""
    if not collection or not embeddings:
        return 0
    
    key = f"u{user_id}_g{girl_handle}_ign"
    cnt = 0
    
    try:
        r = collection.query(query_texts=[key], n_results=1)
        if r["metadatas"] and r["metadatas"][0]:
            cnt = r["metadatas"][0][0].get("c", 0)
    except:
        pass
    
    if not replied:
        cnt += 1
    
    try:
        embed = embeddings.embed_query(key)
        collection.upsert(
            ids=[key],
            embeddings=[embed],
            metadatas=[{"c": cnt, "g": girl_handle}]
        )
    except Exception as e:
        print(f"⚠️ Track ignore error: {e}")
    
    return cnt

def get_ghost_advice(cnt):
    """Get advice based on ignore count"""
    if cnt <= 1:
        return "Follow-up ok. Light tease works."
    if cnt == 2:
        return "Last text. Add value, then ghost."
    return random.choice([
        "Ghost now. No reply = no interest.",
        "Abundance. DM 3 others.",
        "She's not worth your energy, king."
    ])

def update_trust(user_id, girl_handle, delta):
    """Update trust score"""
    if not collection or not embeddings:
        return 50
    
    key = f"t_u{user_id}_g{girl_handle}"
    s = 50
    
    try:
        cur = collection.query(query_texts=[key], n_results=1)
        if cur["metadatas"] and cur["metadatas"][0]:
            s = cur["metadatas"][0][0].get("s", 50)
    except:
        pass
    
    s = max(0, min(100, s + delta))
    
    try:
        embed = embeddings.embed_query(key)
        collection.upsert(
            ids=[key],
            embeddings=[embed],
            metadatas=[{"s": s, "g": girl_handle}]
        )
    except Exception as e:
        print(f"⚠️ Update trust error: {e}")
    
    return s

def get_trust(user_id, girl_handle):
    """Get current trust score"""
    if not collection:
        return 50
    
    key = f"t_u{user_id}_g{girl_handle}"
    try:
        cur = collection.query(query_texts=[key], n_results=1)
        if cur["metadatas"] and cur["metadatas"][0]:
            return cur["metadatas"][0][0].get("s", 50)
    except:
        pass
    return 50

# ============================================================================
# IMAGE ANALYSIS (VISION API)
# ============================================================================

def analyze_insta_grid(image_path: str) -> Dict[str, Any]:
    """Analyze Instagram screenshot using Gemini Vision"""
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
            "h": "@posh_girl"
        }
    
    try:
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Use Gemini Vision
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        prompt = """
        Analyze this Instagram DM screenshot. Extract:
        1. Conversation history - identify who said what (her vs user)
        2. Her bio (if visible)
        3. Grid vibe (fashion/travel/fitness/casual)
        4. Her handle (if visible)
        5. Her last message specifically
        
        Return ONLY valid JSON:
        {
            "history": [{"role": "her" or "user", "message": "text"}],
            "her_last_message": "her most recent text",
            "bio": "her bio text",
            "grid_desc": "brief description",
            "handle": "@username"
        }
        
        If you can't extract something, use "unknown" or empty array.
        CRITICAL: Extract her_last_message accurately - this is what we'll reply to.
        """
        
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_data}
        ])
        
        # Parse response
        text = response.text.strip()
        # Remove markdown if present
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        
        result = json.loads(text.strip())
        
        # Enrich with LLM analysis if available
        if llm:
            try:
                p = ChatPromptTemplate.from_template(
                    "Given Instagram profile: Bio='{bio}', Grid='{grid}'. Analyze: TOON:{{f:followers_estimate,g:following_estimate,v:vibe,i:is_influencer}}. Output ONLY JSON."
                )
                c = p | llm | JsonOutputParser()
                enriched = c.invoke({
                    "bio": result.get("bio", ""),
                    "grid": result.get("grid_desc", "")
                })
                result.update(enriched)
            except:
                pass
        
        # Set defaults
        result.setdefault("f", "unknown")
        result.setdefault("g", "unknown")
        result.setdefault("b", result.get("bio", "No bio"))
        result.setdefault("d", result.get("grid_desc", "Standard grid"))
        result.setdefault("v", "casual_normal")
        result.setdefault("i", "no")
        result.setdefault("h", result.get("handle", "@unknown"))
        result.setdefault("her_last_message", result.get("history", [{}])[-1].get("message", "hey") if result.get("history") else "hey")
        
        return result
        
    except Exception as e:
        print(f"⚠️ Vision analysis error: {e}")
        # Fallback mock
        return {
            "history": [{"role": "her", "message": "hey"}, {"role": "user", "message": "yo what's up"}],
            "her_last_message": "hey",
            "f": "unknown", "g": "unknown", "b": "Analysis failed",
            "d": "Could not analyze grid", "v": "unknown", "i": "no", "h": "@unknown"
        }

# ============================================================================
# MOOD & TRAP DETECTION
# ============================================================================

def sense_her_mood(hist: List[Dict[str, str]], uid: int, girl: str):
    """Analyze her mood from messages"""
    if not llm:
        return {"m": "medium_cool", "s": "neutral_calm", "g": "Keep it cool.", "t": None}
    
    msgs = [m["message"] for m in hist if m["role"] == "her"][-3:]
    ign = track_ignore_count(uid, girl, replied=bool(msgs))
    
    msgs_str = " | ".join(msgs) if msgs else "No messages yet"
    
    try:
        p = ChatPromptTemplate.from_template(
            "Messages: '{m}' | Ignores: {i}. Mood? TOON:{{m:mood,s:shift}}. Moods: high_enthu/medium_cool/low_dry/angry_defensive/ignore. Output ONLY JSON."
        )
        c = p | llm | JsonOutputParser()
        mood = c.invoke({"m": msgs_str, "i": ign})
    except:
        mood = {"m": "medium_cool", "s": "neutral_calm"}
    
    mood["g"] = get_ghost_advice(ign)
    
    # Check for traps
    trap = detect_traps(hist, ign)
    mood["t"] = trap
    
    # Update trust based on message content
    for msg in msgs:
        low = msg.lower()
        if any(p in low for p in ["family", "dream", "fear", "secret", "honest"]):
            update_trust(uid, girl, +10)
        elif any(p in low for p in ["test", "prove", "why should", "convince"]):
            update_trust(uid, girl, -10)
    
    return mood

def detect_traps(hist: List[Dict[str, str]], ign: int):
    """Detect common dating traps"""
    # Love bomb detector
    her_msgs = [m["message"].lower() for m in hist if m["role"] == "her"]
    love_kw = ["miss you", "need you", "cant stop", "perfect", "soulmate", "meant to be"]
    love_flags = sum(any(k in m for k in love_kw) for m in her_msgs)
    
    if love_flags >= 3 and len(hist) < 10:
        return {
            "t": "love_bomb",
            "a": "Early intense affection. Red flag for manipulation.",
            "h": "Stay calm. Slow the pace.",
            "f": "Ask about her day, hobbies. Keep it light."
        }
    
    # Gold digger detector
    money_kw = ["expensive", "buy", "gift", "rich", "afford", "luxury", "shopping"]
    money_flags = sum(any(k in m for k in money_kw) for m in her_msgs)
    
    if money_flags >= 2:
        return {
            "t": "gold_digger",
            "a": "Money focus early. Test her.",
            "h": "Neutral pivot to values.",
            "f": "Talk about her passions, not your wallet."
        }
    
    # Emotion test
    if len(her_msgs) >= 3:
        early_long = any(len(m.split()) > 20 for m in her_msgs[:-1])
        recent_short = len(her_msgs[-1].split()) < 5
        
        if early_long and recent_short:
            return {
                "t": "emotion_test",
                "a": "Sudden coldness after warmth. Testing you.",
                "h": "Stay unfazed. Light tease.",
                "f": "Confidence flips the script."
            }
    
    return None

# ============================================================================
# REPLY GENERATION
# ============================================================================

def generate_cot_reply(extract: Dict[str, Any], uid: int):
    """Generate reply using Chain of Thought"""
    if not llm:
        # Fallback replies
        return {
            "r": [
                {"text": "Yo, scene kya hai? Vibe check karein?", "sub_vibe": "hinglish"},
                {"text": "Hey, what's the vibe? Let's check it out.", "sub_vibe": "english"}
            ],
            "a": "LLM unavailable. Using default replies.",
            "h": "Keep it simple.",
            "s": "False"
        }, None, 50
    
    mood = extract.get("mood", {})
    trap = mood.get("t")
    cur_stage = extract.get("c", 1)
    her_last = extract.get("her_last_message", "hey")
    
    # Get RAG advice
    q = f"stage_{cur_stage} {extract.get('v', 'normal')} {mood.get('m', 'medium')}"
    if trap:
        q += f" {trap['t']}_trap"
    
    snippets = safe_collection_query(q, n_results=3)
    
    # Build prompt
    history_str = "\n".join([
        f"{m['role']}: {m['message']}"
        for m in extract.get('history', [])
    ])
    
    prompt = ChatPromptTemplate.from_template(
        """You are a Delhi Gen-Z DM coach. Generate replies in Hinglish and English.

Context:
- History: {history}
- Her Last Message: "{her_last}"
- Stage: {stage} (1=Stranger, 4=GF)
- Her Mood: {mood}
- Trap: {trap}
- Advice: {advice}

Generate TOON: {{r:[{{text:str,sub_vibe:str}}],a:analysis,h:hook}}

Rules:
- 2 replies: one Hinglish, one English
- Under 30 words each
- High-value, confident tone
- Use Delhi slang naturally (bhai, yaar, scene, vibe, etc.)
- DIRECTLY respond to her last message: "{her_last}"
- Address any trap directly
- Make replies sound natural, not robotic

Output ONLY valid JSON."""
    )
    
    try:
        chain = prompt | llm | JsonOutputParser()
        out = chain.invoke({
            "history": history_str,
            "her_last": her_last,
            "stage": cur_stage,
            "mood": mood.get("m", "medium_cool"),
            "trap": trap["t"] if trap else "none",
            "advice": " | ".join(snippets)
        })
        
        if not isinstance(out, dict) or 'r' not in out:
            raise ValueError("Invalid LLM output")
            
    except Exception as e:
        print(f"⚠️ LLM generation error: {e}")
        # Smart fallback based on her message
        her_last_lower = her_last.lower()
        if "hey" in her_last_lower or "hi" in her_last_lower:
            out = {
                "r": [
                    {"text": f"Yo! Scene kya hai? Vibe check? 👀", "sub_vibe": "hinglish"},
                    {"text": f"Hey! What's the scene? Vibe check? 👀", "sub_vibe": "english"}
                ],
                "a": "Casual opener response.",
                "h": "Match her energy"
            }
        else:
            out = {
                "r": [
                    {"text": f"Haha bet. Toh batao, kya chal raha?", "sub_vibe": "hinglish"},
                    {"text": f"Haha nice. So what's up with you?", "sub_vibe": "english"}
                ],
                "a": "Keep conversation flowing.",
                "h": "Stay engaging"
            }
    
    # Check for warnings
    warn = None
    user_streak = sum(1 for m in reversed(extract.get("history", [])[:5]) if m["role"] == "user")
    if user_streak >= 3:
        warn = "⚠️ 3+ texts in a row. Let her miss you!"
    
    trust = get_trust(uid, extract.get("h", "unknown"))
    
    return out, warn, trust

# ============================================================================
# TELEGRAM HANDLERS
# ============================================================================

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle screenshot uploads"""
    uid = update.effective_user.id
    now = time.time()
    
    # Rate limiting
    if uid in LAST_PHOTO and now - LAST_PHOTO[uid] < 30:
        await update.message.reply_text(
            "⏳ Chill yaar! 30 sec cooldown.\n\n"
            "**Abundance** > desperation, remember? 💯"
        )
        return
    
    LAST_PHOTO[uid] = now
    
    # Download photo
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    fp = os.path.join("/tmp", f"screenshot_{uid}_{int(now)}.jpg")
    
    await update.message.reply_text("🔍 Analyzing your DM game...")
    
    try:
        await file.download_to_drive(fp)
        
        # Validate image
        img = Image.open(fp)
        if img.size[0] < 200 or img.size[1] < 200:
            await update.message.reply_text("🚫 Image too small bhai. Send HD screenshot (720p+).")
            return
        
        # Process
        extract = {"history": [], "h": "@girl"}
        grid = analyze_insta_grid(fp)
        extract.update(grid)
        
        # Get/update stage
        cur = retrieve_stage(uid)
        extract["c"] = cur
        extract["n"] = cur
        
        # Analyze mood
        girl = extract.get("h", "@unknown")
        mood = sense_her_mood(extract.get("history", []), uid, girl)
        extract["mood"] = mood
        
        # Generate reply
        cot, warn, trust = generate_cot_reply(extract, uid)
        
        # Store context for this user
        USER_CONTEXT[uid] = {
            "extract": extract,
            "cot": cot,
            "warn": warn,
            "trust": trust,
            "girl": girl
        }
        
        # Update stage
        new_stage = extract["n"]
        if mood["m"] == "high_enthu" and not mood.get("t"):
            new_stage = min(new_stage + 0.5, 4)
        elif mood["m"] == "low_dry" or mood.get("t"):
            new_stage = max(new_stage - 0.5, 1)
        
        store_stage(uid, round(new_stage))
        
        # Build response
        txt = ["**🔥 DM Coach Report**\n"]
        
        # Show her last message
        txt.append(f"💬 **Her:** \"{extract.get('her_last_message', 'unknown')}\"\n")
        
        # Trap warnings
        trap = mood.get("t")
        if trap:
            txt.append(f"⚠️ **{trap['t'].replace('_', ' ').title()} Detected**")
            txt.append(f"   └ {trap['a']}")
            txt.append(f"   └ **Play:** {trap['f']}\n")
        
        # Stats
        ign = track_ignore_count(uid, girl, replied=bool(extract.get("history")))
        stage_map = {1: "Stranger 🆕", 2: "Friend 👋", 3: "Talking 💬", 4: "GF 💕"}
        txt.append(f"💖 **Trust:** {trust}/100  |  🚫 **Ignores:** {ign}")
        txt.append(f"📊 **Stage:** {stage_map.get(round(new_stage), 'N/A')}  |  **Mood:** {mood['m']}\n")
        
        # Warnings
        if warn:
            txt.append(f"🚨 {warn}\n")
        
        if ign >= 2:
            txt.append(f"💡 **Ghost Advice:** {mood['g']}\n")
        
        # Analysis
        txt.append(f"🧠 **Play:** {cot.get('a', 'Keep it high-value.')}\n")
        
        # Replies
        replies = cot.get('r', [])
        txt.append("**📱 Your Drip:**")
        for i, r in enumerate(replies, 1):
            vibe = r.get('sub_vibe', 'unknown').title()
            txt.append(f"{i}️⃣ `[{vibe}]` {r['text']}")
        
        txt.append("\n_Tap a button to send the reply OR get it as text!_ 👑")
        
        # Keyboard - added "Send as Text" option
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton(f"Copy {i+1}", callback_data=f"copy_{uid}_{i}")]
            for i in range(len(replies[:3]))
        ] + [
            [InlineKeyboardButton("📋 Send Hinglish as Text", callback_data=f"text_{uid}_0")],
            [InlineKeyboardButton("📋 Send English as Text", callback_data=f"text_{uid}_1")]
        ])
        
        await update.message.reply_text("\n".join(txt), reply_markup=kb)
        
    except FileNotFoundError:
        await update.message.reply_text("🚫 Download failed. Telegram server BT. Retry bhai.")
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
        if os.path.exists(fp):
            os.remove(fp)

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button presses"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data.startswith("copy_"):
        # Copy button pressed
        parts = data.split("_")
        if len(parts) >= 3:
            uid = int(parts[1])
            idx = int(parts[2])
            
            if uid in USER_CONTEXT:
                replies = USER_CONTEXT[uid]["cot"].get("r", [])
                if idx < len(replies):
                    reply_text = replies[idx]["text"]
                    await query.message.reply_text(
                        f"📋 **Reply Copied!**\n\n"
                        f"_{reply_text}_\n\n"
                        "Paste this in your DM. Remember:\n"
                        "• Match her energy\n"
                        "• Stay high-value\n"
                        "• Options > obsession\n\n"
                        "Good luck, king! 👑"
                    )
                else:
                    await query.message.reply_text("❌ Reply not found. Send screenshot again.")
            else:
                await query.message.reply_text("❌ Context lost. Send screenshot again bhai.")
    
    elif data.startswith("text_"):
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
                        "💯 Remember: High-value, abundance mindset",
                        "👑 You got this, king!"
                    ]
                    
                    await query.message.reply_text("\n".join(response))
                    
                    # Also send as a separate message for easy copying
                    await query.message.reply_text(
                        f"`{reply_text}`",
                        parse_mode='Markdown'
                    )
                else:
                    await query.message.reply_text("❌ Reply not found. Send screenshot again.")
            else:
                await query.message.reply_text("❌ Context lost. Send screenshot again bhai.")
