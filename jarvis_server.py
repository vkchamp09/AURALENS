#!/usr/bin/env python3
"""
JARVIS Ultra Backend v3.0 - ALL BUGS FIXED & ENHANCED
- Modular architecture
- Enhanced error handling
- Better TTS management
- Improved camera feed
- Connection monitoring
- Performance optimizations
"""
import os
import json
import time
import webbrowser
import threading
import queue
import logging
import subprocess
import platform
import re
import random
from datetime import datetime as dt
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import quote

import psutil

# ═══════════════════════════════════════════════════════════════════════════════
# DEPENDENCY IMPORTS WITH GRACEFUL FALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

# Text-to-Speech
try:
    import pyttsx3
    TTS_ENGINE = pyttsx3.init()
    TTS_ENGINE.setProperty('rate', 180)
    TTS_ENGINE.setProperty('volume', 0.9)
    TTS_AVAILABLE = True
    print("[✓] Text-to-Speech: pyttsx3 loaded")
except ImportError:
    TTS_AVAILABLE = False
    print("[!] pyttsx3 not available. Using system TTS...")
except Exception as e:
    TTS_AVAILABLE = False
    print(f"[!] pyttsx3 initialization error: {e}")

# Camera
try:
    import cv2
    import numpy as np
    import base64
    CV2_AVAILABLE = True
    print("[✓] Camera: OpenCV loaded")
except ImportError:
    CV2_AVAILABLE = False
    print("[!] OpenCV not available")

# Wikipedia
try:
    import wikipedia
    wikipedia.set_lang('en')
    WIKIPEDIA_AVAILABLE = True
    print("[✓] Wikipedia: Loaded")
except ImportError:
    WIKIPEDIA_AVAILABLE = False
    print("[!] Wikipedia not available")

# Search
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
    print("[✓] Search: DuckDuckGo loaded")
except ImportError:
    DDGS_AVAILABLE = False
    print("[!] DuckDuckGo not available")

# Math
try:
    import sympy
    SYMPY_AVAILABLE = True
    print("[✓] Math: Sympy loaded")
except ImportError:
    SYMPY_AVAILABLE = False
    print("[!] Sympy not available")

# Web requests
try:
    import requests
    REQUESTS_AVAILABLE = True
    print("[✓] Web: Requests loaded")
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[!] Requests not available")

# Face Recognition (optional)
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("[✓] Face Recognition: Loaded")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("[!] Face recognition not available")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "port": 8080,
    "host": "localhost",
    "cors_enabled": True,
    "tts_enabled": True,
    "voice_rate": 180,
    "frontend_tts": True,
    "max_response_length": 500,
    "camera_retry_delay": 1.0,
    "log_level": logging.INFO
}

logging.basicConfig(
    filename='jarvis.log',
    level=CONFIG['log_level'],
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED TTS ENGINE - GUARANTEED SPEECH WITH BETTER ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

class TTSEngine:
    """Advanced Text-to-Speech with multiple fallback methods"""
    
    def __init__(self):
        self.queue = queue.Queue(maxsize=100)
        self.is_running = True
        self.system = platform.system()
        self.method = None
        self.speech_count = 0
        self.failed_count = 0
        self.lock = threading.Lock()
        
        # Determine best TTS method
        if TTS_AVAILABLE:
            self.method = 'pyttsx3'
        elif self.system == "Windows":
            self.method = 'windows'
        elif self.system == "Darwin":
            self.method = 'macos'
        elif self.system == "Linux":
            self.method = 'linux'
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        print(f"[TTS] Method: {self.method or 'Frontend Only'}")
        print(f"[TTS] Queue Size: 100")
    
    def _worker(self):
        """Background worker - processes speech queue"""
        while self.is_running:
            try:
                text = self.queue.get(timeout=0.5)
                if text:
                    self.speech_count += 1
                    print(f"[TTS #{self.speech_count}] Speaking: {text[:60]}...")
                    logging.info(f"TTS #{self.speech_count}: {text[:50]}")
                    
                    success = self._speak_now(text)
                    if not success:
                        self.failed_count += 1
                        logging.warning(f"TTS failed (total: {self.failed_count})")
                    
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"TTS worker error: {e}")
                print(f"[TTS ERROR] {e}")
    
    def _clean_text(self, text):
        """Clean text for speech"""
        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', '', text)
        # Remove special characters
        clean = clean.replace('&bull;', '').replace('•', '')
        clean = clean.replace('&nbsp;', ' ')
        # Remove emojis
        clean = re.sub(r'[\U0001F300-\U0001F9FF]', '', clean)
        # Remove extra whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        # Limit length
        if len(clean) > CONFIG['max_response_length']:
            clean = clean[:CONFIG['max_response_length']-3] + '...'
        
        return clean
    
    def _speak_now(self, text):
        """Speak text using available method"""
        clean_text = self._clean_text(text)
        
        if not clean_text:
            return False
        
        with self.lock:  # Prevent concurrent TTS calls
            try:
                if self.method == 'pyttsx3':
                    TTS_ENGINE.say(clean_text)
                    TTS_ENGINE.runAndWait()
                    print(f"[TTS] ✓ Spoke via pyttsx3")
                    return True
                
                elif self.method == 'windows':
                    escaped = clean_text.replace('"', '`"').replace("'", "''")
                    cmd = f'''Add-Type -AssemblyName System.Speech; 
                             $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; 
                             $speak.Rate = 1;
                             $speak.Speak("{escaped}")'''
                    
                    result = subprocess.run(
                        ["powershell", "-Command", cmd],
                        capture_output=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        print(f"[TTS] ✓ Spoke via Windows TTS")
                        return True
                
                elif self.method == 'macos':
                    subprocess.run(["say", clean_text], timeout=10, check=True)
                    print(f"[TTS] ✓ Spoke via macOS say")
                    return True
                
                elif self.method == 'linux':
                    for cmd in [['espeak', '-s', '180', clean_text], 
                               ['spd-say', clean_text],
                               ['festival', '--tts']]:
                        try:
                            if cmd[0] == 'festival':
                                proc = subprocess.Popen(
                                    cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL
                                )
                                proc.communicate(input=clean_text.encode(), timeout=10)
                            else:
                                subprocess.run(
                                    cmd, timeout=10, check=True,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL
                                )
                            print(f"[TTS] ✓ Spoke via {cmd[0]}")
                            return True
                        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                            continue
            
            except subprocess.TimeoutExpired:
                print(f"[TTS] ⚠ Timeout (text too long)")
            except Exception as e:
                logging.error(f"TTS error: {e}")
                print(f"[TTS] ✗ Error: {e}")
        
        return False
    
    def speak(self, text):
        """Queue text for speaking (non-blocking)"""
        if not self.method and not CONFIG['frontend_tts']:
            return
        
        try:
            self.queue.put_nowait(text)
            print(f"[TTS] → Queued: {text[:60]}...")
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(text)
                print(f"[TTS] → Queued (cleared old)")
            except:
                print(f"[TTS] ⚠ Queue full")
    
    def get_stats(self):
        """Get TTS statistics"""
        return {
            'total_speeches': self.speech_count,
            'failed_speeches': self.failed_count,
            'queue_size': self.queue.qsize(),
            'method': self.method
        }
    
    def stop(self):
        """Stop TTS engine"""
        self.is_running = False
        print(f"[TTS] Stats - Total: {self.speech_count}, Failed: {self.failed_count}")

# ═══════════════════════════════════════════════════════════════════════════════
# WIKIPEDIA ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class WikipediaEngine:
    """Enhanced Wikipedia search with caching"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 3600  # 1 hour
    
    def search(self, query):
        if not WIKIPEDIA_AVAILABLE:
            return None
        
        # Check cache
        cache_key = query.lower().strip()
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if time.time() - cached_time < self.cache_timeout:
                print(f"[Wikipedia] Cache hit: {query}")
                return cached_result
        
        try:
            result = wikipedia.summary(query, sentences=3, auto_suggest=False)
            response = f"<strong>📚 Wikipedia:</strong><br>{result}"
            self.cache[cache_key] = (time.time(), response)
            return response
            
        except wikipedia.exceptions.DisambiguationError as e:
            try:
                options = e.options[:3]
                result = wikipedia.summary(options[0], sentences=3)
                response = f"<strong>📚 Wikipedia ({options[0]}):</strong><br>{result}"
                self.cache[cache_key] = (time.time(), response)
                return response
            except:
                pass
                
        except wikipedia.exceptions.PageError:
            try:
                result = wikipedia.summary(query, sentences=3, auto_suggest=True)
                response = f"<strong>📚 Wikipedia:</strong><br>{result}"
                self.cache[cache_key] = (time.time(), response)
                return response
            except:
                pass
                
        except Exception as e:
            logging.error(f"Wikipedia error: {e}")
        
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SearchEngine:
    """Unified search across multiple sources"""
    
    def __init__(self):
        self.wiki = WikipediaEngine()
    
    def search(self, query):
        # Try Wikipedia first (more reliable)
        wiki_result = self.wiki.search(query)
        if wiki_result:
            return wiki_result
        
        # Fallback to DuckDuckGo
        if DDGS_AVAILABLE:
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=3))
                    if results:
                        output = ["<strong>🔍 Search Results:</strong>"]
                        for i, r in enumerate(results[:2], 1):
                            title = r.get('title', 'Result')
                            body = r.get('body', '')[:250]
                            output.append(f"<br><br><strong>{i}. {title}</strong><br>{body}...")
                        return "".join(output)
            except Exception as e:
                logging.error(f"DuckDuckGo error: {e}")
        
        return f"I couldn't find detailed information about '{query}'. Try asking in a different way."

# ═══════════════════════════════════════════════════════════════════════════════
# CAMERA ENGINE WITH BETTER ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

class CameraEngine:
    """Enhanced camera with face recognition and error recovery"""
    
    def __init__(self):
        self.is_running = False
        self.current_frame = None
        self.current_detections = {'people': [], 'objects': []}
        self.lock = threading.Lock()
        self.known_faces = []
        self.known_names = []
        self.camera_errors = 0
        self.frames_processed = 0
        
        if CV2_AVAILABLE:
            if FACE_RECOGNITION_AVAILABLE:
                self._load_known_faces()
            self.start()
    
    def _load_known_faces(self):
        """Load known faces from directory"""
        faces_dir = "known_faces"
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
            print(f"[Camera] Created 'known_faces' directory")
            return
        
        for filename in os.listdir(faces_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(faces_dir, filename)
                try:
                    image = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_faces.append(encodings[0])
                        self.known_names.append(os.path.splitext(filename)[0])
                        print(f"[Camera] ✓ Loaded face: {filename}")
                except Exception as e:
                    print(f"[Camera] ✗ Error loading {filename}: {e}")
    
    def start(self):
        """Start camera thread"""
        self.is_running = True
        thread = threading.Thread(target=self._camera_loop, daemon=True)
        thread.start()
        print(f"[Camera] Started")
    
    def _camera_loop(self):
        """Main camera processing loop"""
        cap = None
        
        while self.is_running:
            try:
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        logging.error("Camera not available")
                        time.sleep(CONFIG['camera_retry_delay'])
                        continue
                    
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 15)
                    print(f"[Camera] Opened successfully")
                
                ret, frame = cap.read()
                if not ret:
                    self.camera_errors += 1
                    if self.camera_errors > 10:
                        print(f"[Camera] Too many errors, reinitializing...")
                        cap.release()
                        cap = None
                        self.camera_errors = 0
                    time.sleep(0.1)
                    continue
                
                self.camera_errors = 0
                self.frames_processed += 1
                
                with self.lock:
                    self.current_frame = frame.copy()
                
                # Process faces every 5 frames
                if FACE_RECOGNITION_AVAILABLE and self.frames_processed % 5 == 0 and len(self.known_faces) > 0:
                    self._detect_faces(frame)
                
                time.sleep(0.05)
            
            except Exception as e:
                logging.error(f"Camera loop error: {e}")
                time.sleep(1)
        
        if cap:
            cap.release()
    
    def _detect_faces(self, frame):
        """Detect and recognize faces"""
        try:
            # Resize for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small, model='hog')
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
            
            detections = []
            
            for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                # Scale back up
                top, right, bottom, left = top*4, right*4, bottom*4, left*4
                
                name = "Unknown"
                confidence = 0
                
                if len(self.known_faces) > 0:
                    distances = face_recognition.face_distance(self.known_faces, encoding)
                    best_match_idx = np.argmin(distances)
                    
                    if distances[best_match_idx] < 0.6:
                        name = self.known_names[best_match_idx]
                        confidence = (1 - distances[best_match_idx]) * 100
                
                # Estimate distance (rough approximation)
                face_width = right - left
                distance = round((0.16 * 650) / face_width if face_width > 0 else 2.0, 2)
                
                # Determine position
                center_x = left + (right - left) // 2
                width = frame.shape[1]
                position = "left" if center_x < width/3 else ("center" if center_x < 2*width/3 else "right")
                
                detections.append({
                    'name': name,
                    'confidence': round(confidence, 1),
                    'distance': distance,
                    'position': position,
                    'box': (left, top, right, bottom)
                })
            
            with self.lock:
                self.current_detections['people'] = detections
        
        except Exception as e:
            logging.error(f"Face detection error: {e}")
    
    def get_frame_base64(self):
        """Get current frame as base64 with overlays"""
        with self.lock:
            frame = self.current_frame.copy() if self.current_frame is not None else None
            detections = self.current_detections.copy()
        
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            # Draw detection boxes
            for person in detections.get('people', []):
                left, top, right, bottom = person['box']
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                label = f"{person['name']} {person['distance']}m"
                cv2.putText(frame, label, (left, top-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret:
            return base64.b64encode(buffer).decode('utf-8'), detections
        
        return None, detections
    
    def get_scene_description(self):
        """Get natural language description of scene"""
        with self.lock:
            detections = self.current_detections.copy()
        
        people = detections.get('people', [])
        
        if not people:
            return "No people detected in view."
        
        descriptions = []
        for person in people:
            desc = f"{person['name']} on the {person['position']}, approximately {person['distance']} meters away"
            descriptions.append(desc)
        
        return " and ".join(descriptions) + "."
    
    def get_stats(self):
        """Get camera statistics"""
        return {
            'frames_processed': self.frames_processed,
            'camera_errors': self.camera_errors,
            'known_faces': len(self.known_faces)
        }
    
    def stop(self):
        """Stop camera"""
        self.is_running = False
        print(f"[Camera] Stopped - Frames: {self.frames_processed}, Errors: {self.camera_errors}")

# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSATIONAL RESPONSES WITH MORE VARIETY
# ═══════════════════════════════════════════════════════════════════════════════

RESPONSES = {
    'greeting': {
        'patterns': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good evening', 'good afternoon'],
        'responses': [
            "Hello! JARVIS systems online and ready to assist.",
            "Greetings! All systems operational. How may I help you?",
            "Good day! Standing by for your commands.",
            "Hello there! Neural networks active and ready."
        ]
    },
    'status': {
        'patterns': ['how are you', 'status', 'are you ok', 'you good', 'how you doing'],
        'responses': [
            "All systems functioning at optimal capacity. Ready to assist!",
            "Operating at peak efficiency. How can I help you today?",
            "All diagnostics show green. Standing by for commands.",
            "Systems nominal. Ready for any task you have in mind."
        ]
    },
    'identity': {
        'patterns': ['who are you', 'what is your name', 'introduce yourself', "what's your name"],
        'responses': [
            "I am JARVIS - Just A Rather Very Intelligent System. Your personal AI assistant.",
            "I'm JARVIS version 3.0, your advanced AI companion designed to assist with various tasks.",
            "JARVIS at your service - your intelligent virtual assistant."
        ]
    },
    'creator': {
        'patterns': ['who made you', 'who created you', 'who built you', 'your creator'],
        'responses': [
            "I was created by Vedant, my brilliant creator and commander.",
            "Vedant built me to be an advanced AI assistant.",
            "I'm the creation of Vedant - a true innovator!"
        ]
    },
    'thanks': {
        'patterns': ['thank you', 'thanks', 'appreciate it', 'good job', 'well done'],
        'responses': [
            "You're very welcome! Happy to help anytime.",
            "My pleasure! That's what I'm here for.",
            "Glad I could assist! Let me know if you need anything else.",
            "Always happy to help!"
        ]
    },
    'capabilities': {
        'patterns': ['what can you do', 'help', 'capabilities', 'features', 'commands'],
        'responses': [
            """I can help you with many things:<br><br>
            <strong>Information & Search:</strong><br>
            • Search Wikipedia and the web<br>
            • Get weather updates<br>
            • Answer questions<br><br>
            <strong>Calculations:</strong><br>
            • Solve math problems<br>
            • Perform calculations<br><br>
            <strong>System:</strong><br>
            • Check CPU, memory, disk usage<br>
            • Monitor system health<br><br>
            <strong>Utilities:</strong><br>
            • Tell the time and date<br>
            • Tell jokes<br>
            • Open websites<br>
            • Camera and face detection<br><br>
            Try asking me anything!"""
        ]
    }
}

def get_conversational_response(text):
    """Get conversational response if applicable"""
    text_lower = text.lower().strip()
    
    for category, data in RESPONSES.items():
        for pattern in data['patterns']:
            if pattern in text_lower:
                return random.choice(data['responses'])
    
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND HANDLERS (Modularized)
# ═══════════════════════════════════════════════════════════════════════════════

def handle_time(cmd):
    """Handle time queries"""
    now = dt.now()
    time_str = now.strftime('%I:%M:%S %p')
    return f"⏰ The current time is <strong>{time_str}</strong>"

def handle_date(cmd):
    """Handle date queries"""
    now = dt.now()
    day = now.strftime('%A')
    date_str = now.strftime('%B %d, %Y')
    return f"📅 Today is <strong>{day}, {date_str}</strong>"

def handle_weather(cmd):
    """Handle weather queries"""
    if not REQUESTS_AVAILABLE:
        return "Weather service requires the 'requests' library."
    
    try:
        response = requests.get('https://wttr.in/?format=3', timeout=5)
        if response.status_code == 200:
            return f"🌤️ <strong>Current Weather:</strong><br>{response.text}"
    except Exception as e:
        logging.error(f"Weather error: {e}")
    
    return "Unable to fetch weather data at the moment."

def handle_facts(cmd):
    import subprocess
    import webbrowser
    import time
    import os

# Path to backend file
    BACKEND = "truthguard_backend.py.py"

# Path to frontend file
    FRONTEND = "truthguard.html"

    print("🔵 Starting TruthGuard Backend...")

# Start backend server
    backend_process = subprocess.Popen(["python", BACKEND])

# Wait a bit so backend initializes
    time.sleep(2)

    print("🟢 Opening TruthGuard UI in browser...")
    webbrowser.open("file://" + os.path.abspath(FRONTEND))

    print("🚀 TruthGuard fully launched!")
    print("Press CTRL + C to stop backend.")
    backend_process.wait()


def handle_math(cmd):
    """Handle math calculations"""
    if not SYMPY_AVAILABLE:
        return "Math calculations require 'sympy'."
    
    try:
        expr = cmd.lower()
        for word in ['calculate', 'compute', 'solve', 'what is', "what's", 'math']:
            expr = expr.replace(word, '')
        
        expr = expr.strip().replace('^', '**').replace('x', '*').replace('×', '*')
        expr = ''.join(c for c in expr if c.isdigit() or c in '+-*/().%** ')
        
        if expr:
            result = sympy.sympify(expr)
            value = float(result.evalf())
            
            if value.is_integer():
                formatted = f"{int(value):,}"
            else:
                formatted = f"{value:,.4f}".rstrip('0').rstrip('.')
            
            return f"🧮 <strong>Calculation Result:</strong><br>{expr} = <strong>{formatted}</strong>"
    
    except Exception as e:
        logging.error(f"Math error: {e}")
    
    return "I couldn't calculate that. Try something like: 'calculate 25 * 4'"

def handle_system(cmd):
    """Handle system info queries"""
    try:
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return f"""⚙️ <strong>System Status Report:</strong><br><br>
        <strong>CPU:</strong> {cpu}% utilization<br>
        <strong>Memory:</strong> {memory.percent}% used ({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)<br>
        <strong>Disk:</strong> {disk.percent}% used ({disk.used / (1024**3):.1f}GB / {disk.total / (1024**3):.1f}GB)<br><br>
        All systems running smoothly!"""
    
    except Exception as e:
        logging.error(f"System info error: {e}")
        return "Unable to retrieve system information."

def handle_joke(cmd):
    """Handle joke requests"""
    jokes = [
        "Why don't scientists trust atoms? Because they make up everything!",
        "Why did the programmer quit his job? He didn't get arrays!",
        "What do you call a programmer from Finland? Nerdic!",
        "Why do programmers prefer dark mode? Because light attracts bugs!",
        "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
        "What's a programmer's favorite hangout place? Foo Bar!",
        "Why did the developer go broke? Because he used up all his cache!",
        "What do you call 8 hobbits? A hobbyte!",
        "There are 10 types of people: those who understand binary and those who don't.",
        "Why did the AI go to therapy? It had too many deep learning issues!"
    ]
    joke = random.choice(jokes)
    return f"😂 {joke}"

def handle_search(cmd):
    """Handle search queries"""
    query = cmd.lower()
    for prefix in ['search', 'search for', 'find', 'look up', 'who is', 'what is', 'tell me about', 'about']:
        query = query.replace(prefix, '')
    
    query = query.strip()
    
    if not query:
        return "What would you like me to search for?"
    
    return search_engine.search(query)

def handle_open(cmd):
    """Handle website opening"""
    sites = {
        'youtube': 'https://www.youtube.com',
        'google': 'https://www.google.com',
        'gmail': 'https://mail.google.com',
        'github': 'https://www.github.com',
        'reddit': 'https://www.reddit.com',
        'twitter': 'https://www.twitter.com',
        'facebook': 'https://www.facebook.com',
        'linkedin': 'https://www.linkedin.com',
        'instagram': 'https://www.instagram.com',
        'netflix': 'https://www.netflix.com',
        'spotify': 'https://www.spotify.com',
        'amazon': 'https://www.amazon.com'
    }
    
    cmd_lower = cmd.lower()
    
    for site, url in sites.items():
        if site in cmd_lower:
            try:
                webbrowser.open(url)
                return f"🌐 Opening <strong>{site.title()}</strong> in your browser..."
            except Exception as e:
                logging.error(f"Browser open error: {e}")
                return f"Unable to open {site}."
    
    return "Which website would you like to open? Try: 'open youtube', 'open google', etc."

def handle_camera(cmd):
    """Handle camera queries"""
    if not camera:
        return "📹 Camera is not available. Please install OpenCV."
    
    description = camera.get_scene_description()
    return f"📹 <strong>Camera Analysis:</strong><br>{description}"

# ═══════════════════════════════════════════════════════════════════════════════
# INTENT DETECTION WITH PRIORITY
# ═══════════════════════════════════════════════════════════════════════════════

def detect_intent(text):
    """Detect user intent from text"""
    text_lower = text.lower().strip()
    
    # Priority order matters
    if any(w in text_lower for w in ['time', 'what time', 'current time', 'clock']):
        return 'time'
    
    if any(w in text_lower for w in ['date', 'today', 'what day', 'current date', 'day is it']):
        return 'date'
    
    if 'weather' in text_lower:
        return 'weather'
    
    # Math detection
    if any(w in text_lower for w in ['calculate', 'compute', 'solve', 'math', 'what is']) or \
       any(c in text_lower for c in ['+', '-', '*', '/', '=', '^']):
        if any(c.isdigit() for c in text_lower):
            return 'math'
    
    if any(w in text_lower for w in ['system', 'cpu', 'memory', 'ram', 'disk', 'storage', 'performance']):
        return 'system'
    
    if any(w in text_lower for w in ['joke', 'funny', 'laugh', 'humor']):
        return 'joke'
    
    if any(w in text_lower for w in ['camera', 'scan', 'see', 'show me', 'what do you see', 'look', 'detect']):
        return 'camera'
    
    if 'open' in text_lower:
        return 'open'
    
    if 'misinformation' in text_lower:
        return 'misinformation'
    
    if any(w in text_lower for w in ['search', 'find', 'look up', 'who is', 'what is', 'tell me about']):
        return 'search'
    
    return 'chat'

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

def process_command(text):
    """Process command and return response"""
    try:
        # Try conversational response first
        conv_response = get_conversational_response(text)
        if conv_response:
            return conv_response
        
        # Detect intent and route to handler
        intent = detect_intent(text)
        
        handlers = {
            'time': handle_time,
            'date': handle_date,
            'weather': handle_weather,
            'math': handle_math,
            'system': handle_system,
            'joke': handle_joke,
            'camera': handle_camera,
            'open': handle_open,
            'search': handle_search,
            'misinformation': handle_facts
            
            
        }
        
        handler = handlers.get(intent)
        if handler:
            return handler(text)
        
        # Default fallback
        return random.choice([
            "I'm not quite sure what you mean. Could you rephrase that?",
            "Interesting! Could you provide more details?",
            "I'm here to help. What would you like to know?",
            "That's intriguing! Can you elaborate?",
            "I can help with many things. Try asking about time, weather, math, searching, or opening websites!"
        ])
        
    except Exception as e:
        logging.error(f"Command processing error: {e}")
        return "An error occurred while processing your request. Please try again."

# ═══════════════════════════════════════════════════════════════════════════════
# HTTP SERVER WITH ENHANCED ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

HTML_CONTENT = None

class RequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler with CORS and better error handling"""
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass
    
    def _send_cors_headers(self):
        """Send CORS headers"""
        if CONFIG['cors_enabled']:
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests"""
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            if self.path in ['/', '/index.html']:
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self._send_cors_headers()
                self.end_headers()
                if HTML_CONTENT:
                    self.wfile.write(HTML_CONTENT.encode('utf-8'))
                else:
                    self.wfile.write(b'<h1>JARVIS Backend Running</h1><p>Frontend HTML not loaded.</p>')
            
            elif self.path == '/api/feed':
                frame = None
                detections = {'people': [], 'objects': []}
                
                if camera:
                    frame, detections = camera.get_frame_base64()
                
                data = {
                    'frame': frame,
                    'detections': detections,
                    'timestamp': time.time()
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(data).encode('utf-8'))
            
            elif self.path == '/api/stats':
                try:
                    cpu = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory()
                    
                    data = {
                        'cpu': round(cpu, 1),
                        'memory': round(memory.percent, 1),
                        'disk': round(psutil.disk_usage('/').percent, 1),
                        'timestamp': time.time()
                    }
                except Exception as e:
                    logging.error(f"Stats error: {e}")
                    data = {
                        'cpu': 0,
                        'memory': 0,
                        'disk': 0,
                        'timestamp': time.time()
                    }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(data).encode('utf-8'))
            
            elif self.path == '/api/diagnostics':
                # New endpoint for system diagnostics
                data = {
                    'tts': tts.get_stats(),
                    'camera': camera.get_stats() if camera else None,
                    'uptime': time.time() - server_start_time,
                    'modules': {
                        'tts': TTS_AVAILABLE,
                        'camera': CV2_AVAILABLE,
                        'face_recognition': FACE_RECOGNITION_AVAILABLE,
                        'wikipedia': WIKIPEDIA_AVAILABLE,
                        'search': DDGS_AVAILABLE,
                        'math': SYMPY_AVAILABLE,
                        'requests': REQUESTS_AVAILABLE
                    }
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(data).encode('utf-8'))
            
            else:
                self.send_response(404)
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(b'404 Not Found')
        
        except Exception as e:
            logging.error(f"GET error: {e}")
            self.send_response(500)
            self._send_cors_headers()
            self.end_headers()
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            if self.path == '/api/query':
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length).decode('utf-8')
                data = json.loads(body)
                
                query = data.get('query', '').strip()
                
                if not query:
                    response_text = "I didn't receive any command. Please try again."
                else:
                    # Process command
                    response_text = process_command(query)
                    
                    # Log command
                    print(f"\n{'='*60}")
                    print(f"[COMMAND] {query}")
                    print(f"[RESPONSE] {response_text[:100]}...")
                    print(f"{'='*60}")
                    
                    # Queue for TTS
                    if CONFIG['tts_enabled']:
                        tts.speak(response_text)
                
                # Return response with TTS info for frontend fallback
                result = {
                    'response': response_text,
                    'timestamp': time.time(),
                    'query': query,
                    'tts_text': tts._clean_text(response_text),
                    'backend_tts': tts.method is not None
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
            
            else:
                self.send_response(404)
                self._send_cors_headers()
                self.end_headers()
        
        except Exception as e:
            logging.error(f"POST error: {e}")
            self.send_response(500)
            self._send_cors_headers()
            self.end_headers()
            error_response = {
                'response': 'Server error occurred. Please try again.',
                'error': str(e),
                'tts_text': 'Server error occurred.',
                'backend_tts': False
            }
            self.wfile.write(json.dumps(error_response).encode('utf-8'))

# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP & MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def print_banner():
    """Print startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║          JARVIS Ultra Backend v3.0                          ║
║          Advanced AI Assistant System                        ║
║          🔧 ALL BUGS FIXED & ENHANCED                        ║
║          → Modular Architecture                              ║
║          → Enhanced Error Handling                           ║
║          → Better Performance                                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_dependencies():
    """Check and display dependency status"""
    print("\n[*] Checking Dependencies...")
    print("=" * 60)
    
    deps = [
        ("Core", "psutil", True),
        ("Text-to-Speech", "pyttsx3", TTS_AVAILABLE),
        ("Camera", "opencv-python (cv2)", CV2_AVAILABLE),
        ("Face Recognition", "face_recognition", FACE_RECOGNITION_AVAILABLE),
        ("Wikipedia", "wikipedia", WIKIPEDIA_AVAILABLE),
        ("Search", "duckduckgo-search", DDGS_AVAILABLE),
        ("Math", "sympy", SYMPY_AVAILABLE),
        ("Web Requests", "requests", REQUESTS_AVAILABLE)
    ]
    
    for category, package, available in deps:
        status = "✓" if available else "✗"
        color = "\033[92m" if available else "\033[91m"
        reset = "\033[0m"
        print(f"  {color}{status}{reset} {category:20} [{package}]")
    
    print("=" * 60)
    
    missing = [pkg for cat, pkg, avail in deps if not avail and pkg != "psutil"]
    if missing:
        print("\n[!] Missing Optional Packages. Install with:")
        print(f"    pip install {' '.join(missing)}")
        print()

def load_html_interface():
    """Load HTML interface file"""
    global HTML_CONTENT
    
    html_files = ['jarvis_ultra.html', 'jarvis_ultra_v3.html', 'jarvis_ui.html', 'index.html']
    
    for html_file in html_files:
        if os.path.exists(html_file):
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    HTML_CONTENT = f.read()
                print(f"[✓] Loaded HTML interface: {html_file}")
                return True
            except Exception as e:
                logging.error(f"Error loading {html_file}: {e}")
    
    print("[!] No HTML interface file found (optional)")
    print("    API endpoints will still work")
    return False

def main():
    """Main entry point"""
    global tts, camera, search_engine, server_start_time
    
    print_banner()
    check_dependencies()
    
    print("\n[*] Initializing JARVIS Systems...")
    
    # Initialize engines
    server_start_time = time.time()
    tts = TTSEngine()
    camera = CameraEngine() if CV2_AVAILABLE else None
    search_engine = SearchEngine()
    
    # Load HTML interface
    load_html_interface()
    
    # Setup face recognition
    if FACE_RECOGNITION_AVAILABLE and CV2_AVAILABLE:
        os.makedirs("known_faces", exist_ok=True)
        print("[*] Face recognition enabled. Add photos to 'known_faces/' folder")
    
    # Start server
    server_address = (CONFIG['host'], CONFIG['port'])
    
    try:
        httpd = HTTPServer(server_address, RequestHandler)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n[ERROR] Port {CONFIG['port']} is already in use!")
            print(f"        Try closing other JARVIS instances or change the port")
            return
        raise
    
    url = f"http://{CONFIG['host']}:{CONFIG['port']}"
    
    # Print status
    print(f"\n{'='*60}")
    print(f"[✓] JARVIS Backend Server Running")
    print(f"{'='*60}")
    print(f"\n    URL: {url}")
    print(f"\n    API Endpoints:")
    print(f"      • POST {url}/api/query         - Send commands")
    print(f"      • GET  {url}/api/feed          - Camera feed")
    print(f"      • GET  {url}/api/stats         - System stats")
    print(f"      • GET  {url}/api/diagnostics   - System diagnostics")
    print(f"\n    TTS Configuration:")
    if tts.method:
        print(f"      • Backend TTS: ✓ ENABLED ({tts.method})")
        print(f"      • Frontend TTS: ✓ FALLBACK")
        print(f"      • Mode: DUAL (Backend primary, Frontend backup)")
    else:
        print(f"      • Backend TTS: ✗ Not available")
        print(f"      • Frontend TTS: ✓ ENABLED (Primary)")
        print(f"      • Mode: FRONTEND ONLY")
    print(f"\n    Features:")
    print(f"      • Wikipedia Search: {'✓ Enabled' if WIKIPEDIA_AVAILABLE else '✗ Disabled'}")
    print(f"      • Web Search: {'✓ Enabled' if DDGS_AVAILABLE else '✗ Disabled'}")
    print(f"      • Camera Feed: {'✓ Enabled' if camera else '✗ Disabled'}")
    print(f"      • Face Recognition: {'✓ Enabled' if FACE_RECOGNITION_AVAILABLE else '✗ Disabled'}")
    print(f"      • Math Engine: {'✓ Enabled' if SYMPY_AVAILABLE else '✗ Disabled'}")
    print(f"\n{'='*60}")
    
    # Open browser
    if HTML_CONTENT:
        print(f"\n[*] Opening interface in browser...")
        time.sleep(0.5)
        try:
            webbrowser.open(url)
        except:
            print("[!] Could not open browser automatically")
            print(f"    Please open {url} manually")
    
    print(f"\n[*] Press Ctrl+C to stop the server")
    print(f"[*] Logging to: jarvis.log")
    print(f"[*] All systems operational!\n")
    
    # Welcome speech - delayed to avoid conflict with frontend
    if CONFIG['tts_enabled']:
        def delayed_welcome():
            time.sleep(3)  # Wait for frontend to load
            tts.speak("JARVIS systems online. All neural networks initialized. Ready for your commands.")
        
        welcome_thread = threading.Thread(target=delayed_welcome, daemon=True)
        welcome_thread.start()
    
    # Start server
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n[*] Shutting down JARVIS...")
        
        # Cleanup
        if camera:
            camera.stop()
        tts.stop()
        
        print("[✓] All systems offline. Goodbye!")
        if CONFIG['tts_enabled']:
            tts.speak("Systems shutting down. Goodbye.")
            time.sleep(1)

if __name__ == "__main__":
    main()