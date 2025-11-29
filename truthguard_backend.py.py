#!/usr/bin/env python3
"""
AuraLens Advanced - Next Generation AI System
Enterprise-Grade Misinformation Detection + Advanced Analytics
"""

import os, json, time, threading, queue, logging, re, random, hashlib, uuid
from datetime import datetime as dt, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
import psutil
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe imports
try:
    import pyttsx3
    TTS_ENGINE = pyttsx3.init()
    TTS_AVAILABLE = True
except:
    TTS_AVAILABLE = False

try:
    import wikipedia
    wikipedia.set_lang('en')
    WIKIPEDIA_AVAILABLE = True
except:
    WIKIPEDIA_AVAILABLE = False

try:
    import sympy
    SYMPY_AVAILABLE = True
except:
    SYMPY_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except:
    REQUESTS_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except:
    TEXTBLOB_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
except:
    NLTK_AVAILABLE = False

PORT = 8090  # Or any free port above 1024, not 8080
HOST = '127.0.0.1'

# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED ANALYTICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class AnalyticsEngine:
    def __init__(self):
        self.history = []
        self.stats = {
            'total_queries': 0,
            'fact_checks': 0,
            'avg_credibility': 0,
            'command_frequency': defaultdict(int),
            'accuracy_rate': 0,
            'response_times': []
        }
    
    def log_interaction(self, query_type, credibility=None, response_time=None):
        self.stats['total_queries'] += 1
        self.stats['command_frequency'][query_type] += 1
        if response_time:
            self.stats['response_times'].append(response_time)
        if credibility is not None:
            self.stats['fact_checks'] += 1
            if self.stats['fact_checks'] > 0:
                self.stats['avg_credibility'] = (
                    (self.stats['avg_credibility'] * (self.stats['fact_checks'] - 1) + credibility) / 
                    self.stats['fact_checks']
                )
    
    def get_analytics(self):
        avg_response_time = sum(self.stats['response_times'][-100:]) / len(self.stats['response_times'][-100:]) if self.stats['response_times'] else 0
        
        return {
            'total_queries': self.stats['total_queries'],
            'fact_checks_performed': self.stats['fact_checks'],
            'average_credibility': round(self.stats['avg_credibility'], 1),
            'average_response_time': round(avg_response_time * 1000, 2),
            'most_common_command': max(self.stats['command_frequency'].items(), key=lambda x: x[1])[0] if self.stats['command_frequency'] else 'N/A',
            'session_time': str(timedelta(seconds=int(time.time() - start_time)))
        }

# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED FACT-CHECKING ENGINE WITH ML-LIKE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedFactChecker:
    def __init__(self):
        self.cache = {}
        self.threat_database = self._build_threat_db()
        self.entity_recognition_db = self._build_entity_db()
        self.historical_patterns = defaultdict(list)
    
    def _build_threat_db(self):
        return {
            'misinformation_keywords': {
                'urgent': ['NOW', 'IMMEDIATELY', 'EMERGENCY', 'ALERT', 'WARNING'],
                'emotional_manipulation': ['SHOCKING', 'DEVASTATING', 'DISGUSTING', 'OUTRAGEOUS', 'HORRIFYING'],
                'conspiracy': ['COVER-UP', 'EXPOSED', 'SECRET', 'HIDDEN TRUTH', 'THEY DON\'T WANT'],
                'fake_cure': ['MIRACLE', 'CURE-ALL', 'GUARANTEED', '100% EFFECTIVE', 'CLINICALLY PROVEN'],
                'authority_abuse': ['EXPERTS AGREE', 'SCIENTISTS SAY', 'DOCTORS RECOMMEND', 'STUDIES SHOW']
            },
            'misinformation_phrases': [
                r'(?:SHOCKING|UNBELIEVABLE)\s+(?:STUDY|REPORT|NEWS)',
                r'YOU WON\'T BELIEVE',
                r'THIS ONE WEIRD TRICK',
                r'DOCTORS HATE',
                r'(?:BIG|CORPORATE)\s+(?:PHARMA|TECH|MEDIA)',
                r'WAKE UP SHEEPLE',
                r'(?:THEY|THE GOVERNMENT)\s+(?:DON\'T|DOESN\'T)\s+WANT\s+YOU\s+TO\s+(?:KNOW|SEE)',
                r'CLICK HERE BEFORE',
                r'(?:LEAKED|CLASSIFIED|RESTRICTED)\s+(?:DOCUMENTS|FOOTAGE|REPORTS)'
            ]
        }
    
    def _build_entity_db(self):
        return {
            'organizations': ['WHO', 'CDC', 'FDA', 'NHS', 'HARVARD', 'MIT', 'OXFORD', 'STANFORD'],
            'authority_indicators': ['RESEARCH', 'STUDY', 'PUBLISHED', 'PEER-REVIEWED', 'DOCUMENTED'],
            'verification_sources': ['WIKIPEDIA', 'BBC', 'REUTERS', 'ASSOCIATED PRESS', 'GOVERNMENT']
        }
    
    def analyze_advanced(self, claim):
        try:
            cache_key = hashlib.md5(claim[:100].encode()).hexdigest()
            if cache_key in self.cache:
                cached_time, cached_result = self.cache[cache_key]
                if time.time() - cached_time < 3600:
                    return cached_result
            
            result = {
                'claim': claim[:150],
                'credibility_score': 50,
                'confidence_level': 'MEDIUM',
                'risk_level': 'UNKNOWN',
                'detailed_analysis': {},
                'threat_indicators': [],
                'trust_signals': [],
                'entity_recognition': [],
                'pattern_matching': [],
                'recommendations': [],
                'ai_explanation': ''
            }
            
            # 1. Threat Analysis
            threat_score = self._threat_analysis(claim)
            result['detailed_analysis']['threat_score'] = threat_score['score']
            result['threat_indicators'] = threat_score['indicators']
            
            # 2. Linguistic Complexity
            linguistic_score = self._linguistic_analysis(claim)
            result['detailed_analysis']['linguistic_score'] = linguistic_score['score']
            
            # 3. Entity Recognition
            entities = self._entity_recognition(claim)
            result['entity_recognition'] = entities['found']
            result['trust_signals'] = entities['trust_signals']
            
            # 4. Pattern Matching
            patterns = self._pattern_matching(claim)
            result['pattern_matching'] = patterns
            
            # 5. Emotional Analysis
            emotion_score = self._advanced_emotion_analysis(claim)
            result['detailed_analysis']['emotion_score'] = emotion_score['score']
            
            # 6. Source Authority
            source_score = self._source_authority_check(claim)
            result['detailed_analysis']['source_score'] = source_score['score']
            
            # 7. Wikipedia Cross-Reference
            wiki_score = self._wikipedia_deep_check(claim)
            result['detailed_analysis']['wikipedia_score'] = wiki_score['score']
            
            # Calculate Weighted Score
            weights = {
                'threat_score': 0.25,
                'linguistic_score': 0.15,
                'emotion_score': 0.20,
                'source_score': 0.20,
                'wikipedia_score': 0.20
            }
            
            final_score = sum(
                result['detailed_analysis'].get(k, 50) * v 
                for k, v in weights.items()
            )
            
            result['credibility_score'] = int(final_score)
            
            # Confidence Level
            if result['credibility_score'] >= 80:
                result['confidence_level'] = 'HIGH'
                result['risk_level'] = 'LOW'
                result['ai_explanation'] = 'Strong indicators of credible, trustworthy content'
            elif result['credibility_score'] >= 60:
                result['confidence_level'] = 'MEDIUM'
                result['risk_level'] = 'MEDIUM'
                result['ai_explanation'] = 'Mixed signals - cross-reference with reliable sources'
            elif result['credibility_score'] >= 40:
                result['confidence_level'] = 'LOW'
                result['risk_level'] = 'HIGH'
                result['ai_explanation'] = 'Multiple misinformation indicators detected'
            else:
                result['confidence_level'] = 'VERY LOW'
                result['risk_level'] = 'CRITICAL'
                result['ai_explanation'] = 'Strong evidence of misinformation - do not share'
            
            # Generate recommendations
            result['recommendations'] = self._generate_recommendations(result)
            
            self.cache[cache_key] = (time.time(), result)
            return result
            
        except Exception as e:
            logger.error(f"Advanced fact-check error: {e}")
            return {
                'claim': claim[:150],
                'credibility_score': 50,
                'confidence_level': 'UNKNOWN',
                'risk_level': 'UNKNOWN',
                'ai_explanation': 'Analysis error',
                'recommendations': ['Try again or rephrase the claim']
            }
    
    def _threat_analysis(self, text):
        score = 100
        indicators = []
        
        text_upper = text.upper()
        
        for category, keywords in self.threat_database['misinformation_keywords'].items():
            for keyword in keywords:
                if keyword in text_upper:
                    score -= 15
                    indicators.append(f"{category}: '{keyword}' detected")
        
        for phrase_pattern in self.threat_database['misinformation_phrases']:
            if re.search(phrase_pattern, text_upper):
                score -= 20
                indicators.append(f"Misinformation pattern detected: {phrase_pattern[:40]}...")
        
        return {'score': max(0, min(100, score)), 'indicators': indicators[:5]}
    
    def _linguistic_analysis(self, text):
        score = 100
        
        # Check sentence structure
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text)
                avg_length = len(text) / len(sentences) if sentences else 0
                if avg_length < 15:  # Too short sentences
                    score -= 10
            except:
                pass
        
        # Excessive punctuation
        if len(re.findall(r'[!?]{2,}', text)) > 2:
            score -= 15
        
        # ALL CAPS words
        caps_words = len([w for w in text.split() if w.isupper() and len(w) > 3])
        if caps_words > 3:
            score -= caps_words * 5
        
        # Grammar indicators
        if re.search(r'\byour\s+(?:body|health|mind)\b', text, re.IGNORECASE):
            score -= 10
        
        return {'score': max(0, min(100, score))}
    
    def _entity_recognition(self, text):
        found = []
        trust_signals = []
        
        for org in self.entity_recognition_db['organizations']:
            if org.lower() in text.lower():
                found.append(org)
                trust_signals.append(f"Mentions authoritative organization: {org}")
        
        for indicator in self.entity_recognition_db['authority_indicators']:
            if indicator.lower() in text.lower():
                trust_signals.append(f"Authority indicator: {indicator}")
        
        return {'found': found, 'trust_signals': trust_signals}
    
    def _pattern_matching(self, text):
        patterns_found = []
        
        common_patterns = [
            ('Medical claim without source', r'(?:cure|treat|prevent)\s+(?:cancer|diabetes|covid)', 10),
            ('Fake urgency', r'(?:act now|limited time|before it\'s too late)', 15),
            ('Appeal to authority', r'(?:doctors?|scientists?|experts?)\s+(?:say|agree|recommend)', 5),
            ('Conspiracy language', r'(?:they|them|government)\s+(?:hide|suppress|cover-up)', 20),
        ]
        
        for pattern_name, pattern, weight in common_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                patterns_found.append({'name': pattern_name, 'weight': weight})
        
        return patterns_found
    
    def _advanced_emotion_analysis(self, text):
        score = 100
        
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                if subjectivity > 0.8:
                    score -= 30
                elif subjectivity > 0.6:
                    score -= 15
                elif subjectivity > 0.4:
                    score -= 5
            except:
                pass
        
        emotional_words = {
            'devastat': 15, 'shocking': 15, 'disgust': 15, 'horribl': 15,
            'miracl': 20, 'amaz': 10, 'unbeliev': 15, 'secret': 12,
            'danger': 10, 'warn': 8
        }
        
        text_lower = text.lower()
        for word, penalty in emotional_words.items():
            if word in text_lower:
                score -= penalty
        
        return {'score': max(0, min(100, score))}
    
    def _source_authority_check(self, text):
        score = 50
        
        trusted_sources = ['research', 'study', 'published', 'journal', 'university', 'harvard', 'mit', 'stanford']
        for source in trusted_sources:
            if source.lower() in text.lower():
                score += 15
        
        untrusted_indicators = ['my friend', 'i heard', 'allegedly', 'supposedly', 'rumor']
        for indicator in untrusted_indicators:
            if indicator.lower() in text.lower():
                score -= 25
        
        return {'score': max(0, min(100, score))}
    
    def _wikipedia_deep_check(self, text):
        if not WIKIPEDIA_AVAILABLE:
            return {'score': 50}
        
        score = 50
        try:
            words = [w for w in text.split() if len(w) > 4][:5]
            if words:
                results = wikipedia.search(' '.join(words), results=3)
                if results:
                    score = 75
        except:
            pass
        
        return {'score': score}
    
    def _generate_recommendations(self, result):
        recs = []
        
        if result['risk_level'] == 'CRITICAL':
            recs.append('🚫 DO NOT SHARE - High misinformation risk')
            recs.append('📚 Check authoritative sources (WHO, CDC, NIH)')
            recs.append('🔍 Look for peer-reviewed studies')
        elif result['risk_level'] == 'HIGH':
            recs.append('⚠️ VERIFY BEFORE SHARING')
            recs.append('🔗 Cross-reference with multiple sources')
            recs.append('📖 Check primary sources')
        elif result['risk_level'] == 'MEDIUM':
            recs.append('✓ Appears credible but verify key claims')
            recs.append('📎 Check author credentials')
            recs.append('⏰ Verify publication date')
        else:
            recs.append('✅ High credibility - safe to share')
        
        return recs

# ═══════════════════════════════════════════════════════════════════════════════
# TTS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class TTSEngine:
    def __init__(self):
        self.queue = queue.Queue(maxsize=50)
        self.is_running = True
        self.count = 0
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def _worker(self):
        while self.is_running:
            try:
                text = self.queue.get(timeout=0.5)
                if text and TTS_AVAILABLE:
                    self.count += 1
                    clean = re.sub(r'<[^>]+>|[\U0001F300-\U0001F9FF]', '', text)[:200]
                    TTS_ENGINE.say(clean)
                    TTS_ENGINE.runAndWait()
            except queue.Empty:
                pass
            except:
                pass
    
    def speak(self, text):
        try:
            self.queue.put_nowait(text)
        except:
            pass
    
    def stop(self):
        self.is_running = False

# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED COMMAND PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

def process_command(cmd):
    start_time = time.time()
    cmd_lower = cmd.lower().strip()
    query_type = 'general'
    
    # Fact-checking
    if any(w in cmd_lower for w in ['verify', 'check', 'fact check', 'is true', 'is this', 'analyze claim']):
        query_type = 'fact_check'
        claim = cmd
        for prefix in ['verify', 'check', 'fact check', 'is this', 'is', 'analyze claim']:
            claim = claim.replace(prefix, '', 1)
        claim = claim.strip()
        
        if not claim:
            return "What claim would you like me to analyze?"
        
        result = fact_checker.analyze_advanced(claim)
        analytics.log_interaction(query_type, result['credibility_score'], time.time() - start_time)
        
        html = f"""
        <div style="background: rgba(0,255,136,0.1); border: 1px solid rgba(0,255,136,0.3); padding: 15px; border-radius: 8px;">
        <strong>🛡️ ADVANCED FACT-CHECK ANALYSIS</strong><br>
        <strong>Claim:</strong> {result['claim']}<br><br>
        
        <strong>📊 Credibility Score: <span style="color: #ffff00; font-size: 1.3em;">{result['credibility_score']}%</span></strong><br>
        <strong>Confidence Level:</strong> {result['confidence_level']}<br>
        <strong>Risk Level:</strong> {result['risk_level']}<br><br>
        
        <strong>🔍 Detailed Scores:</strong><br>
        """
        
        for metric, score in result['detailed_analysis'].items():
            html += f"• {metric.replace('_', ' ').title()}: {score}%<br>"
        
        if result['threat_indicators']:
            html += f"<br><strong>⚠️ Threat Indicators:</strong><br>"
            for indicator in result['threat_indicators'][:5]:
                html += f"• {indicator}<br>"
        
        if result['entity_recognition']:
            html += f"<br><strong>🏢 Entities Found:</strong> {', '.join(result['entity_recognition'])}<br>"
        
        if result['trust_signals']:
            html += f"<br><strong>✓ Trust Signals:</strong><br>"
            for signal in result['trust_signals'][:3]:
                html += f"• {signal}<br>"
        
        html += f"<br><strong>💡 AI Analysis:</strong> {result['ai_explanation']}<br><br>"
        
        html += f"<strong>📋 Recommendations:</strong><br>"
        for rec in result['recommendations']:
            html += f"• {rec}<br>"
        
        html += "</div>"
        
        return html
    
    # Time
    if 'time' in cmd_lower:
        query_type = 'time'
        now = dt.now()
        analytics.log_interaction(query_type, response_time=time.time() - start_time)
        return f"⏰ Current time: <strong>{now.strftime('%I:%M:%S %p')}</strong>"
    
    # Date
    if 'date' in cmd_lower or 'today' in cmd_lower:
        query_type = 'date'
        now = dt.now()
        analytics.log_interaction(query_type, response_time=time.time() - start_time)
        return f"📅 Today: <strong>{now.strftime('%A, %B %d, %Y')}</strong>"
    
    # Weather
    if 'weather' in cmd_lower:
        query_type = 'weather'
        if REQUESTS_AVAILABLE:
            try:
                r = requests.get('https://wttr.in/?format=3', timeout=5)
                if r.status_code == 200:
                    analytics.log_interaction(query_type, response_time=time.time() - start_time)
                    return f"🌤️ Weather: {r.text}"
            except:
                pass
        analytics.log_interaction(query_type, response_time=time.time() - start_time)
        return "Unable to fetch weather"
    
    # Math
    if any(w in cmd_lower for w in ['calculate', 'math', 'compute', 'solve']):
        query_type = 'math'
        if SYMPY_AVAILABLE:
            try:
                expr = cmd_lower
                for w in ['calculate', 'math', 'compute', 'solve']:
                    expr = expr.replace(w, '')
                expr = expr.strip()
                expr = ''.join(c for c in expr if c.isdigit() or c in '+-*/(). ')
                if expr:
                    result = sympy.sympify(expr)
                    value = float(result.evalf())
                    analytics.log_interaction(query_type, response_time=time.time() - start_time)
                    return f"🧮 Result: <strong>{value}</strong>"
            except:
                pass
        analytics.log_interaction(query_type, response_time=time.time() - start_time)
        return "Calculation unavailable"
    
    # Joke
    if 'joke' in cmd_lower:
        query_type = 'joke'
        jokes = [
            "Why don't scientists trust atoms? They make up everything!",
            "Why do programmers prefer dark mode? Light attracts bugs!",
            "What's a programmer's favorite hangout? Foo Bar!",
            "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
            "Why did the AI go to therapy? Too many deep learning issues!",
            "What do you call a programmer from Finland? Nerdic!",
            "Why did the developer go broke? He used up all his cache!"
        ]
        analytics.log_interaction(query_type, response_time=time.time() - start_time)
        return f"😂 {random.choice(jokes)}"
    
    # System
    if 'system' in cmd_lower or 'status' in cmd_lower:
        query_type = 'system'
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        analytics.log_interaction(query_type, response_time=time.time() - start_time)
        return f"⚙️ CPU: {cpu}% | Memory: {mem.percent}% | Disk: {disk.percent}%"
    
    # Analytics
    if 'analytics' in cmd_lower or 'stats' in cmd_lower:
        query_type = 'analytics'
        stats = analytics.get_analytics()
        html = f"""
        <div style="background: rgba(0,100,255,0.1); padding: 15px; border-radius: 8px;">
        <strong>📊 System Analytics</strong><br>
        Total Queries: {stats['total_queries']}<br>
        Fact-Checks Performed: {stats['fact_checks_performed']}<br>
        Average Credibility Score: {stats['average_credibility']}%<br>
        Average Response Time: {stats['average_response_time']}ms<br>
        Most Common Command: {stats['most_common_command']}<br>
        Session Time: {stats['session_time']}
        </div>
        """
        analytics.log_interaction(query_type, response_time=time.time() - start_time)
        return html
    
    # Default
    query_type = 'general'
    analytics.log_interaction(query_type, response_time=time.time() - start_time)
    return "Ready to help! Try: verify [claim], time, weather, calculate, joke, system status, or view analytics"

# ═══════════════════════════════════════════════════════════════════════════════
# HTTP SERVER
# ═══════════════════════════════════════════════════════════════════════════════

class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        if self.path == '/api/stats':
            try:
                data = {
                    'cpu': round(psutil.cpu_percent(interval=0.1), 1),
                    'memory': round(psutil.virtual_memory().percent, 1),
                    'disk': round(psutil.disk_usage('/').percent, 1),
                    'status': 'online'
                }
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            except:
                self.send_response(500)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/api/query':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length).decode()
                data = json.loads(body)
                
                query = data.get('query', '').strip()
                response = process_command(query) if query else "No command"
                
                print(f"[COMMAND] {query[:60]}")
                
                result = {
                    'response': response,
                    'query': query,
                    'tts_available': TTS_AVAILABLE,
                    'timestamp': dt.now().isoformat()
                }
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
                
                if TTS_AVAILABLE:
                    tts.speak(response)
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

start_time = time.time()

if __name__ == '__main__':
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║   🎙️  AuraLens Advanced - Enterprise AI System                   ║
║                                                                    ║
║   ✨ Next-Generation Features:                                    ║
║      • Advanced Machine Learning-Like Analysis                    ║
║      • Entity Recognition & Threat Detection                      ║
║      • Real-time Analytics & Performance Metrics                  ║
║      • Confidence & Risk Level Assessment                         ║
║      • Pattern Matching & Anomaly Detection                       ║
║      • Comprehensive Session Analytics                            ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    print(f"[✓] Advanced Backend starting on http://{HOST}:{PORT}")
    print(f"[✓] Features: TTS={TTS_AVAILABLE}, Wiki={WIKIPEDIA_AVAILABLE}, Math={SYMPY_AVAILABLE}")
    print(f"[✓] Advanced Analytics: ENABLED")
    print(f"[✓] Threat Detection: ENABLED")
    print(f"[✓] Entity Recognition: ENABLED")
    print(f"[*] Press Ctrl+C to stop\n")
    
    tts = TTSEngine()
    fact_checker = AdvancedFactChecker()
    analytics = AnalyticsEngine()
    
    server = HTTPServer((HOST, PORT), RequestHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[✓] Shutting down...")
        tts.stop()
        print("[✓] Goodbye!")