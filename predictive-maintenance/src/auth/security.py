import time
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Optional

class RateLimiter:
    def __init__(self):
        self.attempts = defaultdict(list)
        self.blocked = defaultdict(float)
    
    def is_rate_limited(self, identifier: str, max_attempts: int = 5, window_minutes: int = 15) -> bool:
        now = time.time()
        
        if identifier in self.blocked and now < self.blocked[identifier]:
            return True
        
        cutoff = now - (window_minutes * 60)
        self.attempts[identifier] = [attempt for attempt in self.attempts[identifier] if attempt > cutoff]
        
        if len(self.attempts[identifier]) >= max_attempts:
            self.blocked[identifier] = now + (window_minutes * 60)
            return True
        
        return False
    
    def record_attempt(self, identifier: str):
        self.attempts[identifier].append(time.time())

class TokenManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.algorithm = 'HS256'
    
    def generate_access_token(self, user_data: Dict, expires_hours: int = 1) -> str:
        payload = {
            'user_id': user_data['id'],
            'username': user_data['username'],
            'role': user_data['role'],
            'exp': datetime.utcnow() + timedelta(hours=expires_hours),
            'iat': datetime.utcnow(),
            'type': 'access'
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def generate_refresh_token(self, user_data: Dict, expires_days: int = 30) -> str:
        payload = {
            'user_id': user_data['id'],
            'username': user_data['username'],
            'exp': datetime.utcnow() + timedelta(days=expires_days),
            'iat': datetime.utcnow(),
            'type': 'refresh'
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

class CSRFProtection:
    @staticmethod
    def generate_csrf_token() -> str:
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def validate_csrf_token(token: str, session_token: str) -> bool:
        if not token or not session_token:
            return False
        return secrets.compare_digest(token, session_token)

class SecureHasher:
    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple:
        if salt is None:
            salt = secrets.token_hex(32)
        
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return key.hex(), salt
    
    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return secrets.compare_digest(key.hex(), hashed)