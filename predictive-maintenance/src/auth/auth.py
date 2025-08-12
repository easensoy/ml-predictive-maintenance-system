from flask import session, request, jsonify, redirect, url_for
from functools import wraps
import secrets
import logging
from .validators import InputValidator, PasswordValidator
from .security import RateLimiter, TokenManager, CSRFProtection, SecureHasher

logger = logging.getLogger(__name__)

class AuthManager:
    def __init__(self, secret_key: str = None):
        self.rate_limiter = RateLimiter()
        self.token_manager = TokenManager(secret_key or secrets.token_urlsafe(32))
        self.users = self._init_default_users()
    
    def _init_default_users(self):
        admin_hash, admin_salt = SecureHasher.hash_password('Admin@123!')
        operator_hash, operator_salt = SecureHasher.hash_password('Operator@123!')
        
        return {
            'admin': {
                'id': 1,
                'password_hash': admin_hash,
                'salt': admin_salt,
                'role': 'admin',
                'is_active': True,
                'failed_attempts': 0
            },
            'operator': {
                'id': 2,
                'password_hash': operator_hash, 
                'salt': operator_salt,
                'role': 'operator',
                'is_active': True,
                'failed_attempts': 0
            }
        }
    
    def authenticate_user(self, username: str, password: str, ip_address: str = None):
        username = InputValidator.sanitize_input(username)
        
        if not InputValidator.validate_username(username):
            return None
        
        identifier = f"{username}:{ip_address}" if ip_address else username
        if self.rate_limiter.is_rate_limited(identifier):
            logger.warning(f"Rate limited login attempt for {username} from {ip_address}")
            return None
        
        self.rate_limiter.record_attempt(identifier)
        
        user = self.users.get(username)
        if not user or not user['is_active']:
            logger.warning(f"Login attempt for inactive/nonexistent user: {username}")
            return None
        
        if SecureHasher.verify_password(password, user['password_hash'], user['salt']):
            user['failed_attempts'] = 0
            logger.info(f"Successful authentication for user: {username}")
            return user
        
        user['failed_attempts'] += 1
        if user['failed_attempts'] >= 5:
            user['is_active'] = False
            logger.warning(f"User {username} locked due to failed attempts")
        
        logger.warning(f"Failed authentication for user: {username}")
        return None
    
    def login(self, username: str, password: str, ip_address: str = None):
        user = self.authenticate_user(username, password, ip_address)
        if user:
            session['user_id'] = username
            session['role'] = user['role']
            session['csrf_token'] = CSRFProtection.generate_csrf_token()
            session['login_time'] = secrets.token_hex(8)
            
            access_token = self.token_manager.generate_access_token(user)
            refresh_token = self.token_manager.generate_refresh_token(user)
            
            return {
                'success': True,
                'access_token': access_token,
                'refresh_token': refresh_token,
                'user': {
                    'username': user.get('username', username),
                    'role': user['role']
                }
            }
        return {'success': False, 'error': 'Invalid credentials or account locked'}
    
    def logout(self):
        user_id = session.get('user_id')
        session.clear()
        if user_id:
            logger.info(f"User {user_id} logged out")
        return True
    
    def is_authenticated(self):
        return 'user_id' in session and 'login_time' in session
    
    def get_current_user(self):
        if self.is_authenticated():
            username = session['user_id']
            user = self.users.get(username)
            if user:
                return {
                    'username': username,
                    'role': user['role']
                }
        return None
    
    def has_role(self, required_role: str):
        if not self.is_authenticated():
            return False
        user_role = session.get('role')
        if required_role == 'admin':
            return user_role == 'admin'
        elif required_role == 'operator':
            return user_role in ['admin', 'operator']
        return False
    
    def verify_csrf_token(self, token: str):
        session_token = session.get('csrf_token')
        return CSRFProtection.validate_csrf_token(token, session_token)
    
    def refresh_access_token(self, refresh_token: str):
        payload = self.token_manager.verify_token(refresh_token)
        if payload and payload.get('type') == 'refresh':
            username = payload['username']
            user = self.users.get(username)
            if user and user['is_active']:
                return self.token_manager.generate_access_token(user)
        return None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user_id'):
            if request.is_json:
                return jsonify({'error': 'Authentication required'}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_manager = AuthManager()
            if not auth_manager.has_role(role):
                if request.is_json:
                    return jsonify({'error': 'Insufficient permissions'}), 403
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator