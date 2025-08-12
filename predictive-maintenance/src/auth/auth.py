from flask import session, request, jsonify, redirect, url_for
from functools import wraps
from werkzeug.security import check_password_hash, generate_password_hash
import secrets
import logging

logger = logging.getLogger(__name__)

class AuthManager:
    def __init__(self):
        self.users = {
            'admin': {
                'password_hash': generate_password_hash('admin123'),
                'role': 'admin'
            },
            'operator': {
                'password_hash': generate_password_hash('operator123'),
                'role': 'operator'
            }
        }
    
    def authenticate_user(self, username, password):
        user = self.users.get(username)
        if user and check_password_hash(user['password_hash'], password):
            return user
        return None
    
    def login(self, username, password):
        user = self.authenticate_user(username, password)
        if user:
            session['user_id'] = username
            session['role'] = user['role']
            session['csrf_token'] = secrets.token_hex(16)
            logger.info(f"User {username} logged in successfully")
            return True
        logger.warning(f"Failed login attempt for user {username}")
        return False
    
    def logout(self):
        user_id = session.get('user_id')
        session.clear()
        if user_id:
            logger.info(f"User {user_id} logged out")
        return True
    
    def is_authenticated(self):
        return 'user_id' in session
    
    def get_current_user(self):
        if self.is_authenticated():
            return {
                'username': session['user_id'],
                'role': session['role']
            }
        return None
    
    def has_role(self, required_role):
        if not self.is_authenticated():
            return False
        user_role = session.get('role')
        if required_role == 'admin':
            return user_role == 'admin'
        elif required_role == 'operator':
            return user_role in ['admin', 'operator']
        return False

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