import re
from typing import Dict, List, Optional

class PasswordValidator:
    @staticmethod
    def validate_password(password: str) -> Dict[str, bool]:
        checks = {
            'min_length': len(password) >= 8,
            'has_uppercase': bool(re.search(r'[A-Z]', password)),
            'has_lowercase': bool(re.search(r'[a-z]', password)),
            'has_digit': bool(re.search(r'\d', password)),
            'has_special': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password)),
            'no_common': password.lower() not in ['password', '123456', 'admin', 'qwerty']
        }
        return checks
    
    @staticmethod
    def is_strong_password(password: str) -> bool:
        checks = PasswordValidator.validate_password(password)
        return all(checks.values())
    
    @staticmethod
    def get_password_requirements() -> List[str]:
        return [
            "At least 8 characters long",
            "Contains uppercase letters",
            "Contains lowercase letters", 
            "Contains numbers",
            "Contains special characters",
            "Not a common password"
        ]

class InputValidator:
    @staticmethod
    def validate_username(username: str) -> bool:
        if not username or len(username) < 3 or len(username) > 50:
            return False
        return bool(re.match(r'^[a-zA-Z0-9_.-]+$', username))
    
    @staticmethod
    def validate_email(email: str) -> bool:
        if not email:
            return False
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        if not text:
            return ""
        return text.strip()[:255]