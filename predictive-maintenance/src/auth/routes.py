from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session
from .auth import AuthManager
from .validators import InputValidator, PasswordValidator
import logging

logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')
auth_manager = AuthManager()

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json() or {}
            username = data.get('username', '').strip()
            password = data.get('password', '')
            csrf_token = data.get('csrf_token', '')
        else:
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')
            csrf_token = request.form.get('csrf_token', '')
        
        if not username or not password:
            error_msg = 'Username and password are required'
            if request.is_json:
                return jsonify({'error': error_msg}), 400
            flash(error_msg, 'error')
            return render_template('auth/login.html')
        
        if not InputValidator.validate_username(username):
            error_msg = 'Invalid username format'
            if request.is_json:
                return jsonify({'error': error_msg}), 400
            flash(error_msg, 'error')
            return render_template('auth/login.html')
        
        ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        result = auth_manager.login(username, password, ip_address)
        
        if result['success']:
            if request.is_json:
                return jsonify(result)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            if request.is_json:
                return jsonify(result), 401
            flash(result.get('error', 'Login failed'), 'error')
    
    return render_template('auth/login.html')

@auth_bp.route('/logout', methods=['POST', 'GET'])
def logout():
    csrf_token = request.json.get('csrf_token') if request.is_json else request.form.get('csrf_token')
    
    if request.method == 'POST' and not auth_manager.verify_csrf_token(csrf_token):
        if request.is_json:
            return jsonify({'error': 'Invalid CSRF token'}), 403
        flash('Invalid request', 'error')
        return redirect(url_for('auth.login'))
    
    auth_manager.logout()
    if request.is_json:
        return jsonify({'success': True})
    flash('You have been logged out', 'info')
    return redirect(url_for('auth.login'))

@auth_bp.route('/status')
def status():
    if auth_manager.is_authenticated():
        return jsonify({
            'authenticated': True,
            'user': auth_manager.get_current_user(),
            'csrf_token': session.get('csrf_token')
        })
    return jsonify({'authenticated': False})

@auth_bp.route('/refresh', methods=['POST'])
def refresh_token():
    data = request.get_json() or {}
    refresh_token = data.get('refresh_token')
    
    if not refresh_token:
        return jsonify({'error': 'Refresh token required'}), 400
    
    new_access_token = auth_manager.refresh_access_token(refresh_token)
    if new_access_token:
        return jsonify({'access_token': new_access_token})
    
    return jsonify({'error': 'Invalid or expired refresh token'}), 401

@auth_bp.route('/validate-password', methods=['POST'])
def validate_password():
    data = request.get_json() or {}
    password = data.get('password', '')
    
    checks = PasswordValidator.validate_password(password)
    requirements = PasswordValidator.get_password_requirements()
    
    return jsonify({
        'valid': PasswordValidator.is_strong_password(password),
        'checks': checks,
        'requirements': requirements
    })