from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session
from .auth import AuthManager

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')
auth_manager = AuthManager()

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')
        else:
            username = request.form.get('username')
            password = request.form.get('password')
        
        if auth_manager.login(username, password):
            if request.is_json:
                return jsonify({
                    'success': True,
                    'user': auth_manager.get_current_user()
                })
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            if request.is_json:
                return jsonify({'error': 'Invalid credentials'}), 401
            flash('Invalid username or password', 'error')
    
    return render_template('auth/login.html')

@auth_bp.route('/logout', methods=['POST', 'GET'])
def logout():
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
            'user': auth_manager.get_current_user()
        })
    return jsonify({'authenticated': False})

@auth_bp.route('/check')
def check():
    return jsonify({
        'authenticated': auth_manager.is_authenticated(),
        'csrf_token': session.get('csrf_token')
    })