import { state } from './state.js';
import { showNotification } from './notifications.js';

export function toggleLoginModal(open) {
  const shouldOpen = Boolean(open);
  const loginModal = document.getElementById('login-modal');
  loginModal.classList.toggle('open', shouldOpen);
  loginModal.setAttribute('aria-hidden', shouldOpen ? 'false' : 'true');
  document.body.classList.toggle('modal-open', shouldOpen);
  if (shouldOpen) switchAuthForm('login');
}

export function switchAuthForm(formType) {
  const loginForm = document.getElementById('login-form');
  const registerForm = document.getElementById('register-form');
  if (formType === 'login') {
    loginForm.classList.add('active');
    registerForm.classList.remove('active');
  } else {
    loginForm.classList.remove('active');
    registerForm.classList.add('active');
  }
}

export function updateLoginButton() {
  const btn = document.getElementById('login-btn');
  const token = localStorage.getItem('authToken');
  if (token) {
    btn.innerHTML = '<i class="fas fa-sign-out-alt"></i> Logout';
    btn.onclick = handleLogout;
  } else {
    btn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Login';
    btn.onclick = () => toggleLoginModal(true);
  }
}

export function handleLogout() {
  localStorage.removeItem('authToken');
  localStorage.removeItem('userId');
  updateLoginButton();
}

export function handleLogin(event) {
  event.preventDefault();
  const email = document.getElementById('login-email').value;
  const password = document.getElementById('login-password').value;

  fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.success) {
        if (data.token) localStorage.setItem('authToken', data.token);
        if (data.user_id) localStorage.setItem('userId', data.user_id);
        toggleLoginModal(false);
        updateLoginButton();
        document.getElementById('login-email').value = '';
        document.getElementById('login-password').value = '';

        import('./ui/camera.js').then(({ connectCamera }) => {
          if (state.ws) {
            const oldWs = state.ws;
            state.ws = null;
            oldWs.onclose = null;
            oldWs.close();
            connectCamera();
          }
        });

        if (state.currentDetectedCategories.length > 0) {
          import('./api.js').then(({ initializeSearchSession }) => {
            initializeSearchSession(state.currentDetectedCategories, []);
          });
        }
      } else {
        showNotification('error', 'Login Failed', data.message || 'Unknown error', null, 'OK', 4000);
      }
    })
    .catch((error) => {
      showNotification('error', 'Login Error', error.message || 'Could not reach server.', null, 'OK', 4000);
    });
}

export function handleRegister(event) {
  event.preventDefault();
  const name = document.getElementById('register-name').value;
  const email = document.getElementById('register-email').value;
  const password = document.getElementById('register-password').value;
  const confirm = document.getElementById('register-confirm').value;

  if (password !== confirm) {
    showNotification('warning', "Passwords Don't Match", 'Please verify both password fields and try again.', null, "I'll Fix It", 0);
    return;
  }

  fetch('/api/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, email, password }),
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.success) {
        if (data.token) localStorage.setItem('authToken', data.token);
        if (data.user_id) localStorage.setItem('userId', data.user_id);
        showNotification(
          'success',
          'Registration Complete!',
          'Your account has been created. You can now log in.',
          () => {
            switchAuthForm('login');
            ['register-name', 'register-email', 'register-password', 'register-confirm'].forEach(
              (id) => (document.getElementById(id).value = '')
            );
          },
          'Go to Login',
          0
        );
      } else {
        showNotification('error', 'Registration Failed', data.message || 'An error occurred. Please try again.', null, 'OK', 4000);
      }
    })
    .catch((error) => {
      showNotification('error', 'Registration Error', error.message || 'An error occurred.', null, 'OK', 4000);
    });
}
