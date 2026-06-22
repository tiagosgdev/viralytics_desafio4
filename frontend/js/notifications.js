export function showNotification(
  type = 'success',
  title = '',
  message = '',
  onAction = null,
  actionText = 'OK',
  autoDismiss = 3000
) {
  const shell = document.getElementById('notification-shell');
  const modal = document.createElement('div');
  modal.className = `notification-modal ${type}`;
  modal.setAttribute('role', 'alert');
  modal.setAttribute('aria-live', 'assertive');

  const iconMap = {
    success: '<i class="fas fa-check-circle"></i>',
    error: '<i class="fas fa-exclamation-circle"></i>',
    warning: '<i class="fas fa-exclamation-triangle"></i>',
  };

  modal.innerHTML = `
    <button class="notification-close-btn" aria-label="Dismiss notification">
      <i class="fas fa-times"></i>
    </button>
    <div class="notification-header">
      <div class="notification-icon">${iconMap[type] || iconMap['success']}</div>
      <div class="notification-content">
        ${title ? `<h3 class="notification-title">${title}</h3>` : ''}
        <p class="notification-message">${message}</p>
        ${
          onAction
            ? `<div class="notification-action">
            <button class="notification-btn notification-btn-primary">${actionText}</button>
          </div>`
            : ''
        }
      </div>
    </div>
  `;

  shell.appendChild(modal);

  const closeBtn = modal.querySelector('.notification-close-btn');
  const actionBtn = modal.querySelector('.notification-btn-primary');

  function dismiss() {
    modal.classList.add('removing');
    setTimeout(() => modal.remove(), 280);
  }

  closeBtn.addEventListener('click', dismiss);
  if (actionBtn) {
    actionBtn.addEventListener('click', () => {
      dismiss();
      if (onAction) onAction();
    });
  }

  if (autoDismiss > 0) {
    setTimeout(() => {
      if (modal.parentNode) dismiss();
    }, autoDismiss);
  }
}
