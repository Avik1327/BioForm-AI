
const menuBtn = document.getElementById('menuBtn');
const mainNav = document.getElementById('mainNav');
menuBtn?.addEventListener('click', () => mainNav?.classList.toggle('open'));

window.showLoading = (title, text) => {
  const overlay = document.getElementById('loadingOverlay');
  const titleEl = document.getElementById('loadingTitle');
  const textEl = document.getElementById('loadingText');
  if (title && titleEl) titleEl.textContent = title;
  if (text && textEl) textEl.textContent = text;
  overlay?.classList.add('show');
  overlay?.setAttribute('aria-hidden', 'false');
};

window.hideLoading = () => {
  const overlay = document.getElementById('loadingOverlay');
  overlay?.classList.remove('show');
  overlay?.setAttribute('aria-hidden', 'true');
};
