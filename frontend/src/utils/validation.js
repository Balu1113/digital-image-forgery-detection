/**
 * Shared validation helpers for the auth forms.
 * All functions return an error message string, or null when valid.
 */

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const USERNAME_RE = /^[a-zA-Z0-9_]{3,30}$/;

export function validateUsername(value) {
  if (!value || !value.trim()) {
    return 'Username is required.';
  }
  if (!USERNAME_RE.test(value.trim())) {
    return 'Username must be 3-30 characters using letters, numbers, or underscores.';
  }
  return null;
}

export function validateEmail(value, { required = false } = {}) {
  const trimmed = (value || '').trim();
  if (!trimmed) {
    return required ? 'Email is required.' : null;
  }
  if (!EMAIL_RE.test(trimmed)) {
    return 'Please enter a valid email address.';
  }
  return null;
}

export function validatePassword(value) {
  if (!value) {
    return 'Password is required.';
  }
  if (value.length < 8) {
    return 'Password must be at least 8 characters.';
  }
  if (value.length > 128) {
    return 'Password must be 128 characters or fewer.';
  }
  const hasUpper = /[A-Z]/.test(value);
  const hasLower = /[a-z]/.test(value);
  const hasNumber = /[0-9]/.test(value);
  if (!hasUpper || !hasLower || !hasNumber) {
    return 'Password must include uppercase, lowercase, and a number.';
  }
  return null;
}

export function validatePasswordConfirmation(password, confirm) {
  if (!confirm) {
    return 'Please confirm your password.';
  }
  if (password !== confirm) {
    return 'Passwords do not match.';
  }
  return null;
}

/**
 * Returns a 0-4 strength score for a password.
 * 0 = very weak, 4 = strong.
 */
export function passwordStrength(password) {
  if (!password) return 0;
  let score = 0;
  if (password.length >= 8) score++;
  if (password.length >= 12) score++;
  if (/[A-Z]/.test(password) && /[a-z]/.test(password)) score++;
  if (/[0-9]/.test(password)) score++;
  if (/[^A-Za-z0-9]/.test(password)) score++;
  return Math.min(score, 4);
}

export const strengthLabels = ['Very weak', 'Weak', 'Fair', 'Good', 'Strong'];
export const strengthColors = [
  'bg-red-500',
  'bg-orange-500',
  'bg-amber-400',
  'bg-emerald-500',
  'bg-emerald-600',
];