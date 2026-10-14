const API_URL = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '');

function getToken() {
  return localStorage.getItem('token');
}

function setSession(token, user) {
  localStorage.setItem('token', token);
  localStorage.setItem('user', JSON.stringify(user));
}

function clearSession() {
  localStorage.removeItem('token');
  localStorage.removeItem('user');
}

function getCachedUser() {
  try {
    return JSON.parse(localStorage.getItem('user')) ?? null;
  } catch {
    return null;
  }
}

async function request(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  const token = getToken();
  if (token) headers.Authorization = `Token ${token}`;
  if (options.body && !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }

  const response = await fetch(`${API_URL}${path}`, { ...options, headers });

  if (response.status === 204) return null;

  let data = null;
  try {
    data = await response.json();
  } catch {
    /* non-JSON body */
  }

  if (!response.ok) {
    let detail = data?.error || data?.detail || data?.message || `Request failed (${response.status})`;
    if (data && typeof data === 'object') {
      const first = Object.values(data).find((v) => Array.isArray(v) && v.length > 0);
      if (first) detail = first[0];
    }
    const err = new Error(detail);
    err.status = response.status;
    throw err;
  }
  return data;
}

export const api = {
  apiUrl: API_URL,
  getToken,
  getCachedUser,
  register: (username, password, email) =>
    request('/api/auth/register/', {
      method: 'POST',
      body: JSON.stringify({ username, password, email }),
    }),
  login: (username, password) =>
    request('/api/auth/login/', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    }),
  logout: () => request('/api/auth/logout/', { method: 'POST' }),
  me: () => request('/api/auth/me/'),
  health: () => request('/api/health/'),
  analyze: (file) => {
    const form = new FormData();
    form.append('image', file);
    return request('/api/predictions/', { method: 'POST', body: form });
  },
  history: () => request('/api/predictions/'),
  deletePrediction: (id) => request(`/api/predictions/${id}/`, { method: 'DELETE' }),
  sendFeedback: (id, verdict) =>
    request(`/api/predictions/${id}/feedback/`, {
      method: 'POST',
      body: JSON.stringify({ verdict }),
    }),
};

export { setSession, clearSession };