const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export function getToken() {
  return localStorage.getItem('token') || ''
}

export async function apiFetch(path: string, options: RequestInit = {}) {
  const token = getToken()
  const headers = new Headers(options.headers || {})
  if (token) {
    headers.set('Authorization', `Bearer ${token}`)
  }
  return fetch(`${API_BASE}${path}`, { ...options, headers })
}
