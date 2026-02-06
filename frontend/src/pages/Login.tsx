import { useState } from 'react'

export default function Login() {
  const [email, setEmail] = useState('admin@aline.local')
  const [password, setPassword] = useState('Admin123!')
  const [message, setMessage] = useState('')

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    const body = new URLSearchParams({ username: email, password })
    const response = await fetch(`${import.meta.env.VITE_API_BASE}/api/auth/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body
    })
    if (!response.ok) {
      setMessage('Login failed')
      return
    }
    const data = await response.json()
    localStorage.setItem('token', data.access_token)
    setMessage('Login successful. Go to Dashboard.')
  }

  return (
    <div className="max-w-md bg-white shadow p-6 rounded">
      <h1 className="text-xl font-semibold mb-4">Login</h1>
      <form onSubmit={handleSubmit} className="space-y-3">
        <input className="border p-2 w-full" value={email} onChange={(e) => setEmail(e.target.value)} />
        <input
          className="border p-2 w-full"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <button className="bg-blue-600 text-white px-4 py-2 rounded" type="submit">
          Login
        </button>
      </form>
      {message && <p className="mt-2 text-sm text-slate-600">{message}</p>}
    </div>
  )
}
