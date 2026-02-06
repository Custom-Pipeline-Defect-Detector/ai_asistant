import { useQuery } from '@tanstack/react-query'
import { apiFetch } from '../api'
import { useState } from 'react'

export default function Documents() {
  const [message, setMessage] = useState('')
  const { data, refetch } = useQuery({
    queryKey: ['documents'],
    queryFn: async () => {
      const response = await apiFetch('/api/documents/')
      return response.json()
    }
  })

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    const form = new FormData()
    form.append('file', file)
    const response = await apiFetch('/api/documents/upload', { method: 'POST', body: form })
    if (response.ok) {
      setMessage('Uploaded successfully. Processing...')
      refetch()
    } else {
      setMessage('Upload failed')
    }
  }

  return (
    <div>
      <div className="bg-white p-4 rounded shadow mb-4">
        <input type="file" onChange={handleUpload} />
        {message && <p className="text-sm text-slate-600 mt-2">{message}</p>}
      </div>
      <div className="bg-white p-4 rounded shadow">
        <h2 className="font-semibold mb-2">Uploaded Documents</h2>
        <ul className="space-y-2">
          {data?.map((doc: any) => (
            <li key={doc.id} className="border-b pb-2">
              <p className="font-medium">{doc.filename}</p>
              <p className="text-xs text-slate-500">{doc.created_at}</p>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
