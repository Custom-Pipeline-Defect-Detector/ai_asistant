import { useParams } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { apiFetch } from '../api'

export default function ProposalDetail() {
  const { id } = useParams()
  const [proposal, setProposal] = useState<any>(null)
  const [message, setMessage] = useState('')

  const load = async () => {
    const response = await apiFetch(`/api/proposals/${id}`)
    setProposal(await response.json())
  }

  useEffect(() => {
    load()
  }, [id])

  const handleFieldChange = (key: string, value: string) => {
    setProposal({ ...proposal, proposed_fields: { ...proposal.proposed_fields, [key]: value } })
  }

  const handleSave = async () => {
    const response = await apiFetch(`/api/proposals/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ proposed_fields: proposal.proposed_fields })
    })
    if (response.ok) {
      setMessage('Saved')
    }
  }

  const handleDecision = async (decision: 'approve' | 'reject') => {
    const reason = decision === 'reject' ? prompt('Reason for rejection?') : undefined
    const response = await apiFetch(`/api/proposals/${id}/decision`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision, reason })
    })
    if (response.ok) {
      setProposal(await response.json())
      setMessage(`Proposal ${decision}d`)
    }
  }

  if (!proposal) return <p>Loading...</p>

  return (
    <div className="space-y-4">
      <div className="bg-white p-4 rounded shadow">
        <h2 className="font-semibold mb-2">Proposal #{proposal.id}</h2>
        <p className="text-sm text-slate-500">Status: {proposal.status}</p>
      </div>
      <div className="bg-white p-4 rounded shadow">
        <h3 className="font-semibold mb-2">Fields</h3>
        {Object.entries(proposal.proposed_fields).map(([key, value]) => (
          <div key={key} className="mb-3">
            <label className="block text-sm font-medium">{key}</label>
            <input
              className="border p-2 w-full"
              value={value as string}
              onChange={(event) => handleFieldChange(key, event.target.value)}
            />
            <p className="text-xs text-slate-500">
              Confidence: {proposal.field_confidence?.[key] ?? 'n/a'}
            </p>
            <p className="text-xs text-slate-500">
              Evidence: {proposal.evidence?.[key]?.snippet}
            </p>
          </div>
        ))}
        <div className="flex gap-2">
          <button className="bg-slate-200 px-3 py-2 rounded" onClick={handleSave}>
            Save
          </button>
          <button className="bg-green-600 text-white px-3 py-2 rounded" onClick={() => handleDecision('approve')}>
            Approve
          </button>
          <button className="bg-red-600 text-white px-3 py-2 rounded" onClick={() => handleDecision('reject')}>
            Reject
          </button>
        </div>
        {message && <p className="text-sm text-slate-600 mt-2">{message}</p>}
      </div>
    </div>
  )
}
