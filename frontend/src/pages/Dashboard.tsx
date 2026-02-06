import { useQuery } from '@tanstack/react-query'
import { apiFetch } from '../api'

export default function Dashboard() {
  const { data } = useQuery({
    queryKey: ['dashboard'],
    queryFn: async () => {
      const response = await apiFetch('/api/dashboard/counts')
      return response.json()
    }
  })

  return (
    <div className="grid grid-cols-2 gap-4">
      <div className="bg-white p-4 rounded shadow">
        <h2 className="text-sm text-slate-500">Projects</h2>
        <p className="text-2xl font-semibold">{data?.projects ?? 0}</p>
      </div>
      <div className="bg-white p-4 rounded shadow">
        <h2 className="text-sm text-slate-500">Tasks</h2>
        <p className="text-2xl font-semibold">{data?.tasks ?? 0}</p>
      </div>
      <div className="bg-white p-4 rounded shadow">
        <h2 className="text-sm text-slate-500">Pending Proposals</h2>
        <p className="text-2xl font-semibold">{data?.pending_proposals ?? 0}</p>
      </div>
      <div className="bg-white p-4 rounded shadow">
        <h2 className="text-sm text-slate-500">NCRs</h2>
        <p className="text-2xl font-semibold">{data?.ncrs ?? 0}</p>
      </div>
    </div>
  )
}
