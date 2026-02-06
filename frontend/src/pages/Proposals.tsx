import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { apiFetch } from '../api'

export default function Proposals() {
  const { data } = useQuery({
    queryKey: ['proposals'],
    queryFn: async () => {
      const response = await apiFetch('/api/proposals/')
      return response.json()
    }
  })

  return (
    <div className="bg-white p-4 rounded shadow">
      <h2 className="font-semibold mb-2">Proposals</h2>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left">
            <th>ID</th>
            <th>Target</th>
            <th>Status</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {data?.map((proposal: any) => (
            <tr key={proposal.id} className="border-t">
              <td>{proposal.id}</td>
              <td>{proposal.target_table}</td>
              <td>{proposal.status}</td>
              <td>
                <Link className="text-blue-600" to={`/proposals/${proposal.id}`}>
                  Review
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
