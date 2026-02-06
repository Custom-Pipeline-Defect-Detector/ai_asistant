import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { apiFetch } from '../api'

export default function Projects() {
  const { data } = useQuery({
    queryKey: ['projects'],
    queryFn: async () => {
      const response = await apiFetch('/api/projects/')
      return response.json()
    }
  })

  return (
    <div className="bg-white p-4 rounded shadow">
      <h2 className="font-semibold mb-2">Projects</h2>
      <ul className="space-y-2">
        {data?.map((project: any) => (
          <li key={project.id} className="border-b pb-2">
            <Link className="text-blue-600" to={`/projects/${project.id}`}>
              {project.project_code} - {project.name}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  )
}
