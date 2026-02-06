import { useParams } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { apiFetch } from '../api'

export default function ProjectDetail() {
  const { id } = useParams()
  const [project, setProject] = useState<any>(null)

  useEffect(() => {
    const load = async () => {
      const response = await apiFetch(`/api/projects/${id}`)
      setProject(await response.json())
    }
    load()
  }, [id])

  if (!project) return <p>Loading...</p>

  return (
    <div className="bg-white p-4 rounded shadow">
      <h2 className="font-semibold mb-2">{project.project_code}</h2>
      <p>{project.name}</p>
      <p className="text-sm text-slate-500">Status: {project.status}</p>
    </div>
  )
}
