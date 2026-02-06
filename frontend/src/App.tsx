import { Link, Route, Routes } from 'react-router-dom'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import Documents from './pages/Documents'
import Proposals from './pages/Proposals'
import ProposalDetail from './pages/ProposalDetail'
import Projects from './pages/Projects'
import ProjectDetail from './pages/ProjectDetail'

const Nav = () => (
  <nav className="bg-white border-b px-6 py-3 flex gap-4">
    <Link to="/dashboard">Dashboard</Link>
    <Link to="/documents">Documents</Link>
    <Link to="/proposals">Proposals</Link>
    <Link to="/projects">Projects</Link>
  </nav>
)

export default function App() {
  return (
    <div>
      <Nav />
      <div className="p-6">
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/documents" element={<Documents />} />
          <Route path="/proposals" element={<Proposals />} />
          <Route path="/proposals/:id" element={<ProposalDetail />} />
          <Route path="/projects" element={<Projects />} />
          <Route path="/projects/:id" element={<ProjectDetail />} />
          <Route path="*" element={<Login />} />
        </Routes>
      </div>
    </div>
  )
}
