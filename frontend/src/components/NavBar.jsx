import { Link, NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';

export default function NavBar() {
  const { user, isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();

  async function handleLogout() {
    await logout();
    navigate('/');
  }

  return (
    <nav className="navbar">
      <Link to="/" className="brand">
        🔍 <span>Forgery Detection</span>
      </Link>

      <div className="nav-links">
        {isAuthenticated ? (
          <>
            <span className="nav-user">
              {user?.username}
            </span>
            <button className="btn ghost small" onClick={handleLogout}>
              Log out
            </button>
          </>
        ) : (
          <>
            <NavLink to="/login" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
              Log in
            </NavLink>
            <NavLink to="/register" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
              Register
            </NavLink>
          </>
        )}
      </div>
    </nav>
  );
}