import { Link, NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';
import { useTheme } from '../theme/ThemeContext';

function SunIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v2.25m6.364.386l-1.591 1.591M21 12h-2.25m-.386 6.364l-1.591-1.591M12 18.75V21m-4.773-4.227l-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M21.752 15.002A9.718 9.718 0 0118 15.75c-5.385 0-9.75-4.365-9.75-9.75 0-1.33.266-2.597.748-3.752A9.753 9.753 0 003 11.25C3 16.635 7.365 21 12.75 21a9.753 9.753 0 009.002-5.998z" />
    </svg>
  );
}

export default function NavBar() {
  const { user, isAuthenticated, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();

  const themeLabel = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';

  async function handleLogout() {
    await logout();
    navigate('/');
  }

  return (
    <nav className="navbar">
      <Link to="/app" className="brand">
        🔍 <span>Forgery Detection</span>
      </Link>

      <div className="nav-links">
        <button
          type="button"
          className="btn ghost small theme-toggle"
          onClick={toggleTheme}
          aria-label={themeLabel}
          title={themeLabel}
        >
          {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
        </button>
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
