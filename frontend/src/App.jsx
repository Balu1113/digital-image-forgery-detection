import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { AuthProvider } from './auth/AuthContext';
import { ThemeProvider } from './theme/ThemeContext';
import NavBar from './components/NavBar';
import HomePage from './pages/HomePage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';

export default function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <BrowserRouter>
          <div className="page">
            <NavBar />
            <Routes>
              <Route path="/" element={<LoginPage />} />
              <Route path="/login" element={<LoginPage />} />
              <Route path="/register" element={<RegisterPage />} />
              <Route path="/app" element={<HomePage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
            <footer className="footer muted small">
              Flask inference service · Django REST gateway · React front-end
            </footer>
          </div>
        </BrowserRouter>
      </AuthProvider>
    </ThemeProvider>
  );
}
