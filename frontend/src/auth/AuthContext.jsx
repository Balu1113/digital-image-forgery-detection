import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { api, clearSession, setSession } from '../api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(api.getCachedUser());
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    if (!api.getToken()) {
      setReady(true);
      return () => {
        cancelled = true;
      };
    }
    api
      .me()
      .then((me) => !cancelled && setUser(me))
      .catch(() => {
        clearSession();
        if (!cancelled) setUser(null);
      })
      .finally(() => !cancelled && setReady(true));

    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(async (username, password) => {
    const data = await api.login(username, password);
    setSession(data.token, data.user);
    setUser(data.user);
    return data.user;
  }, []);

  const register = useCallback(async (username, password, email) => {
    const data = await api.register(username, password, email);
    setSession(data.token, data.user);
    setUser(data.user);
    return data.user;
  }, []);

  const logout = useCallback(async () => {
    try {
      await api.logout();
    } catch {
      /* ignore network errors on logout */
    }
    clearSession();
    setUser(null);
  }, []);

  const value = { user, ready, login, register, logout, isAuthenticated: Boolean(user) };
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within an AuthProvider');
  return context;
}