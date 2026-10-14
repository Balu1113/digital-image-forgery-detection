import { useState } from 'react';

/**
 * Reusable password visibility toggle button.
 * Works with any input via aria-label and aria-pressed.
 */
export default function PasswordVisibilityToggle({
  visible,
  onToggle,
  id,
  disabled = false,
}) {
  return (
    <button
      type="button"
      id={id}
      aria-label={visible ? 'Hide password' : 'Show password'}
      aria-pressed={visible}
      disabled={disabled}
      onClick={() => onToggle(!visible)}
      className="absolute inset-y-0 right-3 flex items-center text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:ring-offset-2 dark:ring-offset-slate-900 rounded-md px-1"
    >
      {visible ? (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M13.875 18.825A10.05 10.05 0 0112 5.75M12 5.75a10.05 10.05 0 00-8.25 4.875M12 5.75a10.05 10.05 0 018.25 4.875M12 5.75a1.5 1.5 0 010 3M12 5.75a1.5 1.5 0 000 3m0 0v.031M3 3l18 18" />
        </svg>
      ) : (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.477 0 8.268 2.943 9.542 7-1.274 4.057-5.065 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
          <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      )}
    </button>
  );
}