import { forwardRef } from 'react';
import PasswordVisibilityToggle from './PasswordVisibilityToggle';

/**
 * Form input with integrated validation state and error display.
 * - Supports text, email, password, and other input types.
 * - Shows success/error states with aria-live announcements.
 * - Optional show/hide toggle for password fields.
 */
const FormInput = forwardRef(
  (
    {
      label,
      name,
      type = 'text',
      value,
      onChange,
      error,
      success,
      hint,
      showToggle = false,
      showPassword,
      onTogglePassword,
      disabled = false,
      required = false,
      autoComplete,
      autoFocus,
      minLength,
      maxLength,
      placeholder,
      className = '',
      inputId,
      ...rest
    },
    ref
  ) => {
    const inputIdFinal = inputId || `input-${name}-${Math.random().toString(36).slice(2, 9)}`;
    const hasError = !!error;
    const hasSuccess = !!success && !hasError;

    const baseInput =
      'w-full rounded-lg border px-3 py-2.5 text-sm text-slate-800 dark:text-slate-200 placeholder-slate-400 dark:placeholder-slate-500 transition focus:outline-none focus:ring-2 focus:ring-offset-1 dark:ring-offset-slate-900 disabled:cursor-not-allowed disabled:bg-slate-100 dark:disabled:bg-slate-800';

    let borderClasses =
      'border-slate-300 dark:border-slate-600 focus:border-brand-500 focus:ring-brand-500';
    if (hasError) {
      borderClasses = 'border-red-500 focus:border-red-500 focus:ring-red-500';
    } else if (hasSuccess) {
      borderClasses = 'border-emerald-500 focus:border-emerald-500 focus:ring-emerald-500';
    }

    return (
      <div className={className}>
        <label
          htmlFor={inputIdFinal}
          className="mb-1.5 block text-sm font-semibold text-slate-700 dark:text-slate-300"
        >
          {label}
          {required && (
            <span className="ml-1 text-red-500" aria-hidden="true">
              *
            </span>
          )}
        </label>
        <div className="relative">
          <input
            ref={ref}
            id={inputIdFinal}
            name={name}
            type={type}
            value={value}
            onChange={onChange}
            disabled={disabled}
            required={required}
            autoComplete={autoComplete}
            autoFocus={autoFocus}
            minLength={minLength}
            maxLength={maxLength}
            placeholder={placeholder}
            aria-invalid={hasError}
            aria-describedby={
              (hasError && `${inputIdFinal}-error`) ||
              (hint && `${inputIdFinal}-hint`) ||
              undefined
            }
            className={`${baseInput} ${borderClasses} ${showToggle ? 'pr-10' : ''}`}
            {...rest}
          />
          {showToggle && (
            <PasswordVisibilityToggle
              id={`toggle-${inputIdFinal}`}
              visible={showPassword}
              onToggle={onTogglePassword}
              disabled={disabled}
            />
          )}
        </div>
        {hint && !hasError && (
          <p id={`${inputIdFinal}-hint`} className="mt-1.5 text-xs text-slate-500 dark:text-slate-400">
            {hint}
          </p>
        )}
        {hasError && (
          <p
            id={`${inputIdFinal}-error`}
            role="alert"
            aria-live="polite"
            className="mt-1.5 text-xs font-medium text-red-600 dark:text-red-400"
          >
            {error}
          </p>
        )}
      </div>
    );
  }
);

FormInput.displayName = 'FormInput';
export default FormInput;