import type { ButtonHTMLAttributes, ReactNode } from "react";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  children: ReactNode;
};

export function Button({ children, className = "", ...props }: ButtonProps) {
  return (
    <button
      className={`inline-flex h-11 items-center justify-center rounded-md bg-[#0a5f9e] px-5 text-sm font-semibold text-white shadow-sm transition hover:bg-[#084f84] focus:outline-none focus:ring-2 focus:ring-[#0a5f9e] focus:ring-offset-2 disabled:cursor-not-allowed disabled:bg-slate-300 ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
