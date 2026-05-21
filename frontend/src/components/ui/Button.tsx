import type { ButtonHTMLAttributes, ReactNode } from "react";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  children: ReactNode;
};

export function Button({ children, className = "", ...props }: ButtonProps) {
  return (
    <button
      className={`inline-flex h-11 items-center justify-center rounded-full bg-[#007ae5] px-5 text-sm font-semibold text-white shadow-sm transition hover:bg-[#005fb3] focus:outline-none focus:ring-2 focus:ring-[#007ae5]/30 focus:ring-offset-2 disabled:cursor-not-allowed disabled:bg-[#c7c6b7] ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
