import type { HTMLAttributes, ReactNode } from "react";

type CardProps = HTMLAttributes<HTMLDivElement> & {
  children: ReactNode;
};

export function Card({ children, className = "", ...props }: CardProps) {
  return (
    <div
      className={`rounded-lg border border-[#d9e5ee] bg-white shadow-sm ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}
