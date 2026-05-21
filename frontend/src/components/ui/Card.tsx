import type { HTMLAttributes, ReactNode } from "react";

type CardProps = HTMLAttributes<HTMLDivElement> & {
  children: ReactNode;
};

export function Card({ children, className = "", ...props }: CardProps) {
  return (
    <div
      className={`rounded-2xl bg-white shadow-sm shadow-[#073f73]/10 ring-1 ring-[#c7c6b7]/35 ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}
