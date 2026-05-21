import type { ReactNode } from "react";

type BadgeProps = {
  children: ReactNode;
};

export function Badge({ children }: BadgeProps) {
  return (
    <span className="inline-flex w-fit items-center rounded-md border border-[#c9dfef] bg-[#eef7fc] px-2.5 py-1 text-xs font-semibold uppercase text-[#0a5f9e]">
      {children}
    </span>
  );
}
