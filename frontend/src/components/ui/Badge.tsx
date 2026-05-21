import type { ReactNode } from "react";

type BadgeProps = {
  children: ReactNode;
};

export function Badge({ children }: BadgeProps) {
  return (
    <span className="inline-flex w-fit items-center rounded-full bg-[#f2f8ff] px-3 py-1 text-xs font-semibold uppercase text-[#007ae5]">
      {children}
    </span>
  );
}
