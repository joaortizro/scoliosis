"use client";

import { usePathname } from "next/navigation";

export function Footer() {
  const pathname = usePathname();

  if (pathname.startsWith("/prediction")) {
    return null;
  }

  return (
    <footer className="border-t border-[#c7c6b7]/45 bg-white">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-3 px-5 py-6 text-sm text-[#182433]/75 sm:flex-row sm:items-center sm:justify-between lg:px-8">
        <p>Research frontend for AI-powered scoliosis detection.</p>
        <p>Not intended for clinical diagnosis.</p>
      </div>
    </footer>
  );
}
