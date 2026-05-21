"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

const menuItems = [
  { label: "Home", href: "/" },
  { label: "Flash Prediction", href: "/prediction" },
];

export function Header() {
  const [isOpen, setIsOpen] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    let isMounted = true;

    async function loadSession() {
      const response = await fetch("/api/demo-session").catch(() => null);
      const payload = response?.ok
        ? ((await response.json()) as { isAuthenticated?: boolean })
        : null;

      if (isMounted) {
        setIsAuthenticated(Boolean(payload?.isAuthenticated));
      }
    }

    void loadSession();

    return () => {
      isMounted = false;
    };
  }, []);

  async function handleLogout() {
    await fetch("/api/demo-logout", { method: "POST" });
    setIsAuthenticated(false);
    setIsOpen(false);
    window.location.href = "/login";
  }

  return (
    <header className="sticky top-0 z-30 border-b border-[#d9e5ee] bg-white/95 backdrop-blur">
      <div className="relative mx-auto grid h-16 w-full max-w-7xl grid-cols-3 items-center px-5 lg:px-8">
        <div className="flex justify-start">
          <button
            aria-expanded={isOpen}
            aria-label="Open navigation menu"
            className="inline-flex h-10 w-10 items-center justify-center rounded-md border border-[#d9e5ee] text-[#102a43] transition hover:bg-[#f4f8fb] focus:outline-none focus:ring-2 focus:ring-[#0a5f9e] focus:ring-offset-2"
            type="button"
            onClick={() => setIsOpen((value) => !value)}
          >
            <span className="grid gap-1">
              <span className="block h-0.5 w-5 bg-current" />
              <span className="block h-0.5 w-5 bg-current" />
              <span className="block h-0.5 w-5 bg-current" />
            </span>
          </button>
        </div>

        <div className="flex justify-center">
          <Link
            className="text-sm font-bold uppercase tracking-[0.18em] text-[#102a43]"
            href="/"
          >
            SpineView AI
          </Link>
        </div>

        <div className="flex justify-end">
          {isAuthenticated ? (
            <button
              className="inline-flex h-10 items-center justify-center rounded-md bg-[#102a43] px-4 text-sm font-semibold text-white transition hover:bg-[#0a5f9e] focus:outline-none focus:ring-2 focus:ring-[#0a5f9e] focus:ring-offset-2"
              onClick={handleLogout}
              type="button"
            >
              Logout
            </button>
          ) : (
            <Link
              className="inline-flex h-10 items-center justify-center rounded-md bg-[#102a43] px-4 text-sm font-semibold text-white transition hover:bg-[#0a5f9e] focus:outline-none focus:ring-2 focus:ring-[#0a5f9e] focus:ring-offset-2"
              href="/login"
            >
              Login
            </Link>
          )}
        </div>
      </div>

      {isOpen ? (
        <div className="border-t border-[#d9e5ee] bg-white">
          <nav className="mx-auto grid w-full max-w-7xl gap-1 px-5 py-4 lg:px-8">
            {menuItems.map((item) => (
              <Link
                key={item.href}
                className="rounded-md px-3 py-2 text-sm font-semibold text-[#102a43] transition hover:bg-[#f4f8fb] hover:text-[#0a5f9e]"
                href={item.href}
                onClick={() => setIsOpen(false)}
              >
                {item.label}
              </Link>
            ))}
            {isAuthenticated ? (
              <button
                className="rounded-md px-3 py-2 text-left text-sm font-semibold text-[#102a43] transition hover:bg-[#f4f8fb] hover:text-[#0a5f9e]"
                onClick={handleLogout}
                type="button"
              >
                Logout
              </button>
            ) : (
              <Link
                className="rounded-md px-3 py-2 text-sm font-semibold text-[#102a43] transition hover:bg-[#f4f8fb] hover:text-[#0a5f9e]"
                href="/login"
                onClick={() => setIsOpen(false)}
              >
                Login
              </Link>
            )}
          </nav>
        </div>
      ) : null}
    </header>
  );
}
