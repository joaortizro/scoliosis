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
  const [logoFailed, setLogoFailed] = useState(false);

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
    <header className="sticky top-0 z-30 border-b border-[#c7c6b7]/45 bg-white/95 backdrop-blur">
      <div className="relative mx-auto grid h-16 w-full max-w-7xl grid-cols-3 items-center px-5 lg:px-8">
        <div className="flex justify-start">
          <button
            aria-expanded={isOpen}
            aria-label="Open navigation menu"
            className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-[#f2f8ff] text-[#1c3f9a] transition hover:bg-[#dcedff] focus:outline-none focus:ring-2 focus:ring-[#007ae5]/30 focus:ring-offset-2"
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
            className="inline-flex items-center justify-center"
            href="/"
          >
            {logoFailed ? (
              <span className="text-sm font-bold uppercase tracking-[0.18em] text-[#1c3f9a]">
                MIRO
              </span>
            ) : (
              <>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  alt="MIRO"
                  className="h-10 w-auto"
                  onError={() => setLogoFailed(true)}
                  src="/logo/miro_logo.svg"
                />
                <span className="sr-only">MIRO</span>
              </>
            )}
          </Link>
        </div>

        <div className="flex justify-end">
          {isAuthenticated ? (
            <button
              className="inline-flex h-10 items-center justify-center rounded-full bg-[#1c3f9a] px-5 text-sm font-semibold text-white transition hover:bg-[#007ae5] focus:outline-none focus:ring-2 focus:ring-[#007ae5]/30 focus:ring-offset-2"
              onClick={handleLogout}
              type="button"
            >
              Logout
            </button>
          ) : (
            <Link
              className="inline-flex h-10 items-center justify-center rounded-full bg-[#1c3f9a] px-5 text-sm font-semibold text-white transition hover:bg-[#007ae5] focus:outline-none focus:ring-2 focus:ring-[#007ae5]/30 focus:ring-offset-2"
              href="/login"
            >
              Login
            </Link>
          )}
        </div>
      </div>

      {isOpen ? (
        <div className="border-t border-[#c7c6b7]/45 bg-white">
          <nav className="mx-auto grid w-full max-w-7xl gap-1 px-5 py-4 lg:px-8">
            {menuItems.map((item) => (
              <Link
                key={item.href}
                className="rounded-xl px-3 py-2 text-sm font-semibold text-[#1c3f9a] transition hover:bg-[#f2f8ff] hover:text-[#007ae5]"
                href={item.href}
                onClick={() => setIsOpen(false)}
              >
                {item.label}
              </Link>
            ))}
            {isAuthenticated ? (
              <button
                className="rounded-xl px-3 py-2 text-left text-sm font-semibold text-[#1c3f9a] transition hover:bg-[#f2f8ff] hover:text-[#007ae5]"
                onClick={handleLogout}
                type="button"
              >
                Logout
              </button>
            ) : (
              <Link
                className="rounded-xl px-3 py-2 text-sm font-semibold text-[#1c3f9a] transition hover:bg-[#f2f8ff] hover:text-[#007ae5]"
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
