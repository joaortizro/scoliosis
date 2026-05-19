export function Header() {
  return (
    <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/95 backdrop-blur">
      <div className="mx-auto flex h-16 w-full max-w-6xl items-center justify-between px-6 lg:px-8">
        <a href="#" className="text-sm font-bold tracking-normal text-slate-950">
          Scoliosis AI
        </a>
        <nav className="flex items-center gap-5 text-sm font-medium text-slate-600">
          <a className="transition hover:text-slate-950" href="#prediction">
            Prediction
          </a>
          <a className="transition hover:text-slate-950" href="#methodology">
            Methodology
          </a>
        </nav>
      </div>
    </header>
  );
}
