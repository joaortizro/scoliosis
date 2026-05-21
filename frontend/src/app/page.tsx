import Link from "next/link";
import { Badge } from "@/components/ui/Badge";
import { PROJECT_DISCLAIMER } from "@/lib/constants";

export default function Home() {
  return (
    <>
      <section className="bg-white">
        <div className="mx-auto grid min-h-[calc(100vh-4rem)] w-full max-w-6xl gap-10 px-5 py-12 lg:grid-cols-[1fr_0.92fr] lg:px-8">
          <div className="flex flex-col justify-center">
            <Badge>Master&apos;s research project</Badge>
            <h1 className="mt-6 max-w-3xl text-4xl font-semibold leading-tight text-[#1c3f9a] sm:text-6xl">
              Spine X-ray segmentation for scoliosis research.
            </h1>
            <p className="mt-6 max-w-2xl text-lg leading-8 text-[#182433]/75">
              A simple demo interface for uploading spine radiographs and
              reviewing vertebra-level segmentation output from a research
              model.
            </p>
            <div className="mt-9 flex flex-col gap-3 sm:flex-row">
              <Link
                className="inline-flex h-12 items-center justify-center rounded-full bg-[#007ae5] px-7 text-sm font-semibold text-white shadow-sm transition hover:bg-[#005fb3] focus:outline-none focus:ring-2 focus:ring-[#007ae5]/30 focus:ring-offset-2"
                href="/login"
              >
                Open demo
              </Link>
              <Link
                className="inline-flex h-12 items-center justify-center rounded-full bg-[#f2f8ff] px-7 text-sm font-semibold text-[#1c3f9a] transition hover:bg-[#dcedff] focus:outline-none focus:ring-2 focus:ring-[#007ae5]/30 focus:ring-offset-2"
                href="/prediction"
              >
                Flash Prediction
              </Link>
            </div>
          </div>

          <div className="flex items-center">
            <div className="w-full rounded-3xl bg-[#1c3f9a] p-5 text-white shadow-sm">
              <div className="min-h-[380px] rounded-2xl bg-[#073f73] p-6">
                <div className="flex items-center justify-between text-sm text-white/85">
                  <span>MIRO</span>
                  <span className="rounded-full bg-[#ff5c00] px-3 py-1 text-xs font-semibold text-white">
                    Demo
                  </span>
                </div>

                <div className="mx-auto mt-10 flex h-64 max-w-xs items-center justify-center rounded-full bg-white/5 ring-1 ring-white/15">
                  <div className="relative h-64 w-32">
                    <div className="absolute left-1/2 top-2 h-60 w-3 -translate-x-1/2 rounded-full bg-white/80 shadow-[0_0_36px_rgba(255,255,255,0.35)]" />
                    {Array.from({ length: 9 }).map((_, index) => (
                      <div
                        key={index}
                        className="absolute left-1/2 h-2 w-24 -translate-x-1/2 rounded-full bg-[#8fd4ff]/80"
                        style={{
                          top: `${26 + index * 22}px`,
                          transform: `translateX(-50%) rotate(${
                            index % 2 === 0 ? -6 : 6
                          }deg)`,
                        }}
                      />
                    ))}
                  </div>
                </div>

                <p className="mt-8 text-sm leading-6 text-white/85">
                  Upload a JPG or PNG X-ray, send it to the segmentation API,
                  and compare the returned overlay with the original image.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="border-y border-[#c7c6b7]/45 bg-[#f2f8ff]">
        <div className="mx-auto grid w-full max-w-6xl gap-8 px-5 py-14 lg:grid-cols-[0.8fr_1.2fr] lg:px-8">
          <div>
            <Badge>Research demo</Badge>
            <h2 className="mt-4 text-3xl font-semibold text-[#1c3f9a]">
              Built for review, not diagnosis.
            </h2>
          </div>
          <div className="grid gap-5 text-base leading-8 text-[#182433]/75">
            <p>
              The application is intentionally focused: a public overview, a
              demo login, and a protected workspace for running the current
              segmentation model.
            </p>
            <p className="rounded-2xl bg-[#fff4ed] p-4 text-sm leading-6 text-[#9a3500]">
              {PROJECT_DISCLAIMER}
            </p>
          </div>
        </div>
      </section>
    </>
  );
}
