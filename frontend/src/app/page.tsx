import Link from "next/link";
import { Badge } from "@/components/ui/Badge";
import { PROJECT_DISCLAIMER } from "@/lib/constants";

const workflowSteps = [
  "Upload a spine X-ray",
  "Model analyzes the image",
  "Research result is shown",
];

export default function Home() {
  return (
    <>
      <section className="bg-white">
        <div className="mx-auto grid min-h-[calc(100vh-4rem)] w-full max-w-7xl gap-10 px-5 py-12 lg:grid-cols-[1.05fr_0.95fr] lg:px-8">
          <div className="flex flex-col justify-center">
            <Badge>Master&apos;s research project</Badge>
            <h1 className="mt-6 max-w-4xl text-5xl font-semibold leading-[1.02] text-[#102a43] sm:text-6xl lg:text-7xl">
              Clear AI support for scoliosis research.
            </h1>
            <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-600">
              A calm, focused frontend for reviewing spine X-ray predictions
              from a deep learning scoliosis detection pipeline.
            </p>
            <div className="mt-9 flex flex-col gap-3 sm:flex-row">
              <Link
                className="inline-flex h-12 items-center justify-center rounded-md bg-[#0a5f9e] px-6 text-sm font-semibold text-white shadow-sm transition hover:bg-[#084f84] focus:outline-none focus:ring-2 focus:ring-[#0a5f9e] focus:ring-offset-2"
                href="/prediction"
              >
                Start prediction
              </Link>
              <a
                className="inline-flex h-12 items-center justify-center rounded-md border border-[#d9e5ee] bg-white px-6 text-sm font-semibold text-[#102a43] transition hover:bg-[#f4f8fb] focus:outline-none focus:ring-2 focus:ring-[#0a5f9e] focus:ring-offset-2"
                href="#methodology"
              >
                Learn more
              </a>
            </div>
          </div>

          <div className="flex items-center">
            <div className="w-full rounded-lg bg-[#102a43] p-5 text-white shadow-sm">
              <div className="min-h-[440px] rounded-md border border-white/10 bg-[linear-gradient(180deg,#15395a,#102a43)] p-6">
                <div className="flex items-center justify-between text-sm text-blue-100">
                  <span>SpineView AI</span>
                  <span className="rounded-md bg-[#f97316] px-2 py-1 text-xs font-semibold text-white">
                    Research
                  </span>
                </div>

                <div className="mx-auto mt-12 flex h-72 max-w-xs items-center justify-center rounded-full border border-blue-200/20 bg-white/5">
                  <div className="relative h-64 w-32">
                    <div className="absolute left-1/2 top-2 h-60 w-3 -translate-x-1/2 rounded-full bg-white/80 shadow-[0_0_36px_rgba(255,255,255,0.35)]" />
                    {Array.from({ length: 10 }).map((_, index) => (
                      <div
                        key={index}
                        className="absolute left-1/2 h-2 w-24 -translate-x-1/2 rounded-full bg-blue-100/80"
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

                <div className="mt-10 grid gap-3 sm:grid-cols-3">
                  {workflowSteps.map((step, index) => (
                    <div
                      key={step}
                      className="rounded-md border border-white/10 bg-white/5 p-3"
                    >
                      <p className="text-xs font-semibold text-[#f97316]">
                        0{index + 1}
                      </p>
                      <p className="mt-2 text-sm leading-5 text-blue-50">
                        {step}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="methodology" className="border-y border-[#d9e5ee] bg-[#f4f8fb]">
        <div className="mx-auto grid w-full max-w-7xl gap-8 px-5 py-16 lg:grid-cols-[0.8fr_1.2fr] lg:px-8">
          <div>
            <Badge>Methodology</Badge>
            <h2 className="mt-4 text-3xl font-semibold text-[#102a43]">
              Simple interface, research-grade pipeline.
            </h2>
          </div>
          <div className="grid gap-5 text-base leading-8 text-slate-600">
            <p>
              This master&apos;s project explores AI-powered scoliosis detection
              using deep learning, reproducible data workflows, experiment
              tracking, and a FastAPI inference boundary.
            </p>
            <p>
              The frontend stays intentionally separate from model training and
              backend internals. It presents the project, collects an image, and
              displays prediction output for research review.
            </p>
            <p className="rounded-lg border border-orange-200 bg-orange-50 p-4 text-sm leading-6 text-orange-900">
              {PROJECT_DISCLAIMER}
            </p>
          </div>
        </div>
      </section>
    </>
  );
}
