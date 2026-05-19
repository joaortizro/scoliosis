import { Badge } from "@/components/ui/Badge";
import { UploadCard } from "@/components/prediction/UploadCard";
import { PROJECT_DISCLAIMER } from "@/lib/constants";

const workflowSteps = [
  {
    title: "Upload X-ray",
    description: "Select a frontal spine radiograph image from your device.",
  },
  {
    title: "AI model analyzes",
    description: "The frontend sends the image to the FastAPI prediction API.",
  },
  {
    title: "Review result",
    description: "A structured prediction response is displayed for inspection.",
  },
];

export default function Home() {
  return (
    <>
      <section className="border-b border-slate-200 bg-white">
        <div className="mx-auto grid w-full max-w-6xl gap-10 px-6 py-16 lg:grid-cols-[1.05fr_0.95fr] lg:px-8 lg:py-20">
          <div className="flex flex-col justify-center">
            <Badge>Master&apos;s research project</Badge>
            <h1 className="mt-5 max-w-3xl text-4xl font-semibold text-slate-950 sm:text-5xl">
              AI-powered scoliosis detection from spine X-rays
            </h1>
            <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-600">
              A clean research interface for uploading radiographs, sending
              them to a deep learning backend, and reviewing prediction output
              from the scoliosis detection pipeline.
            </p>
            <div className="mt-8 flex flex-col gap-3 sm:flex-row">
              <a
                href="#prediction"
                className="inline-flex h-11 items-center justify-center rounded-md bg-cyan-700 px-5 text-sm font-semibold text-white shadow-sm transition hover:bg-cyan-800 focus:outline-none focus:ring-2 focus:ring-cyan-600 focus:ring-offset-2"
              >
                Try image prediction
              </a>
              <a
                href="#methodology"
                className="inline-flex h-11 items-center justify-center rounded-md border border-slate-300 bg-white px-5 text-sm font-semibold text-slate-800 transition hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-cyan-600 focus:ring-offset-2"
              >
                View methodology
              </a>
            </div>
          </div>

          <div className="rounded-lg border border-slate-200 bg-slate-900 p-5 text-white shadow-sm">
            <div className="grid min-h-[360px] gap-3 rounded-md border border-white/10 bg-[radial-gradient(circle_at_50%_18%,rgba(34,211,238,0.2),transparent_32%),linear-gradient(180deg,#0f172a,#111827)] p-5">
              <div className="flex items-start justify-between text-sm text-slate-300">
                <span>Research preview</span>
                <span>FastAPI ready</span>
              </div>
              <div className="mx-auto flex h-full w-full max-w-[260px] items-center justify-center">
                <div className="relative h-[280px] w-28 rounded-full border border-cyan-200/30 bg-white/5">
                  <div className="absolute left-1/2 top-5 h-[240px] w-3 -translate-x-1/2 rounded-full bg-cyan-100/80 shadow-[0_0_30px_rgba(125,211,252,0.35)]" />
                  {Array.from({ length: 11 }).map((_, index) => (
                    <div
                      key={index}
                      className="absolute left-1/2 h-2 w-20 -translate-x-1/2 rounded-full bg-slate-200/70"
                      style={{
                        top: `${36 + index * 20}px`,
                        transform: `translateX(-50%) rotate(${
                          index % 2 === 0 ? -5 : 5
                        }deg)`,
                      }}
                    />
                  ))}
                </div>
              </div>
              <p className="text-sm leading-6 text-slate-300">
                {PROJECT_DISCLAIMER}
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-slate-50">
        <div className="mx-auto w-full max-w-6xl px-6 py-14 lg:px-8">
          <div className="grid gap-4 md:grid-cols-3">
            {workflowSteps.map((step, index) => (
              <div
                key={step.title}
                className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm"
              >
                <div className="flex h-9 w-9 items-center justify-center rounded-md bg-cyan-50 text-sm font-bold text-cyan-800">
                  {index + 1}
                </div>
                <h2 className="mt-4 text-lg font-semibold text-slate-950">
                  {step.title}
                </h2>
                <p className="mt-2 text-sm leading-6 text-slate-600">
                  {step.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="prediction" className="border-y border-slate-200 bg-white">
        <div className="mx-auto grid w-full max-w-6xl gap-8 px-6 py-14 lg:grid-cols-[0.8fr_1.2fr] lg:px-8">
          <div>
            <Badge>Prediction</Badge>
            <h2 className="mt-4 text-3xl font-semibold text-slate-950">
              Upload an image for analysis
            </h2>
            <p className="mt-4 leading-7 text-slate-600">
              Choose a radiograph image and submit it to the backend endpoint at
              <span className="font-medium text-slate-800"> POST /predict/</span>.
              The result card accepts flexible response fields while the backend
              contract evolves.
            </p>
            <p className="mt-5 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm leading-6 text-amber-900">
              {PROJECT_DISCLAIMER}
            </p>
          </div>
          <UploadCard />
        </div>
      </section>

      <section id="methodology" className="bg-slate-50">
        <div className="mx-auto w-full max-w-6xl px-6 py-14 lg:px-8">
          <div className="max-w-3xl">
            <Badge>About the system</Badge>
            <h2 className="mt-4 text-3xl font-semibold text-slate-950">
              Methodology and project context
            </h2>
            <p className="mt-4 leading-7 text-slate-600">
              This frontend supports a master&apos;s project focused on
              AI-powered scoliosis detection. The repository contains a
              reproducible deep learning pipeline, experiment tracking with
              MLflow, data versioning with DVC, and a FastAPI backend for model
              inference.
            </p>
            <p className="mt-4 leading-7 text-slate-600">
              The website stays intentionally small: presentation, upload,
              preview, API submission, and response display are kept separate
              from the training and evaluation code.
            </p>
          </div>
        </div>
      </section>
    </>
  );
}
