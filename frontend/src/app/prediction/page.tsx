import { Badge } from "@/components/ui/Badge";
import { UploadCard } from "@/components/prediction/UploadCard";
import { PROJECT_DISCLAIMER } from "@/lib/constants";

export default function PredictionPage() {
  return (
    <section className="bg-[#fbfcfd]">
      <div className="mx-auto grid w-full max-w-7xl gap-8 px-5 py-12 lg:grid-cols-[0.78fr_1.22fr] lg:px-8 lg:py-16">
        <div>
          <Badge>Prediction workspace</Badge>
          <h1 className="mt-5 text-4xl font-semibold leading-tight text-[#102a43] sm:text-5xl">
            Upload an X-ray for research analysis.
          </h1>
          <p className="mt-5 text-lg leading-8 text-slate-600">
            Select a radiograph image, preview it, and submit it to the prepared
            prediction API boundary. Authentication and backend connectivity can
            be added later without changing the page structure.
          </p>
          <p className="mt-6 rounded-lg border border-orange-200 bg-orange-50 p-4 text-sm leading-6 text-orange-900">
            {PROJECT_DISCLAIMER}
          </p>
        </div>
        <UploadCard />
      </div>
    </section>
  );
}
