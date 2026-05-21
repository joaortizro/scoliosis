import { Badge } from "@/components/ui/Badge";
import { UploadCard } from "@/components/prediction/UploadCard";
import { PROJECT_DISCLAIMER } from "@/lib/constants";

export default function PredictionPage() {
  return (
    <section className="bg-[#fbfcfd]">
      <div className="mx-auto flex min-h-[calc(100vh-144px)] w-full max-w-5xl flex-col items-center justify-center gap-8 px-5 py-10 lg:px-8">
        <div className="max-w-3xl text-center">
          <Badge>Flash prediction</Badge>
          <h1 className="mt-5 text-3xl font-semibold leading-tight text-[#102a43] sm:text-5xl">
            Upload an X-ray and view the segmented result.
          </h1>
          <p className="mx-auto mt-5 max-w-2xl text-base leading-7 text-slate-600 sm:text-lg">
            Select a JPG or PNG radiograph. The image is sent to the
            segmentation API and the returned overlay can be compared with the
            original upload.
          </p>
          <p className="mx-auto mt-6 max-w-2xl rounded-lg border border-orange-200 bg-orange-50 p-4 text-sm leading-6 text-orange-900">
            {PROJECT_DISCLAIMER}
          </p>
        </div>
        <UploadCard />
      </div>
    </section>
  );
}
