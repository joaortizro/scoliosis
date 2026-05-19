import { Card } from "@/components/ui/Card";
import type { PredictionResponse, PredictionState } from "@/types/prediction";

type PredictionResultProps = {
  state: PredictionState;
};

function getPrimaryLabel(result: PredictionResponse) {
  return (
    result.prediction ??
    result.label ??
    result.message ??
    "Prediction response received"
  );
}

function getConfidence(result: PredictionResponse) {
  const value = result.confidence ?? result.probability;

  if (typeof value !== "number") {
    return null;
  }

  return `${Math.round(value * 100)}%`;
}

export function PredictionResult({ state }: PredictionResultProps) {
  if (state.status === "idle") {
    return (
      <Card className="p-5">
        <h3 className="text-lg font-semibold text-slate-950">
          Result placeholder
        </h3>
        <p className="mt-2 text-sm leading-6 text-slate-600">
          Upload an image and submit it to view the backend prediction response.
        </p>
      </Card>
    );
  }

  if (state.status === "loading") {
    return (
      <Card className="p-5">
        <h3 className="text-lg font-semibold text-slate-950">Analyzing image</h3>
        <p className="mt-2 text-sm leading-6 text-slate-600">
          Sending the selected file to the prediction API.
        </p>
      </Card>
    );
  }

  if (state.status === "error") {
    return (
      <Card className="border-red-200 bg-red-50 p-5">
        <h3 className="text-lg font-semibold text-red-950">
          Prediction failed
        </h3>
        <p className="mt-2 text-sm leading-6 text-red-800">{state.error}</p>
      </Card>
    );
  }

  const confidence = getConfidence(state.data);

  return (
    <Card className="p-5">
      <h3 className="text-lg font-semibold text-slate-950">
        {getPrimaryLabel(state.data)}
      </h3>
      <dl className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
        {confidence ? (
          <div className="rounded-md bg-slate-50 p-3">
            <dt className="font-medium text-slate-500">Confidence</dt>
            <dd className="mt-1 font-semibold text-slate-950">{confidence}</dd>
          </div>
        ) : null}
        {typeof state.data.cobb_angle === "number" ? (
          <div className="rounded-md bg-slate-50 p-3">
            <dt className="font-medium text-slate-500">Cobb angle</dt>
            <dd className="mt-1 font-semibold text-slate-950">
              {state.data.cobb_angle.toFixed(1)} degrees
            </dd>
          </div>
        ) : null}
      </dl>
      <details className="mt-4">
        <summary className="cursor-pointer text-sm font-medium text-cyan-800">
          Raw API response
        </summary>
        <pre className="mt-3 max-h-72 overflow-auto rounded-md bg-slate-950 p-4 text-xs leading-5 text-slate-100">
          {JSON.stringify(state.data, null, 2)}
        </pre>
      </details>
    </Card>
  );
}
