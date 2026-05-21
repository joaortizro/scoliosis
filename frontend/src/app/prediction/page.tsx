import { readdir } from "node:fs/promises";
import { join, parse } from "node:path";
import { UploadCard } from "@/components/prediction/UploadCard";

export const dynamic = "force-dynamic";

const SAMPLE_IMAGE_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".webp"]);

function formatSampleLabel(filename: string) {
  const name = parse(filename).name.replace(/[-_]+/g, " ").trim();

  if (!name) {
    return "Example X-ray";
  }

  return name.replace(/\b\w/g, (character) => character.toUpperCase());
}

async function getSampleImages() {
  const imagesDirectory = join(process.cwd(), "public", "images");

  try {
    const entries = await readdir(imagesDirectory, { withFileTypes: true });

    return entries
      .filter((entry) => entry.isFile())
      .map((entry) => entry.name)
      .filter((filename) =>
        SAMPLE_IMAGE_EXTENSIONS.has(parse(filename).ext.toLowerCase()),
      )
      .sort((first, second) => first.localeCompare(second))
      .slice(0, 3)
      .map((filename) => ({
        alt: `${formatSampleLabel(filename)} sample spine X-ray`,
        label: formatSampleLabel(filename),
        src: `/images/${filename}`,
      }));
  } catch {
    return [];
  }
}

export default async function PredictionPage() {
  const sampleImages = await getSampleImages();

  return (
    <section className="h-[calc(100vh-4rem)] overflow-hidden bg-white">
      <UploadCard sampleImages={sampleImages} />
    </section>
  );
}
