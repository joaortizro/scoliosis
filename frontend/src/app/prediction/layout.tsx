export default function PredictionLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return <div className="prediction-workspace">{children}</div>;
}
