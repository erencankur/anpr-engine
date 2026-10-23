import type { Metadata } from "next";
import notice from "../../../../packages/api-contract/fixtures/valid/processing-notice.json";
import { Workspace } from "@/components/workspace";

const origin = "https://anpr-engine.vibesofters.com";

export const metadata: Metadata = {
  title: "ANPR Engine — Plaka tespit ve tanıma gösterimi",
  description: "Tek görüntüden plaka tespiti ve tanıma gösterimi. Her sonuç insan incelemesi gerektirir.",
  alternates: { canonical: origin },
  openGraph: { title: "ANPR Engine", description: "İnsan incelemesi gerektiren tek görüntülü plaka tanıma gösterimi.", url: origin, locale: "tr_TR", type: "website" },
  robots: { index: false, follow: false }
};

export default function RootPage() {
  const inferenceAvailable = process.env.NODE_ENV !== "production" || process.env.ANPR_LOCAL_DIRECT_MODE === "1" ||
    (process.env.ANPR_TRUSTED_PROXY_IP_HEADER === "x-real-ip" && Boolean(process.env.ANPR_DAILY_QUOTA_PATH));
  return <Workspace notices={notice.notices} inferenceAvailable={inferenceAvailable} />;
}
