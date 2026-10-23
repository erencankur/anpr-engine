"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { createContext, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { flushSync } from "react-dom";

export type Locale = "en" | "tr";
type Theme = "dark" | "light";
const LocaleContext = createContext<Locale>("tr");

export function useLocale(): Locale { return useContext(LocaleContext); }

const copy = {
  en: { home: "ANPR Engine home", navigation: "Contact, language and appearance", language: "Switch to Turkish", contactUs: "Contact us", light: "Light", dark: "Dark", switchLight: "Switch to light theme", switchDark: "Switch to dark theme", information: "Information", privacy: "Privacy", responsible: "Responsible use", limitations: "Limitations", repo: "Repository", contact: "Contact" },
  tr: { home: "ANPR Engine ana sayfa", navigation: "İletişim, dil ve görünüm", language: "İngilizceye geç", contactUs: "İletişime Geç", light: "Açık", dark: "Koyu", switchLight: "Açık temaya geç", switchDark: "Koyu temaya geç", information: "Bilgilendirme", privacy: "Gizlilik", responsible: "Sorumlu kullanım", limitations: "Sınırlamalar", repo: "Depo", contact: "İletişim" }
} as const;

const pageTitles = {
  en: { home: "ANPR Engine — Plate detection and recognition demonstration", privacy: "Privacy · ANPR Engine", "responsible-use": "Responsible use · ANPR Engine", limitations: "Limitations · ANPR Engine" },
  tr: { home: "ANPR Engine — Plaka tespit ve tanıma gösterimi", privacy: "Gizlilik · ANPR Engine", "responsible-use": "Sorumlu kullanım · ANPR Engine", limitations: "Sınırlamalar · ANPR Engine" }
} as const;

export function SiteChrome({ initialTheme, children }: { initialTheme: Theme; children: ReactNode }) {
  const [locale, setLocale] = useState<Locale>("tr");
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const pathname = usePathname();
  const t = copy[locale];

  useEffect(() => {
    const topic = pathname.split("/").at(-1);
    document.documentElement.lang = locale;
    document.title = topic === "privacy" || topic === "responsible-use" || topic === "limitations" ? pageTitles[locale][topic] : pageTitles[locale].home;
  }, [locale, pathname]);

  function selectLanguage(next: Locale) {
    if (next === locale) return;
    const applyLanguage = () => {
      document.documentElement.lang = next;
      flushSync(() => setLocale(next));
    };
    if (document.startViewTransition && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      document.startViewTransition(applyLanguage);
    } else {
      applyLanguage();
    }
  }

  function selectTheme(next: Theme) {
    if (next === theme) return;
    const applyTheme = () => {
      document.documentElement.dataset.theme = next;
      document.cookie = `anpr_theme=${next}; Path=/; Max-Age=31536000; SameSite=Strict${location.protocol === "https:" ? "; Secure" : ""}`;
      flushSync(() => setTheme(next));
    };

    if (document.startViewTransition && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      document.startViewTransition(applyTheme);
    } else {
      applyTheme();
    }
  }

  return <LocaleContext.Provider value={locale}>
    <header className="site-header"><div className="shell header-inner">
      <Link className="brand" href="/" aria-label={t.home}><Image src="/icon.png" alt="" width={38} height={38} priority /><span className="brand-wordmark">ANPR <span className="brand-accent">Engine</span><small>BY VIBESOFTERS</small></span></Link>
      <nav className="header-controls" aria-label={t.navigation}>
        <a className="header-control contact-control" href="https://vibesofters.com">{t.contactUs}</a>
        <div className="header-segment language-control" role="group" aria-label={locale === "tr" ? "Dil seçimi" : "Language"}>
          <button type="button" lang="tr" aria-label="Türkçe" aria-pressed={locale === "tr"} onClick={() => selectLanguage("tr")}>TR</button>
          <button type="button" lang="en" aria-label="English" aria-pressed={locale === "en"} onClick={() => selectLanguage("en")}>EN</button>
        </div>
        <div className="header-segment theme-control" role="group" aria-label={locale === "tr" ? "Tema seçimi" : "Appearance"}>
          <button type="button" aria-label={t.switchLight} title={t.light} aria-pressed={theme === "light"} onClick={() => selectTheme("light")}><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/></svg></button>
          <button type="button" aria-label={t.switchDark} title={t.dark} aria-pressed={theme === "dark"} onClick={() => selectTheme("dark")}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20.5 14.2A8.8 8.8 0 0 1 9.8 3.5a8.8 8.8 0 1 0 10.7 10.7Z"/></svg></button>
        </div>
      </nav>
    </div></header>
    {children}
    <footer className="site-footer">
      <div className="shell footer-signature"><div><p className="eyebrow">VIBESOFTERS / VISION LAB</p><p className="footer-title">{locale === "en" ? "A different perspective." : "Farklı bir bakış açısı."}</p></div><a className="footer-round-link" href="https://vibesofters.com" aria-label={t.contactUs}>↗</a></div>
      <div className="shell footer-inner"><span>© 2026 VibeSofters</span><nav aria-label={t.information}><Link href="/information/privacy">{t.privacy}</Link><Link href="/information/responsible-use">{t.responsible}</Link><Link href="/information/limitations">{t.limitations}</Link><a href="https://github.com/erencankur/anpr-engine" rel="noreferrer">{t.repo}</a><a href="mailto:erncnkr@gmail.com">{t.contact}</a></nav></div>
    </footer>
  </LocaleContext.Provider>;
}
