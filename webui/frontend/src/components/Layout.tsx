import {
  Activity,
  Blocks,
  Languages,
  Menu,
  Moon,
  PanelLeftClose,
  Play,
  Scale,
  Sun,
  Wrench,
  X
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { api } from "../api";
import { useI18n } from "../i18n";
import type { Health } from "../types";
import { IconButton } from "./ui";

const links = [
  { to: "/", key: "workbench", icon: Wrench },
  { to: "/runs", key: "runs", icon: Play },
  { to: "/compare", key: "compare", icon: Scale },
  { to: "/extensions", key: "extensions", icon: Blocks }
] as const;

export function Layout() {
  const { t, toggleLanguage } = useI18n();
  const [healthFailed, setHealthFailed] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const [mobileOpen, setMobileOpen] = useState(false);
  const mobileBackdrop = useRef<HTMLDivElement>(null);
  const mobileDrawer = useRef<HTMLElement>(null);
  const [collapsed, setCollapsed] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    const stored = localStorage.getItem("know-surgery-theme");
    return stored === "dark" ? "dark" : "light";
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("know-surgery-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (!mobileOpen || !mobileDrawer.current || !mobileBackdrop.current) return;
    const drawer = mobileDrawer.current;
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const siblings = Array.from(mobileBackdrop.current.parentElement?.children || [])
      .filter((element): element is HTMLElement => element instanceof HTMLElement && element !== mobileBackdrop.current)
      .map((element) => ({ element, inert: element.inert }));
    siblings.forEach(({ element }) => { element.inert = true; });
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const focusable = () => Array.from(drawer.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex="0"]'
    )).filter((element) => element.getClientRects().length > 0);
    (focusable()[0] || drawer).focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setMobileOpen(false);
      } else if (event.key === "Tab") {
        const elements = focusable();
        const first = elements[0] || drawer;
        const last = elements.at(-1) || drawer;
        if (event.shiftKey && (document.activeElement === first || document.activeElement === drawer)) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && (document.activeElement === last || document.activeElement === drawer)) {
          event.preventDefault();
          first.focus();
        }
      }
    };
    const keepFocusInside = (event: FocusEvent) => {
      if (event.target instanceof Node && !drawer.contains(event.target)) (focusable()[0] || drawer).focus();
    };
    const desktop = window.matchMedia("(min-width: 821px)");
    const closeOnDesktop = () => { if (desktop.matches) setMobileOpen(false); };
    document.addEventListener("keydown", handleKeyDown);
    document.addEventListener("focusin", keepFocusInside);
    desktop.addEventListener("change", closeOnDesktop);
    closeOnDesktop();
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.removeEventListener("focusin", keepFocusInside);
      desktop.removeEventListener("change", closeOnDesktop);
      siblings.forEach(({ element, inert }) => { element.inert = inert; });
      document.body.style.overflow = previousOverflow;
      if (previousFocus?.isConnected) previousFocus.focus();
    };
  }, [mobileOpen]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const value = await api.health();
        if (!cancelled) { setHealth(value); setHealthFailed(false); }
      } catch {
        if (!cancelled) { setHealth(null); setHealthFailed(true); }
      }
    };
    void load();
    const timer = window.setInterval(load, 15_000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const navigation = (
    <>
      <div className="brand">
        <div className="brand-symbol" aria-hidden="true">
          <img src="/know-surgery-workbench.png" alt="" width={64} height={64} />
        </div>
        <div className="brand-copy">
          <strong>Know-Surgery</strong>
        </div>
      </div>
      <nav className="primary-nav" aria-label={t("primaryNavigation")}>
        {links.map(({ to, key, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            title={collapsed ? t(key) : undefined}
            onClick={() => setMobileOpen(false)}
          >
            <Icon size={18} aria-hidden="true" />
            <span>{t(key)}</span>
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-spacer" />
      <div className="system-readout">
        <div className="system-status">
          <span
            className={
              "system-pulse " + (health ? "is-ready" : "is-offline")
            }
            aria-hidden="true"
          />
          <div>
            <strong>{health ? t("ready") : healthFailed ? t("connectionLost") : t("loading")}</strong>
            <span>
              {health
                ? health.skills +
                  " " +
                  t("skills")
                : "API"}
            </span>
          </div>
        </div>
        <div className="sidebar-tools">
          <IconButton
            label={theme === "light" ? t("dark") : t("light")}
            icon={theme === "light" ? Moon : Sun}
            onClick={() =>
              setTheme((value) => (value === "light" ? "dark" : "light"))
            }
          />
          <IconButton
            label={
              collapsed ? t("expandNavigation") : t("collapseNavigation")
            }
            icon={PanelLeftClose}
            onClick={() => setCollapsed((value) => !value)}
          />
        </div>
      </div>
    </>
  );

  return (
    <div className={"app-shell " + (collapsed ? "sidebar-collapsed" : "")}>
      <a className="skip-link" href="#main-content">
        {t("skipToContent")}
      </a>
      <aside className="sidebar">{navigation}</aside>
      <header className="mobile-header">
        <button
          className="mobile-brand"
          onClick={() => setMobileOpen(true)}
          aria-label={t("openNavigation")}
          aria-expanded={mobileOpen}
          aria-controls={mobileOpen ? "mobile-navigation" : undefined}
        >
          <Menu size={20} />
          <span className="brand-symbol" aria-hidden="true">
            <img src="/know-surgery-mark.png" alt="" width={24} height={24} />
          </span>
          <strong>Know-Surgery</strong>
        </button>
        <div className="mobile-actions">
          <button
            className="language-toggle"
            aria-label={t("switchLanguage")}
            title={t("switchLanguage")}
            onClick={toggleLanguage}
          >
            <Languages size={17} aria-hidden="true" />
            <span>{t("language")}</span>
          </button>
          <IconButton
            label={theme === "light" ? t("dark") : t("light")}
            icon={theme === "light" ? Moon : Sun}
            onClick={() =>
              setTheme((value) => (value === "light" ? "dark" : "light"))
            }
          />
        </div>
      </header>
      {mobileOpen ? (
        <div
          ref={mobileBackdrop}
          className="mobile-drawer-backdrop"
          onClick={() => setMobileOpen(false)}
        >
          <aside
            ref={mobileDrawer}
            id="mobile-navigation"
            className="mobile-drawer"
            role="dialog"
            aria-modal="true"
            aria-label={t("primaryNavigation")}
            tabIndex={-1}
            onClick={(event) => event.stopPropagation()}
          >
            <IconButton
              label={t("closeNavigation")}
              icon={X}
              onClick={() => setMobileOpen(false)}
            />
            {navigation}
          </aside>
        </div>
      ) : null}
      <main id="main-content" className="main-content">
        <div className="top-context">
          <div className="context-title">
            <Activity size={16} aria-hidden="true" />
            <span>{t("agentFirst")}</span>
          </div>
          <div className="top-context-actions">
            <button
              className="language-toggle"
              aria-label={t("switchLanguage")}
              title={t("switchLanguage")}
              onClick={toggleLanguage}
            >
              <Languages size={16} aria-hidden="true" />
              <span>{t("language")}</span>
            </button>
          </div>
        </div>
        <Outlet context={{ health }} />
      </main>
    </div>
  );
}
