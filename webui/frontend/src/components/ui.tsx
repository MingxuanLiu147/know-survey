import {
  Check,
  Clipboard,
  LoaderCircle,
  TriangleAlert,
  type LucideIcon
} from "lucide-react";
import {
  useEffect,
  useState,
  type ButtonHTMLAttributes,
  type ReactNode
} from "react";
import { useI18n } from "../i18n";
import type { EvidenceLevel, Operation } from "../types";

export function Button({
  children,
  variant = "primary",
  icon: Icon,
  loading = false,
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  icon?: LucideIcon;
  loading?: boolean;
}) {
  return (
    <button
      className={"button button-" + variant + " " + className}
      {...props}
      disabled={loading || props.disabled}
    >
      {loading ? (
        <LoaderCircle size={16} className="spin" aria-hidden="true" />
      ) : Icon ? (
        <Icon size={16} aria-hidden="true" />
      ) : null}
      <span>{children}</span>
    </button>
  );
}

export function IconButton({
  label,
  icon: Icon,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  label: string;
  icon: LucideIcon;
}) {
  return (
    <button
      className="icon-button"
      aria-label={label}
      title={label}
      {...props}
    >
      <Icon size={18} aria-hidden="true" />
    </button>
  );
}

export function StatusBadge({
  level,
  compact = false
}: {
  level: EvidenceLevel | string;
  compact?: boolean;
}) {
  const { t } = useI18n();
  const label =
    level in {
      planned: true,
      configured: true,
      local_validation: true,
      verified_smoke: true,
      verified_result: true
    }
      ? t(level as EvidenceLevel)
      : level.replaceAll("_", " ");
  return (
    <span
      className={"status-badge status-" + level + (compact ? " compact" : "")}
    >
      <span className="status-dot" aria-hidden="true" />
      {label}
    </span>
  );
}

export function OperationBadge({
  operation
}: {
  operation: Operation | "evaluation" | "unknown";
}) {
  return (
    <span className={"operation-badge operation-" + operation}>
      {operation}
    </span>
  );
}

export function SectionHeader({
  eyebrow,
  title,
  action,
  description
}: {
  eyebrow?: string;
  title: string;
  action?: ReactNode;
  description?: string;
}) {
  return (
    <div className="section-header">
      <div>
        {eyebrow ? <div className="eyebrow">{eyebrow}</div> : null}
        <h2>{title}</h2>
        {description ? (
          <p className="section-description">{description}</p>
        ) : null}
      </div>
      {action ? <div className="section-action">{action}</div> : null}
    </div>
  );
}

export function CopyButton({ text }: { text: string }) {
  const { t } = useI18n();
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return;
    const timer = window.setTimeout(() => setCopied(false), 1600);
    return () => window.clearTimeout(timer);
  }, [copied]);
  return (
    <IconButton
      label={copied ? t("copied") : t("copy")}
      icon={copied ? Check : Clipboard}
      onClick={() => {
        void navigator.clipboard.writeText(text);
        setCopied(true);
      }}
    />
  );
}

export function InlineError({ message }: { message: string }) {
  return (
    <div className="inline-message inline-error" role="alert">
      <TriangleAlert size={17} aria-hidden="true" />
      <span>{message}</span>
    </div>
  );
}

export function LoadingBlock({ label }: { label: string }) {
  return (
    <div className="loading-block" role="status">
      <LoaderCircle size={20} className="spin" aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}

export function EmptyState({
  icon: Icon,
  title,
  description
}: {
  icon: LucideIcon;
  title: string;
  description?: string;
}) {
  return (
    <div className="empty-state">
      <Icon size={24} aria-hidden="true" />
      <strong>{title}</strong>
      {description ? <span>{description}</span> : null}
    </div>
  );
}

