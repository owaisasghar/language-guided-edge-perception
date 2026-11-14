import type { ReactNode } from "react";

export function PageHead({ title, sub, actions }: { title: string; sub?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="page-head">
      <div>
        <h1 className="page-title">{title}</h1>
        {sub && <p className="page-sub">{sub}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  );
}

export function Card({ title, sub, icon, actions, children, bodyClass = "card-b" }: {
  title?: string; sub?: ReactNode; icon?: ReactNode; actions?: ReactNode; children: ReactNode; bodyClass?: string;
}) {
  return (
    <section className="card">
      {(title || actions) && (
        <div className="card-h">
          {icon && <span className="card-icon">{icon}</span>}
          <div className="grow">
            {title && <h2>{title}</h2>}
            {sub && <div className="sub">{sub}</div>}
          </div>
          {actions}
        </div>
      )}
      <div className={bodyClass}>{children}</div>
    </section>
  );
}
