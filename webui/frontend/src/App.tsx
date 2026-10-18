import { lazy, Suspense } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { LoadingBlock } from "./components/ui";
import { I18nProvider, useI18n } from "./i18n";

const Workbench = lazy(() =>
  import("./pages/Workbench").then((module) => ({
    default: module.Workbench
  }))
);
const Runs = lazy(() =>
  import("./pages/Runs").then((module) => ({ default: module.Runs }))
);
const Compare = lazy(() =>
  import("./pages/Compare").then((module) => ({ default: module.Compare }))
);
const Extensions = lazy(() =>
  import("./pages/Extensions").then((module) => ({
    default: module.Extensions
  }))
);

function AppRoutes() {
  const { t } = useI18n();
  return (
    <BrowserRouter>
      <Suspense fallback={<LoadingBlock label={t("loading")} />}>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Workbench />} />
            <Route path="runs" element={<Runs />} />
            <Route path="compare" element={<Compare />} />
            <Route path="extensions" element={<Extensions />} />
          </Route>
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}

export default function App() {
  return (
    <I18nProvider>
      <AppRoutes />
    </I18nProvider>
  );
}
