# SPDX-License-Identifier: Apache-2.0
"""ATK's Cognitive Coder panel. Destination: `atk/ui/ccoder_panel.py`.

**This is the only file in the adapter that imports Qt**, which is what keeps
`ccoder_host.py` testable without a QApplication.

ATK's UI doctrine, honoured rather than reinvented (§7.1):

  * **One tab per function, no pop-ups to go hunting for.** This is a
    workspace tab with sub-tabs, not a dialog. The single justified modal is
    the diff approval, and the owner has chosen auto-apply, so in practice
    even that does not appear.
  * **Detachable panes.** The console, the diff view and the CodeMap tree
    register with `atk/ui/detach.py` so they can be popped to another
    monitor — which is how anyone actually watches a build.
  * **The GUI thread never blocks.** `Session.step()` runs on a `QRunnable`
    via `atk/core/workers.py`; `EventPort` calls arrive on the worker thread
    and are marshalled back with signals. `Session.cancel()` is the ONE
    cross-thread call (§5.2), and it is what the Stop button does.

**"Attach a screenshot" is here on purpose** (§7.2). Devstral is multimodal
and the owner habitually debugs by screenshot — the clipped Setup page, the
sliced RF rail, the `(x0.001)` axis label were all caught that way and none
would have been caught by reading code. A coding agent that can be handed a
picture of a broken UI *and* the code that renders it is a different tool
from one that can only read source. It routes through `Message.images`;
a host without vision ignores it and `capabilities()` says so, which the
panel surfaces rather than silently dropping the image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction, QFont, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .ccoder_host import ATK_CONVENTIONS, build_host, build_session, preflight


class _Bridge(QObject):
    """Marshals worker-thread events onto the GUI thread.

    Every `EventPort` call arrives on a `QRunnable`'s thread. Touching a
    widget from there is the classic Qt crash — intermittent, unreproducible,
    and blamed on everything except the real cause. These signals are the
    whole fix: emit from anywhere, connect with the default
    `AutoConnection`, and Qt queues them onto the GUI thread.
    """

    status = Signal(str)
    console = Signal(str, str)
    flow = Signal(object)
    remote = Signal(str)
    finished = Signal(object)
    failed = Signal(str)


class CognitiveCoderPanel(QWidget):
    """The workspace tab. Replaces the Developer Sandbox's engine, keeps the tab."""

    def __init__(self, ctx: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.session: Any = None
        self._worker: Any = None
        self._images: list[tuple[bytes, str]] = []
        self._bridge = _Bridge()
        self._build_ui()
        self._connect()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        # -- the REMOTE banner. Persistent, and it stays up while true. ---
        # M42.6. It is the first widget so it cannot be scrolled out of
        # sight, and it is invisible rather than empty when remote is off —
        # a banner that is always present is a banner nobody reads.
        self.remote_banner = QLabel()
        self.remote_banner.setWordWrap(True)
        self.remote_banner.setStyleSheet(
            "background:#7f1d1d; color:#fff; padding:6px; font-weight:bold;")
        self.remote_banner.setVisible(False)
        outer.addWidget(self.remote_banner)

        # -- the request row ---------------------------------------------
        row = QHBoxLayout()
        row.addWidget(QLabel("Build:"))
        self.request = QLineEdit()
        self.request.setPlaceholderText(
            "what you want built, in a sentence — e.g. a CSV parser with tests")
        row.addWidget(self.request, 1)

        self.language = QComboBox()
        row.addWidget(self.language)

        self.attach_button = QPushButton("Attach screenshot…")
        self.attach_button.setToolTip(
            "Devstral is multimodal. A picture of a broken dialog plus the "
            "code that renders it catches things reading the source cannot.")
        row.addWidget(self.attach_button)

        self.go = QPushButton("Build")
        self.stop = QPushButton("Stop")
        self.stop.setEnabled(False)
        row.addWidget(self.go)
        row.addWidget(self.stop)
        outer.addLayout(row)

        # -- the model line ----------------------------------------------
        self.model_label = QLabel("no model loaded")
        self.model_label.setStyleSheet("color:#94a3b8;")
        outer.addWidget(self.model_label)

        # -- sub-tabs, not dialogs (ATK doctrine D4) ---------------------
        self.tabs = QTabWidget()

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("Consolas", 9))
        self.tabs.addTab(self.console, "Console")

        self.diff = QPlainTextEdit()
        self.diff.setReadOnly(True)
        self.diff.setFont(QFont("Consolas", 9))
        self.tabs.addTab(self.diff, "Changes")

        self.codemap = QTreeWidget()
        self.codemap.setHeaderLabels(["Symbol", "Kind", "Where"])
        self.tabs.addTab(self.codemap, "CodeMap")

        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.tabs.addTab(self.report, "Recommendation")

        self.history = QTreeWidget()
        self.history.setHeaderLabels(["#", "State", "Task", "Files"])
        self.tabs.addTab(self.history, "History")

        outer.addWidget(self.tabs, 1)

        # -- the honest footer --------------------------------------------
        self.footer = QLabel("")
        self.footer.setWordWrap(True)
        self.footer.setStyleSheet("color:#94a3b8; padding-top:4px;")
        outer.addWidget(self.footer)

        self._register_detachable()

    def _register_detachable(self) -> None:
        """Pop the console, diff and CodeMap to another monitor (§7.1)."""
        try:
            from atk.ui.detach import register_detachable
            register_detachable(self.console, "Cognitive Coder — Console")
            register_detachable(self.diff, "Cognitive Coder — Changes")
            register_detachable(self.codemap, "Cognitive Coder — CodeMap")
        except Exception:                                # noqa: BLE001
            # Detaching is a convenience; the panel works without it, and a
            # missing helper must not stop the tab from loading.
            pass

    def _connect(self) -> None:
        self.go.clicked.connect(self.start_build)
        self.stop.clicked.connect(self.stop_build)
        self.attach_button.clicked.connect(self.attach_screenshot)
        self._bridge.status.connect(self._on_status)
        self._bridge.console.connect(self._on_console)
        self._bridge.flow.connect(self._on_flow)
        self._bridge.remote.connect(self._on_remote)
        self._bridge.finished.connect(self._on_finished)
        self._bridge.failed.connect(self._on_failed)
        self.refresh_languages()

    # ------------------------------------------------------------------
    def refresh_languages(self) -> None:
        """Only the languages whose toolchain is present RIGHT NOW (§6.1).

        Offering Rust in the dropdown on a machine with no `rustc` is a
        promise the engine cannot keep, and the operator finds out three
        minutes later.
        """
        from cognitive_coder import langs

        from .ccoder_host import ATKExec

        ex = ATKExec()
        self.language.clear()
        available = set(langs.available_ids(ex))
        for lang_id, label in langs.labels():
            if lang_id in available:
                self.language.addItem(label, lang_id)
            else:
                lang = langs.get(lang_id)
                self.language.addItem(f"{label} — not installed", lang_id)
                index = self.language.count() - 1
                self.language.model().item(index).setEnabled(False)
                self.language.setItemData(
                    index, lang.install_hint or "no toolchain found",
                    Qt.ToolTipRole)

    def refresh_model(self) -> None:
        """M13/M10 — what is loaded RIGHT NOW, including nothing."""
        engine = getattr(self.ctx, "llm_engine", None)
        if engine is None or not getattr(engine, "is_loaded", False):
            self.model_label.setText(
                "No model loaded — load one in Setup. If Whisper has the "
                "VRAM, unload it first: at 16 GB they are mutually "
                "exclusive.")
            self.go.setEnabled(False)
            return
        meta = dict(getattr(engine, "metadata", {}) or {})
        self.model_label.setText(
            f"{meta.get('model_file', 'a model')} · "
            f"{meta.get('n_ctx', '?')} tokens of context")
        self.go.setEnabled(True)

    # ------------------------------------------------------------------
    def attach_screenshot(self) -> None:
        """§7.2 — a picture of the bug alongside the code that causes it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Attach a screenshot", "",
            "Images (*.png *.jpg *.jpeg *.webp)")
        if not path:
            return
        engine = getattr(self.ctx, "llm_engine", None)
        if engine is not None and not getattr(engine, "has_vision", False):
            # C7: say what the absence costs rather than dropping it quietly.
            QMessageBox.information(
                self, "No vision model loaded",
                "The loaded model cannot see images, so this screenshot "
                "would be ignored. Load a multimodal model — Devstral with "
                "its mmproj file — and attach it again.")
            return
        media = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        self._images.append((Path(path).read_bytes(), media))
        self.footer.setText(
            f"{len(self._images)} screenshot(s) attached to the next build.")

    # ------------------------------------------------------------------
    def start_build(self) -> None:
        request = self.request.text().strip()
        if not request:
            return
        engine = getattr(self.ctx, "llm_engine", None)
        project = getattr(self.ctx, "project_root", None) or str(Path.cwd())

        problems = preflight(engine, project)
        if problems:
            self._on_console("warning", "\n".join(problems))
            if any("No model is loaded" in p for p in problems):
                return

        host = build_host(
            self.ctx, engine, project,
            auto_apply=self._auto_apply_enabled(),
            status=self._bridge.status.emit,
            console=lambda kind, text: self._bridge.console.emit(kind, text),
            flow=self._bridge.flow.emit,
            remote_banner=self._bridge.remote.emit,
            ask_diff=None,          # auto-apply is the owner's choice
            ask_remote=self._ask_remote)
        self.session = build_session(
            host, lang=self.language.currentData() or "python",
            conventions=ATK_CONVENTIONS)

        self.console.clear()
        self.go.setEnabled(False)
        self.stop.setEnabled(True)

        # The GUI thread never blocks (§7.1). Everything below runs on the
        # pool, and every widget touch comes back through _Bridge.
        try:
            from atk.core.workers import Worker, submit

            def work() -> Any:
                self.session.run(request, self._profile())
                return self.session

            worker = Worker(work, description="Cognitive Coder build",
                            kind="inference")
            worker.signals.result.connect(self._bridge.finished.emit)
            worker.signals.error.connect(self._bridge.failed.emit)
            self._worker = submit(worker)
        except Exception as exc:                         # noqa: BLE001
            self._on_failed(str(exc))

    def stop_build(self) -> None:
        """The ONE cross-thread call (§5.2), and the whole of the Stop button.

        It sets a token the engine checks at every phase boundary. Work
        already verified is kept; anything half-applied is rolled back.
        """
        if self.session is not None:
            self.session.cancel()
            self._on_status("Stopping at the next safe point…")

    # ------------------------------------------------------------------
    def _auto_apply_enabled(self) -> bool:
        """Setup → System & Resources → Advanced (§6.5).

        Default FALSE here even though the owner has chosen auto-apply,
        because the library default is approval-required and a panel that
        silently inverts it is the wrong place to make that decision.
        """
        settings = getattr(self.ctx, "settings", {}) or {}
        return bool(settings.get("ccoder", {}).get("auto_apply", False))

    def _profile(self) -> dict:
        settings = getattr(self.ctx, "settings", {}) or {}
        profile = dict(settings.get("ccoder", {}).get("profile", {}))
        profile.setdefault("skill_level", "senior")
        return profile

    def _ask_remote(self, provider: str, bytes_out: int,
                    estimate: str) -> bool:
        """The one thing that always asks, whatever the diff setting says.

        C3 is ATK's core promise. Auto-approving outbound traffic in an
        air-gapped tool would be a contradiction, so this is a real modal
        with a real default of No.
        """
        answer = QMessageBox.question(
            self, "Send data off this machine?",
            f"{provider} would receive about {bytes_out:,} bytes.\n\n"
            f"{estimate}\n\n"
            f"ATK is offline by default. Nothing has been sent yet.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return answer == QMessageBox.Yes

    # ------------------------------------------------------------------
    # slots — all on the GUI thread
    # ------------------------------------------------------------------
    def _on_status(self, message: str) -> None:
        setter = getattr(self.ctx, "set_status", None)
        if callable(setter):
            setter(message)
        self.footer.setText(message)

    def _on_console(self, kind: str, text: str) -> None:
        if kind == "token":
            self.console.moveCursor(QTextCursor.End)
            self.console.insertPlainText(text)
            return
        prefix = {"error": "!!", "warning": " !", "patch": " +",
                  "diagnostic": " ×"}.get(kind, "  ")
        self.console.appendPlainText(f"{prefix} {text}")
        if kind == "patch":
            self._refresh_history()

    def _on_flow(self, step: Any) -> None:
        emit = getattr(self.ctx, "cognitive_flow", None)
        if callable(emit):
            emit(step)

    def _on_remote(self, banner: str) -> None:
        self.remote_banner.setText(banner)
        self.remote_banner.setVisible(bool(banner))

    def _on_finished(self, session: Any) -> None:
        self.go.setEnabled(True)
        self.stop.setEnabled(False)
        self._images.clear()
        self.console.appendPlainText("\n" + session.report())
        self._refresh_history()
        self._refresh_codemap()
        self._show_recommendation()

    def _on_failed(self, detail: str) -> None:
        self.go.setEnabled(True)
        self.stop.setEnabled(False)
        # C6: the operator sees a sentence. The traceback goes to the log.
        self._on_status("The build stopped with an unexpected error. The "
                        "details are in the log.")
        self.console.appendPlainText(f"!! {detail.strip().splitlines()[-1]}")

    # ------------------------------------------------------------------
    def _refresh_history(self) -> None:
        """"What did it just do to my project", answered from the log (§6.5)."""
        if self.session is None:
            return
        self.history.clear()
        for record in self.session.history():
            item = QTreeWidgetItem([
                str(record.seq),
                record.state + (" · sealed" if record.sealed else ""),
                record.task_id, ", ".join(record.files)])
            self.history.addTopLevelItem(item)
        diffs = [tx.diff for tx in getattr(self.session.patcher, "_open", [])
                 if getattr(tx, "diff", "")]
        if diffs:
            self.diff.setPlainText("\n".join(diffs))

    def _refresh_codemap(self) -> None:
        if self.session is None:
            return
        self.codemap.clear()
        store = self.session.codemap.store
        for row in store.files():
            parent = QTreeWidgetItem([row["path"], row["lang"], ""])
            if row.get("approximate"):
                # C7 again: the label survives all the way to the UI.
                parent.setText(2, "outline is pattern-matched, not parsed")
            for symbol in store.symbols_in(row["path"]):
                parent.addChild(QTreeWidgetItem([
                    symbol["name"], symbol["kind"], f"line {symbol['line']}"]))
            self.codemap.addTopLevelItem(parent)
        stats = self.session.codemap.stats()
        self.footer.setText(stats.one_line())

    def _show_recommendation(self) -> None:
        if self.session is None:
            return
        path = self.session.config.recommendation_path
        try:
            self.report.setMarkdown(self.session.host.fs.read(path))
            self.tabs.setCurrentWidget(self.report)
        except Exception:                                # noqa: BLE001
            self.report.setPlainText(
                "No recommendation document was produced — the review stage "
                "runs only after everything builds and its tests pass.")


def make_actions(panel: CognitiveCoderPanel) -> list[QAction]:
    """Menu entries ATK's shell can hang off the panel."""
    doctor = QAction("Cognitive Coder: what can this machine build?", panel)

    def show_doctor() -> None:
        from cognitive_coder import langs

        from .ccoder_host import ATKExec
        ex = ATKExec()
        usable = langs.available_ids(ex)
        missing = [langs.get(i).label for i in langs.ids()
                   if i not in usable]
        QMessageBox.information(
            panel, "Toolchains",
            f"Usable now: {', '.join(usable)}\n\n"
            f"Not installed: {', '.join(missing)}\n\n"
            f"Each missing one disables that language, not the tool.")

    doctor.triggered.connect(show_doctor)
    return [doctor]
