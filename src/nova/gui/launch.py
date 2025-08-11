#!/usr/bin/env python3
"""
NOVA GUI Launcher

Entry points for graphical user interfaces.
"""

import sys

import click

try:
    from PyQt5.QtWidgets import QApplication  # noqa: F401
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """Launch NOVA graphical interfaces."""
    if ctx.invoked_subcommand is None:
        # Default to main GUI
        launch_main_gui()


@main.command()
def main_gui():
    """Launch the main GUI interface."""
    launch_main_gui()


@main.command()
def compare():
    """Launch the ansatz comparison GUI."""
    launch_compare_gui()


@main.command()
def launcher():
    """Launch the GUI launcher/selector."""
    launch_launcher_gui()


def launch_main_gui():
    """Launch the main GUI interface."""
    if not PYQT_AVAILABLE:
        click.echo("Error: PyQt5 not available. Install with: pip install PyQt5", err=True)
        sys.exit(1)

    try:
        from nova.gui.gui_interface import main as gui_main
        gui_main()
    except Exception as e:
        click.echo(f"Error launching main GUI: {e}", err=True)
        sys.exit(1)


def launch_compare_gui():
    """Launch the ansatz comparison GUI."""
    if not PYQT_AVAILABLE:
        click.echo("Error: PyQt5 not available. Install with: pip install PyQt5", err=True)
        sys.exit(1)

    try:
        from nova.gui.ansatz_comparison_gui import main as compare_main
        compare_main()
    except Exception as e:
        click.echo(f"Error launching comparison GUI: {e}", err=True)
        sys.exit(1)


def launch_launcher_gui():
    """Launch the GUI launcher/selector."""
    if not PYQT_AVAILABLE:
        click.echo("Error: PyQt5 not available. Install with: pip install PyQt5", err=True)
        sys.exit(1)

    try:
        # Import the launcher from the CLI module (converted to PyQt launcher)
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QFont
        from PyQt5.QtWidgets import (
            QApplication,
            QButtonGroup,
            QGroupBox,
            QLabel,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QRadioButton,
            QVBoxLayout,
            QWidget,
        )

        class NOVALauncher(QMainWindow):
            """NOVA GUI Launcher."""

            def __init__(self):
                super().__init__()
                self.setWindowTitle("NOVA Launcher")
                self.setGeometry(100, 100, 800, 600)

                # Central widget
                central_widget = QWidget()
                self.setCentralWidget(central_widget)
                layout = QVBoxLayout(central_widget)

                # Header
                title = QLabel("NOVA Quantum Neural Network")
                title.setAlignment(Qt.AlignCenter)
                font = QFont()
                font.setPointSize(18)
                font.setBold(True)
                title.setFont(font)
                layout.addWidget(title)

                subtitle = QLabel("Select a component to launch")
                subtitle.setAlignment(Qt.AlignCenter)
                layout.addWidget(subtitle)

                # Options
                options_group = QGroupBox("Available Components")
                options_layout = QVBoxLayout(options_group)

                self.button_group = QButtonGroup()

                # Main GUI
                main_radio = QRadioButton("Main GUI Interface")
                main_radio.setChecked(True)
                self.button_group.addButton(main_radio, 0)
                options_layout.addWidget(main_radio)

                # Comparison GUI
                compare_radio = QRadioButton("Ansatz Comparison Tool")
                self.button_group.addButton(compare_radio, 1)
                options_layout.addWidget(compare_radio)

                # CLI
                cli_radio = QRadioButton("Command-Line Interface")
                self.button_group.addButton(cli_radio, 2)
                options_layout.addWidget(cli_radio)

                layout.addWidget(options_group)

                # Launch button
                launch_btn = QPushButton("Launch")
                launch_btn.setMinimumHeight(50)
                font = QFont()
                font.setPointSize(12)
                font.setBold(True)
                launch_btn.setFont(font)
                launch_btn.clicked.connect(self.launch_selected)
                layout.addWidget(launch_btn)

            def launch_selected(self):
                """Launch the selected component."""
                selected = self.button_group.checkedId()

                try:
                    if selected == 0:  # Main GUI
                        launch_main_gui()
                    elif selected == 1:  # Compare GUI
                        launch_compare_gui()
                    elif selected == 2:  # CLI
                        import subprocess
                        # Using a fixed argument list without shell to avoid shell injection concerns
                        subprocess.Popen([sys.executable, "-m", "nova.cli"], shell=False)  # noqa: S603

                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to launch: {e}")

        app = QApplication(sys.argv)
        window = NOVALauncher()
        window.show()
        sys.exit(app.exec_())

    except Exception as e:
        click.echo(f"Error launching GUI launcher: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
