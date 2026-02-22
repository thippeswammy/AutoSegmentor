"""
UITheme.py — Centralized dark theme and color scheme for the CVAT-like annotation UI.

Provides QSS stylesheets, annotation class colors, and font/spacing constants.
"""

from PyQt5.QtGui import QColor, QFont


# ─── Color Palette ──────────────────────────────────────────────────────────────

class Colors:
    """CVAT-inspired dark color palette."""

    # Backgrounds
    BG_DARKEST = "#1a1a2e"
    BG_DARK = "#1e1e2e"
    BG_MID = "#2b2b3c"
    BG_LIGHT = "#3b3b4f"
    BG_HOVER = "#454560"
    BG_SELECTED = "#4a4a6a"

    # Borders
    BORDER = "#3d3d5c"
    BORDER_LIGHT = "#55557a"

    # Text
    TEXT_PRIMARY = "#e0e0e0"
    TEXT_SECONDARY = "#a0a0b0"
    TEXT_MUTED = "#707080"
    TEXT_BRIGHT = "#ffffff"

    # Accents
    ACCENT_BLUE = "#4fc3f7"
    ACCENT_GREEN = "#66bb6a"
    ACCENT_ORANGE = "#ffa726"
    ACCENT_RED = "#ef5350"
    ACCENT_PURPLE = "#ab47bc"
    ACCENT_CYAN = "#26c6da"

    # Status
    SUCCESS = "#66bb6a"
    WARNING = "#ffa726"
    ERROR = "#ef5350"
    INFO = "#4fc3f7"

    # Canvas
    CANVAS_BG = "#2a2a3a"
    CROSSHAIR = "#ffffff80"
    GRID = "#ffffff15"
    ZOOM_BORDER = "#4fc3f7"

    # Toolbar
    TOOLBAR_BG = "#1e1e2e"
    TOOLBAR_BUTTON_HOVER = "#3b3b4f"
    TOOLBAR_BUTTON_ACTIVE = "#4a4a6a"


# ─── Annotation Class Colors ────────────────────────────────────────────────────

# BGR tuples for OpenCV rendering (kept for mask processing compatibility)
ANNOTATION_COLORS_BGR = {
    1: (0, 0, 255),      # Red
    2: (255, 0, 0),      # Blue
    3: (0, 255, 0),      # Green
    4: (0, 255, 255),    # Yellow
    5: (255, 0, 255),    # Magenta
    6: (255, 255, 0),    # Cyan
    7: (128, 0, 128),    # Purple
    8: (0, 165, 255),    # Orange
    9: (255, 255, 255),  # White
    10: (128, 128, 128), # Gray
}

# QColor objects for PyQt rendering (RGB)
ANNOTATION_COLORS_QT = {
    1: QColor(255, 80, 80),     # Red
    2: QColor(80, 140, 255),    # Blue
    3: QColor(80, 220, 80),     # Green
    4: QColor(255, 220, 50),    # Yellow
    5: QColor(220, 80, 220),    # Magenta
    6: QColor(50, 220, 220),    # Cyan
    7: QColor(160, 80, 200),    # Purple
    8: QColor(255, 165, 50),    # Orange
    9: QColor(220, 220, 220),   # White
    10: QColor(150, 150, 150),  # Gray
}

# Bright highlight variants for selected/hovered points
ANNOTATION_COLORS_BRIGHT = {
    k: QColor(min(c.red() + 40, 255), min(c.green() + 40, 255), min(c.blue() + 40, 255))
    for k, c in ANNOTATION_COLORS_QT.items()
}


# ─── Font Configuration ─────────────────────────────────────────────────────────

class Fonts:
    """Font presets."""

    @staticmethod
    def header():
        f = QFont("Segoe UI", 11)
        f.setBold(True)
        return f

    @staticmethod
    def body():
        return QFont("Segoe UI", 9)

    @staticmethod
    def small():
        return QFont("Segoe UI", 8)

    @staticmethod
    def mono():
        return QFont("Cascadia Code", 9)

    @staticmethod
    def mono_small():
        return QFont("Cascadia Code", 8)


# ─── Layout Constants ───────────────────────────────────────────────────────────

SIDEBAR_WIDTH = 240
TOOLBAR_HEIGHT = 38
STATUSBAR_HEIGHT = 26
ZOOM_VIEW_SIZE = 180
POINT_RADIUS = 5
POINT_RADIUS_SELECTED = 7
BADGE_FONT_SIZE = 8


# ─── QSS Stylesheet ─────────────────────────────────────────────────────────────

DARK_STYLESHEET = f"""
/* ── Global ── */
QWidget {{
    background-color: {Colors.BG_DARK};
    color: {Colors.TEXT_PRIMARY};
    font-family: "Segoe UI";
    font-size: 9pt;
}}

/* ── QMainWindow / QDialog ── */
QMainWindow, QDialog {{
    background-color: {Colors.BG_DARKEST};
}}

/* ── Menu Bar ── */
QMenuBar {{
    background-color: {Colors.BG_DARKEST};
    color: {Colors.TEXT_PRIMARY};
    border-bottom: 1px solid {Colors.BORDER};
    padding: 2px;
}}
QMenuBar::item {{
    padding: 4px 10px;
    border-radius: 3px;
}}
QMenuBar::item:selected {{
    background-color: {Colors.BG_HOVER};
}}
QMenu {{
    background-color: {Colors.BG_MID};
    border: 1px solid {Colors.BORDER};
    padding: 4px;
}}
QMenu::item {{
    padding: 5px 30px 5px 20px;
    border-radius: 3px;
}}
QMenu::item:selected {{
    background-color: {Colors.BG_HOVER};
}}
QMenu::separator {{
    height: 1px;
    background: {Colors.BORDER};
    margin: 4px 8px;
}}

/* ── Toolbar ── */
QToolBar {{
    background-color: {Colors.TOOLBAR_BG};
    border-bottom: 1px solid {Colors.BORDER};
    spacing: 4px;
    padding: 3px 6px;
}}
QToolBar::separator {{
    width: 1px;
    background: {Colors.BORDER};
    margin: 4px 6px;
}}
QToolButton {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px 8px;
    color: {Colors.TEXT_PRIMARY};
    font-size: 9pt;
}}
QToolButton:hover {{
    background-color: {Colors.TOOLBAR_BUTTON_HOVER};
    border-color: {Colors.BORDER_LIGHT};
}}
QToolButton:checked, QToolButton:pressed {{
    background-color: {Colors.TOOLBAR_BUTTON_ACTIVE};
    border-color: {Colors.ACCENT_BLUE};
}}

/* ── Status Bar ── */
QStatusBar {{
    background-color: {Colors.BG_DARKEST};
    color: {Colors.TEXT_SECONDARY};
    border-top: 1px solid {Colors.BORDER};
    font-size: 8pt;
}}
QStatusBar::item {{
    border: none;
}}

/* ── Dock Widget ── */
QDockWidget {{
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
    color: {Colors.TEXT_PRIMARY};
    font-weight: bold;
}}
QDockWidget::title {{
    background-color: {Colors.BG_MID};
    border: 1px solid {Colors.BORDER};
    padding: 6px;
    text-align: left;
}}

/* ── Scroll Area / ScrollBar ── */
QScrollArea {{
    border: none;
    background-color: {Colors.BG_DARK};
}}
QScrollBar:vertical {{
    background: {Colors.BG_DARK};
    width: 10px;
}}
QScrollBar::handle:vertical {{
    background: {Colors.BG_LIGHT};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {Colors.BG_HOVER};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {Colors.BG_DARK};
    height: 10px;
}}
QScrollBar::handle:horizontal {{
    background: {Colors.BG_LIGHT};
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {Colors.BG_HOVER};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── Push Button ── */
QPushButton {{
    background-color: {Colors.BG_LIGHT};
    color: {Colors.TEXT_PRIMARY};
    border: 1px solid {Colors.BORDER};
    border-radius: 4px;
    padding: 5px 14px;
    font-size: 9pt;
}}
QPushButton:hover {{
    background-color: {Colors.BG_HOVER};
    border-color: {Colors.BORDER_LIGHT};
}}
QPushButton:pressed {{
    background-color: {Colors.BG_SELECTED};
}}
QPushButton#acceptButton {{
    background-color: #2d5a3d;
    border-color: {Colors.SUCCESS};
    color: {Colors.SUCCESS};
    font-weight: bold;
}}
QPushButton#acceptButton:hover {{
    background-color: #3a6d4f;
}}
QPushButton#resetButton {{
    background-color: #5a2d2d;
    border-color: {Colors.ERROR};
    color: {Colors.ERROR};
}}
QPushButton#resetButton:hover {{
    background-color: #6d3a3a;
}}

/* ── Combo Box ── */
QComboBox {{
    background-color: {Colors.BG_LIGHT};
    border: 1px solid {Colors.BORDER};
    border-radius: 4px;
    padding: 3px 8px;
    color: {Colors.TEXT_PRIMARY};
    min-width: 60px;
}}
QComboBox:hover {{
    border-color: {Colors.BORDER_LIGHT};
}}
QComboBox::drop-down {{
    border: none;
    padding-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {Colors.BG_MID};
    border: 1px solid {Colors.BORDER};
    selection-background-color: {Colors.BG_HOVER};
}}

/* ── Spin Box ── */
QSpinBox {{
    background-color: {Colors.BG_LIGHT};
    border: 1px solid {Colors.BORDER};
    border-radius: 4px;
    padding: 2px 6px;
    color: {Colors.TEXT_PRIMARY};
}}

/* ── Label ── */
QLabel {{
    color: {Colors.TEXT_PRIMARY};
    background: transparent;
}}
QLabel#sectionHeader {{
    color: {Colors.ACCENT_BLUE};
    font-weight: bold;
    font-size: 9pt;
    padding: 4px 0px;
}}
QLabel#statusCoord {{
    color: {Colors.ACCENT_CYAN};
    font-family: "Cascadia Code";
    font-size: 8pt;
}}
QLabel#statusInfo {{
    color: {Colors.TEXT_SECONDARY};
    font-size: 8pt;
}}

/* ── List Widget ── */
QListWidget {{
    background-color: {Colors.BG_DARK};
    border: 1px solid {Colors.BORDER};
    border-radius: 4px;
    outline: none;
}}
QListWidget::item {{
    padding: 4px 8px;
    border-radius: 3px;
}}
QListWidget::item:selected {{
    background-color: {Colors.BG_SELECTED};
}}
QListWidget::item:hover {{
    background-color: {Colors.BG_HOVER};
}}

/* ── Group Box ── */
QGroupBox {{
    border: 1px solid {Colors.BORDER};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 14px;
    font-weight: bold;
    color: {Colors.ACCENT_BLUE};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    left: 10px;
}}

/* ── Progress Bar ── */
QProgressBar {{
    background-color: {Colors.BG_DARK};
    border: 1px solid {Colors.BORDER};
    border-radius: 4px;
    text-align: center;
    color: {Colors.TEXT_PRIMARY};
    font-size: 8pt;
    height: 16px;
}}
QProgressBar::chunk {{
    background-color: {Colors.ACCENT_BLUE};
    border-radius: 3px;
}}

/* ── Splitter ── */
QSplitter::handle {{
    background-color: {Colors.BORDER};
    width: 2px;
}}
QSplitter::handle:hover {{
    background-color: {Colors.ACCENT_BLUE};
}}

/* ── Tooltip ── */
QToolTip {{
    background-color: {Colors.BG_MID};
    border: 1px solid {Colors.BORDER};
    color: {Colors.TEXT_PRIMARY};
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 8pt;
}}
"""
