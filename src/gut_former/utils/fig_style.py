import matplotlib as mpl
import seaborn as sns


def apply_style(
    font_family: str = "DejaVu Sans",
    base_font_size: int = 12,
    style = "whitegrid",
) -> None:
    rc = {
        "font.family": font_family,
        "font.size": base_font_size,
        "axes.titlesize": base_font_size,
        "axes.labelsize": base_font_size,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": base_font_size,
        "figure.titlesize": base_font_size,
        "axes.linewidth": 1.5,
    }

    # Matplotlib global
    mpl.rcParams.update(rc)

    # Seaborn global
    sns.set_theme(style=style, font=font_family, rc=rc)