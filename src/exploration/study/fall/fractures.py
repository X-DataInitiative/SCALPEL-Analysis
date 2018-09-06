from src.exploration.core.cohort import Cohort
from src.exploration.core.decorators import logged, xlabel, ylabel, title, \
    ylabel_fontsize
from matplotlib.figure import Figure
import logging
import seaborn as sns


@logged(logging.INFO, "X-CNAM", "Saving plot for fractures by site")
@xlabel("Nombre de fractures")
@ylabel("Site")
@title("Distribution de nombre de fractures par site")
@ylabel_fontsize(8)
def plot_fractures_by_site_distribution(figure: Figure, cohort: Cohort) -> Figure:
    axe = figure.gca()
    df = cohort.events
    fractures_site = df.groupBy("groupID").count().toPandas().sort_values("count", ascending=True)
    axe.barh(y=range(len(fractures_site)), width=fractures_site["count"].values,
             tick_label=fractures_site.groupID.values,
             color=sns.xkcd_rgb["pumpkin orange"], )
    axe.grid(True, which="major", axis="x")
    return figure


registry = [plot_fractures_by_site_distribution, ]