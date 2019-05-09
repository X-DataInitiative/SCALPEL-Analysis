from src.exploration.core.decorators import save_plots
from src.exploration.flattening.flat_table_metadata import FlatTableMetadata
from src.exploration.stats.flattening_stat import registry


def plot_and_save_flattening_stat(
    json, pdf_path, figsize=(8, 5), id_col="NUM_ENQ", date_col="EXE_SOI_DTD", years=None
):
    """
    This method is used to visualize flattening stat and save the result in a pdf
    :param json: flat table meta data
    :param pdf_path: the path of pdf to save the stat
    :param figsize: size of the figure default = (8,5)
    :param id_col: 'str' identity column default = 'NUM_ENQ'
    :param date_col: 'str' data column used for 'group by' statement
    default = 'EXE_SOI_DTD'
    :param years: a list of special years in which the data will be loaded,
    default is None
    :return:
    """
    assert isinstance(json, str), "expected a string in the json format"
    metadata = FlatTableMetadata.from_json(json)
    save_plots(
        registry,
        pdf_path,
        list(metadata.flat_tables.values()),
        figsize=figsize,
        id_col=id_col,
        date_col=date_col,
        years=years,
    )
