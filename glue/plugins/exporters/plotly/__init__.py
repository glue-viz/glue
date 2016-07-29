def setup():
    from glue.config import exporters
    from glue.plugins.exporters.plotly.export_plotly import save_plotly, can_save_plotly
    exporters.add('Plotly', save_plotly, can_save_plotly)
