from extract_pitch import urmp_evaluate_all_except_one_instrument, urmp_evaluate_per_instrument, bach10_evaluate_all
import pandas as pd

results = {
    "non_violin_bach": pd.DataFrame(bach10_evaluate_all(instrument_to_discard='violin', pitch_range=(190,4000))).T,
    "violin_bach": pd.DataFrame(bach10_evaluate_all(instrument='violin')).T,
    "non_violin_urmp": pd.DataFrame(urmp_evaluate_all_except_one_instrument(instrument_to_discard='vn', pitch_range=(190,4000))).T,
    "violin_urmp": pd.DataFrame(urmp_evaluate_per_instrument(instrument='vn')).T
}

for name, table in results.items():
    table[['rpa50','rpa25','rpa10','rpa5']].round(4).to_latex('table_' + name + '.tex',
        position="h", label="table:" + name, caption=name
    )
    table.round(4).to_csv('table_' + name + '.csv')



