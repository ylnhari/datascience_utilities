def horizontal_bar_plot_grouped_for_two_columns_annotate_bars_after_scaling(df, main_column, num_columns1, num_column2):
    """Generate grouped Bar plots for two numerical columns for a categorical column.
    
    df = pd.DataFrame([['A', 50, 5], ['B', 99, 18], ['C', 78, 3]],
                  columns=['Main_Col', 'Num_Col1', 'NumCol2'])
    Plot
        on Y Axis -> Main Col
        on X Axis -> Top -> NumCol2
                  -> Bottom -> NumCol1
    """
    import numpy as np
    import pandas as pd
    import seaborn
    from matplotlib import pyplot as plt
    
    if len(df) > 30:  # restrict the plot have maximum of 30 bars
    df = df[:30]

    df_melted = pd.melt(df, id_vars='Main_Col', var_name='col_names', value_name='col_values')
    mask = df_melted.col_names.isin(['num_columns1'])
    if df_melted[~mask].col_values.mean() >= df_melted[mask].col_values.mean():  # pallets mean > loads mean
        smaller_col = 'Num_Col1'
        bigger_col = 'NumCol2'
    else:
        bigger_col = 'Num_Col1'
        smaller_col = 'NumCol2'

    mask = df_melted.col_names.isin([smaller_col])
    scale = df_melted[~mask].col_values.mean() // df_melted[mask].col_values.mean()  # scale Pallets
    df_melted.loc[mask, 'col_values'] = df_melted.loc[mask, 'col_values'] * scale  # scale loads

    fig, ax1 = plt.subplots(figsize=[7, len(df) * 2])
    g = seaborn.barplot(y='Main_Col', x='col_values', hue='col_names', data=df_melted, ax=ax1,
                        orient='h', hue_order=[smaller_col, bigger_col])
    g.set_axisbelow(True)
    g.xaxis.grid(color='gray', linestyle='dashed')
    g.set(title="Your Plot Title")

    # annotate bars
    bars = g.patches
    total_bars = len(bars)
    last_bar_indexes_of_scaled_column = (total_bars // 2) - 1
    for i in range(0, total_bars):
        p = bars[i]
        if i > last_bar_indexes_of_scaled_column:
            value = int(p.get_width())
        else:  # down-scale values of upscaled bar values for lower valued column
            value = int(p.get_width() / scale)

        x = p.get_x() + p.get_width() + 0.4
        y = p.get_y() + p.get_height()
        g.text(x, y, value, ha='left', va='bottom')

    # Create a second y-axis with the scaled ticks
    ax1.set_xlabel(bigger_col)
    ax2 = ax1.twiny()

    # Ensure ticks occur at the same positions, then modify labels
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticklabels(np.round(ax1.get_xticks() / scale, 2))
    ax2.set_xlabel(smaller_col)

    plt.show()
