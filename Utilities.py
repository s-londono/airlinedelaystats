import numpy as np
import matplotlib.pyplot as plt


def bi_bar_plot(labels, x1, x2, label1, label2, ylabel, title, bar_width=0.4, figsize=(18, 6)):
    """
    Creates and shows a double bar plot (two bar plots combined in one figure). The bar plots
    depict two different dependent variable in terms of a single independent variable. Based on the example at:
    https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    INPUT
        labels - categories to be plotted in the x-axis
        x1 - first set of bar heights for each category
        x2 - second set of bar heights for each category
        label1 - name of the variable depicted by the first set of bars
        label2 - name of the variable depicted by the second set of bars
        ylabel - legend for the y-axis
        title - title of the plot
        bar_width - width of each bar
        figzie - dimensions of the plot (in inches)
    OUTPUT
        none, shows the plot automatically
    """
    x = np.arange(len(labels))  # the label locations

    fig = plt.figure(figsize=figsize)
    ax = fig.subplots()

    rects1 = ax.bar(x - bar_width/2, x1, bar_width, label=label1)
    rects2 = ax.bar(x + bar_width/2, x2, bar_width, label=label2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Attach a text label above each bar in *rects*, displaying its height
    for rects in [rects1, rects2]:
        for rect in rects:
            ax.annotate('{}'.format(rect.get_height()),
                        xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    fig.tight_layout()

    plt.show()

