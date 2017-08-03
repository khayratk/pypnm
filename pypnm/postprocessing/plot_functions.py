import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator


def format_axes(ax, size=20, tick_space=0.2):
    """
    Function to format a matplotlib ax for publications
    """
    f = size/20.

    for tick in ax.get_xaxis().get_major_ticks():
        tick.set_pad(15.*f)
        tick.label1 = tick._get_text1()

    for tick in ax.get_yaxis().get_major_ticks():
        tick.set_pad(15.*f)
        tick.label1 = tick._get_text1()

    minorLocator = AutoMinorLocator()
    majorLocator = MultipleLocator(tick_space)

    ax.xaxis.set_minor_locator(minorLocator)
    ax.xaxis.set_major_locator(majorLocator)

    minorLocator = AutoMinorLocator()
    ax.yaxis.set_minor_locator(minorLocator)

    ax.tick_params(axis='both', which='major', labelsize=20*f, width=2*f, length=10*f)
    ax.tick_params(axis='both', which='minor', labelsize=20*f, width=2*f, length=5*f)

    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    ax.yaxis.label.set_size(30*f)
    ax.xaxis.label.set_size(30*f)

    ax.ticklabel_format(style='sci', axis='y')
    ax.yaxis.major.formatter.set_powerlimits((-2, 2))

    ax.legend(loc=0, fontsize=30*f)


def plot_on_axis(ax, color, *xy_data_symbol):
    if color is None:
        for x_data, y_data, y_symbol in xy_data_symbol:
            ax.plot(x_data, y_data, y_symbol, linewidth=2.0, mfc='none', mew=2)
    else:
        for x_data, y_data, y_symbol in xy_data_symbol:
            ax.plot(x_data, y_data, y_symbol, color=color, linewidth=2.0, mfc='none', mew=2)


def set_plot_labels(ax, x_label, y_label):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def plot(x_label, y_label, x_lim, y_lim,  *xy_data_symbol, **params):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if 'color' in params:
        plot_on_axis(ax, params['color'], *xy_data_symbol)
    else:
        plot_on_axis(ax, None, *xy_data_symbol)

    set_plot_labels(ax, x_label, y_label)
    ax.set_ylim(y_lim)
    ax.set_xlim(x_lim)

    if 'size' in params:
        format_axes(ax, size=params['size'], tick_space=params['tick_space'])
    else:
        format_axes(ax)

    fig.tight_layout()
    return fig


def plot_histogram(data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for x in data:
        n, bins, patches = ax.hist(x, bins=100, normed=1)
    return fig


def plot_save_histogram(plot_name, data):
    fig = plot_histogram(data)
    fig.savefig(plot_name + '.svg')


def plot_save(plot_name, x_label, y_label, x_lim, y_lim,  *xy_data_symbol, **params):
    fig = plot(x_label, y_label, x_lim, y_lim, *xy_data_symbol, **params)
    fig.savefig(plot_name + '.svg')