# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import warnings

from extrap.gui.plots.BaseGraphWidget import BaseContourGraph


#####################################################################


class IsolinesDisplay(BaseContourGraph):
    def __init__(self, graphWidget, main_widget, width=5, height=4, dpi=100):
        super().__init__(graphWidget, main_widget, width, height, dpi)

    def draw_figure(self):
        """ 
          This function draws the graph
        """

        # Get data
        model_list, selected_callpaths = self._get_models_to_draw()
        if model_list is None:
            return

        # Get max x and max y value as a initial default value or a value provided by user
        maxX, maxY = self.get_max()

        X, Y, Z_List, z_List = self.calculate_z_models(maxX, maxY, model_list)

        # Set the x_label and y_label based on parameter selected.
        x_label = self.main_widget.data_display.getAxisParameter(0).name
        if x_label.startswith("_"):
            x_label = x_label[1:]
        y_label = self.main_widget.data_display.getAxisParameter(1).name
        if y_label.startswith("_"):
            y_label = y_label[1:]

        # Get the callpath color map
        widget = self.main_widget
        dict_callpath_color = widget.model_color_map
        number_of_subplots = 1
        if len(Z_List) > 1:
            number_of_subplots = len(Z_List) + 1

            # Adjusting subplots in order to avoid overlapping of labels
            # Reference : https://stackoverflow.com/questions/2418125/matplotlib-subplots-adjust-hspace-so-titles-and-xlabels-dont-overlap
            # left = 0.1
            # right = 0.9
            # bottom = 0.2
            # top = 0.9
            # wspace = 0.5
            # hspace = 0.2
            # self.fig.subplots_adjust(
            #     left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

            # Set the axis details for the subplot where we will draw all isolines
            ax_all = self.fig.add_subplot(
                1, number_of_subplots, number_of_subplots)
            ax_all.xaxis.major.formatter._useMathText = True
            ax_all.yaxis.major.formatter._useMathText = True
            ax_all.set_xlabel('\n' + x_label)
            ax_all.set_ylabel('\n' + y_label)
            ax_all.set_title(r'All')

        # Draw isolines
        for i in range(len(Z_List)):
            ax = self.fig.add_subplot(1, number_of_subplots, i + 1)
            ax.xaxis.major.formatter._useMathText = True
            ax.yaxis.major.formatter._useMathText = True
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 'No contour levels were found within the data range.')
                    cs = ax.contour(X, Y, Z_List[i], colors=dict_callpath_color[selected_callpaths[i]])
                    ax.clabel(cs, cs.levels[::2], inline=True,
                              fontsize=self.main_widget.plot_formatting_options.font_size * 0.8)
            except ValueError:  # raised if function selected is constant
                pass
            ax.set_xlabel('\n' + x_label)
            ax.set_ylabel('\n' + y_label)

            # ax.set_title ('function'+ str(i+1))
            ax.set_title(selected_callpaths[i].name,
                         fontdict={'fontsize': self.main_widget.plot_formatting_options.legend_font_size})

            if len(Z_List) > 1:
                try:
                    cs_all = ax_all.contour(
                        X, Y, Z_List[i], colors=dict_callpath_color[selected_callpaths[i]])
                    ax_all.clabel(
                        cs_all, cs_all.levels[::2], inline=True,
                        fontsize=self.main_widget.plot_formatting_options.font_size * 0.8)
                except ValueError:  # raised if function selected is constant
                    pass
            # self.fig.subplots_adjust(
            #     left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
            # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
            #     item.set_fontsize(10)
            # self.fig.colorbar(CS, ax=ax)
            # cax = ax.imshow(Z_List[i], interpolation='nearest', cmap=cm.coolwarm)
            # self.fig.colorbar(cax)

        if len(Z_List) > 1:
            # draw legend
            self.draw_legend(ax_all, dict_callpath_color)
