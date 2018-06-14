import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
# These are the "Tableau 20" colors as RGB.  
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
            (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
            (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
            (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
            (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
for i in range(len(tableau20)):  
    r, g, b = tableau20[i]  
    tableau20[i] = (r / 255., g / 255., b / 255.)

# This function takes an array of numbers and smoothes them out.  
# Smoothing is useful for making plots a little easier to read.  
def sliding_mean(data_array, window=5):  
    data_array = np.array(data_array)
    new_list = []  
    for i in range(len(data_array)):  
        indices = range(max(i - window + 1, 0),  
                        min(i + window + 1, len(data_array)))  
        avg = 0  
        for j in indices:  
            avg += data_array[j]
        avg /= float(len(indices))  
        new_list.append(avg)  
          
    return np.array(new_list)

def csvplot(title, file_list, xr, yr, ytick, ypos):
    # You typically want your plot to be ~1.33x wider than tall. This plot is a rare  
    # exception because of the number of lines being plotted on it.  
    # Common sizes: (10, 7.5) and (12, 9)  
    plt.figure(figsize=(11, 9))  

    # Remove the plot frame lines. They are unnecessary chartjunk.  
    ax = plt.subplot(111)  
    ax.spines["top"].set_visible(False)  
    ax.spines["bottom"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.spines["left"].set_visible(False)  

    # Ensure that the axis ticks only show up on the bottom and left of the plot.  
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  

    # Limit the range of the plot to only where the data is.  
    # Avoid unnecessary whitespace.
    plt.xlim(xr[0], xr[1])
    plt.ylim(yr[0], yr[1])

    # Make sure your axis ticks are large enough to be easily read.  
    # You don't want your viewers squinting to read your plot.

    ax.set_yscale('log')
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.yticks(ytick, ['%.1e' % x for x in ytick], fontsize=14)
    plt.xticks(range(0, 300001, 50000), ['0', '50k', '100k', '150k', '200k', '250k', '300k'], fontsize=14)

    # # Provide tick lines across the plot to help your viewers trace along  
    # # the axis ticks. Make sure that the lines are light and small so they  
    # # don't obscure the primary data lines.  
    # for y in range(10, 91, 10):  
    #     plt.plot(range(1968, 2012), [y] * len(range(1968, 2012)), "--", lw=0.5, color="black", alpha=0.3)  
    for y in ytick[:-1]:
        plt.plot(range(0,300001,50000), [y] * len(range(0,300001,50000)), 'k:', lw=0.5, alpha=0.3)

    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.  
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                    labelbottom="on", left="on", right="off", labelleft="on")  

    
    for i, file_path in enumerate(file_list):
        # Read the data into a pandas DataFrame.  
        data = pd.read_csv(file_path)
        # plt.plot(data.Step.values, data.Value.values, lw=2.5, color=tableau20[i])

        # plt.plot(data.Step.values, data.Value.values, lw=2.5, color=tableau20[i*2])
        # mean = data.Value.values
        cid = i*2 % 8

        window_size = 50 # 0.01 10 over 1000
        mean = sliding_mean(data.Value.values, window=window_size)
        err = np.abs(data.Value.values - mean)*1.6
        err = sliding_mean(err, window=10)
        err_c = list(tableau20[cid+1]) + [0.3]
        # upper = mean+err
        # lower = mean-err
        # plt.plot(data.Step.values, upper, lw=0.5, color=err_c)
        # plt.plot(data.Step.values, lower, lw=0.5, color=err_c)
        # plt.plot(data.Step.values, data.Value.values, lw=0.5, color=err_c)
        plt.fill_between(data.Step.values, mean-err, mean+err, color=err_c)
        plt.plot(data.Step.values, mean, lw=2.5, color=tableau20[cid])

        
        # Add a text label to the right end of every line. Most of the code below  
        # is adding specific offsets y position because some labels overlapped.  
        y_pos = mean[-1] + ypos[i]
        
        # Again, make sure that all labels are large enough to be easily read  
        # by the viewer.  
        plt.text(305000, y_pos, label[i], fontsize=12, color=tableau20[cid], ha="left")

        # matplotlib's title() call centers the title on the plot, but not the graph,  
        # so I used the text() call to customize where the title goes.  
        # Make the title big enough so it spans the entire plot, but don't make it  
        # so big that it requires two lines to show.  

        # Note that if the title is descriptive enough, it is unnecessary to include  
        # axis labels; they are self-evident, in this plot's case.  
        # plt.text(5000, yr[1]+5e-2, "Figure Title", fontsize=17, ha="left")

        # Always include your data source(s) and copyright notice! And for your  
        # data sources, tell your viewers exactly where the data came from,  
        # preferably with a direct link to the data. Just telling your viewers  
        # that you used data from the "U.S. Census Bureau" is completely useless:  
        # the U.S. Census Bureau provides all kinds of data, so how are your  
        # viewers supposed to know which data set you used?  
        # plt.text(5000, yr[0], "Footnote", fontsize=10)

        # Finally, save the figure as a PNG.  
        # You can also save it as a PDF, JPEG, etc.  
        # Just change the file extension in this call.  
        # bbox_inches="tight" removes all the extra whitespace on the edges of your plot.  
        # plt.savefig("csvplot.png", bbox_inches="tight")
        plt.savefig('log/'+title+'.png', bbox_inches="tight") # pdf: font-embedding problem
    plt.show()
    
def err_plot(title, file_list, xr, yr, ytick, ypos):
    # You typically want your plot to be ~1.33x wider than tall. This plot is a rare  
    # exception because of the number of lines being plotted on it.  
    # Common sizes: (10, 7.5) and (12, 9)  
    plt.figure(figsize=(11, 3))  

    # Remove the plot frame lines. They are unnecessary chartjunk.  
    ax = plt.subplot(111)  
    ax.spines["top"].set_visible(False)  
    ax.spines["bottom"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.spines["left"].set_visible(False)  

    # Ensure that the axis ticks only show up on the bottom and left of the plot.  
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  

    # Limit the range of the plot to only where the data is.  
    # Avoid unnecessary whitespace.
    plt.xlim(xr[0], xr[1])
    plt.ylim(yr[0], yr[1])

    # Make sure your axis ticks are large enough to be easily read.  
    # You don't want your viewers squinting to read your plot.

    xtick = [0, 50, 100, 150]
    # ax.set_yscale('log')
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.yticks(ytick, ytick, fontsize=14)
    plt.xticks(xtick, ['0', '50', '100', '150'], fontsize=14)

    # # Provide tick lines across the plot to help your viewers trace along  
    # # the axis ticks. Make sure that the lines are light and small so they  
    # # don't obscure the primary data lines.  
    # for y in range(10, 91, 10):  
    #     plt.plot(range(1968, 2012), [y] * len(range(1968, 2012)), "--", lw=0.5, color="black", alpha=0.3)  
    for y in ytick:
        plt.plot(xtick, [y] * len(xtick), 'k:', lw=0.5, alpha=0.3)

    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.  
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                    labelbottom="on", left="on", right="off", labelleft="on")  

    
    for i, file_path in enumerate(file_list):
        print(file_path)
        # Read the data into a pandas DataFrame.  
        data = pd.read_csv(file_path)
        plt.plot(range(xr[1]), data.values, lw=2.5, color=tableau20[i*2])
        # plt.plot(data.Step.values, data.Value.values, lw=2.5, color=tableau20[i])

        # plt.plot(data.Step.values, data.Value.values, lw=2.5, color=tableau20[i*2])
        # mean = data.Value.values
        # cid = i*2 % 8

        # window_size = 50 # 0.01 10 over 1000
        # mean = sliding_mean(data.Value.values, window=window_size)
        # err = np.abs(data.Value.values - mean)*1.6
        # err = sliding_mean(err, window=10)
        # err_c = list(tableau20[cid+1]) + [0.3]
        # # upper = mean+err
        # # lower = mean-err
        # # plt.plot(data.Step.values, upper, lw=0.5, color=err_c)
        # # plt.plot(data.Step.values, lower, lw=0.5, color=err_c)
        # # plt.plot(data.Step.values, data.Value.values, lw=0.5, color=err_c)
        # plt.fill_between(data.Step.values, mean-err, mean+err, color=err_c)
        # plt.plot(data.Step.values, mean, lw=2.5, color=tableau20[cid])

        
        # # Add a text label to the right end of every line. Most of the code below  
        # # is adding specific offsets y position because some labels overlapped.  
        y_pos = ypos[i]
        
        # Again, make sure that all labels are large enough to be easily read  
        # by the viewer.  
        plt.text(120, y_pos, label[i], fontsize=12, color=tableau20[i*2], ha="left")

        # matplotlib's title() call centers the title on the plot, but not the graph,  
        # so I used the text() call to customize where the title goes.  
        # Make the title big enough so it spans the entire plot, but don't make it  
        # so big that it requires two lines to show.  

        # Note that if the title is descriptive enough, it is unnecessary to include  
        # axis labels; they are self-evident, in this plot's case.  
        # plt.text(5000, yr[1]+5e-2, "Figure Title", fontsize=17, ha="left")

        # Always include your data source(s) and copyright notice! And for your  
        # data sources, tell your viewers exactly where the data came from,  
        # preferably with a direct link to the data. Just telling your viewers  
        # that you used data from the "U.S. Census Bureau" is completely useless:  
        # the U.S. Census Bureau provides all kinds of data, so how are your  
        # viewers supposed to know which data set you used?  
        # plt.text(5000, yr[0], "Footnote", fontsize=10)

        # Finally, save the figure as a PNG.  
        # You can also save it as a PDF, JPEG, etc.  
        # Just change the file extension in this call.  
        # bbox_inches="tight" removes all the extra whitespace on the edges of your plot.  
        # plt.savefig("csvplot.png", bbox_inches="tight")
        plt.savefig('log/'+title+'.png', bbox_inches="tight") # pdf: font-embedding problem
    plt.show()

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    xr = [0,149]
    title = 'obs_err'
    file_list = [
        'log/de/velocity/smoke3_obs11_buo4_f150/final/eval/p2_n150/4_2_mae.csv',
        'log/de/velocity/smoke3_obs11_buo4_f150/final/eval/p2_n150/4.5_2_mae.csv',
        'log/de/velocity/smoke3_obs11_buo4_f150/final/eval/p2_n150/5_2_mae.csv',
        'log/de/velocity/smoke3_obs11_buo4_f150/final/eval/p2_n150/4.5_2_linear_mae.csv',
    ]
    yr = [0, 0.4]
    ytick = [0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4]
    ypos = [
        0.0, # 0.02
        0.12,
        0.06,
        0.35,
        ]
    label = [
        r'CNN $p_x$=0.44',
        r'CNN $\hat{p}_x$=0.47',
        r'CNN $p_x$=0.50',
        r'Linear Blend $\hat{p}_x$=0.47',
        ]
    err_plot(title, file_list, xr, yr, ytick, ypos)

    # xr = [0,330000]

    # title = 'lossPlot'
    # file_list = [
    #     'log/plot/run_smoke_pos21_size5_f200_long-tag-loss_g_loss_l1.csv',
    #     # 'log/plot/run_smoke_pos21_size5_f200_0119_213138_11-tag-loss_g_loss_msssim.csv',
    # ]
    # yr = [1e-3, 0.05]
    # ytick = [1e-3, 0.01, 0.02, 0.035, 0.05]
    # ypos = [
    #     0,
    #     #0,
    #     ]
    # label = [
    #     r'$L_1$ Loss',
    #     #r'MS-SSIM Loss',
    #     ]
    # csvplot(title, file_list, xr, yr, ytick, ypos)
    
    # xr = [0,370000]
    # for s in range(2,3):
    #     # if s == 0:
    #     #     title = 'div'
    #     #     file_list = [
    #     #         'log/plot/run_smoke_pos10_f200_0108_132248_110.01Joff-tag-loss_g_loss_div.csv',
    #     #         'log/plot/run_smoke_pos10_f200_0108_132248_10-tag-loss_g_loss_div.csv',
    #     #         'log/plot/run_smoke_pos10_f200_0108_132248_11-tag-loss_g_loss_div.csv',
    #     #         'log/plot/run_smoke_pos10_f200_0108_132248_110.01Jon-tag-loss_g_loss_div.csv',
    #     #     ]
    #     #     yr = [1e-4, 1e-1]
    #     #     ytick = [1e-4, 1e-3, 1e-2, 1e-1]
    #     #     ypos = [0.003, -0.003, -0.0025, 0.00008]
    #     #     label = ['+GAN w/ Velocity', 'Generator wo/ Jacobian Loss', 'Generator w/ Jacobian Loss', '+GAN w/ Jacobian']
    #     # elif s == 1:
    #     #     title = 'div_max'
    #     #     file_list = [
    #     #         'log/plot/run_smoke_pos10_f200_0108_132248_110.01Joff-tag-loss_g_loss_div_max.csv',
    #     #         'log/plot/run_smoke_pos10_f200_0108_132248_10-tag-loss_g_loss_div_max.csv',
    #     #         'log/plot/run_smoke_pos10_f200_0108_132248_11-tag-loss_g_loss_div_max.csv',
    #     #         'log/plot/run_smoke_pos10_f200_0108_132248_110.01Jon-tag-loss_g_loss_div_max.csv',
    #     #     ]
    #     #     yr = [1e-3, 1e0]
    #     #     ytick = [1e-3, 1e-2, 1e-1, 1e0]
    #     #     ypos = [0.03, -0.03, -0.01, 0.0003]
    #     #     label = ['+GAN w/ Velocity', 'Generator wo/ Jacobian Loss', 'Generator w/ Jacobian Loss', '+GAN w/ Jacobian']
    #     # elif s == 2:
    #     #     title = 'div_z'
    #     #     file_list = [
    #     #         'log/plot/run_smoke_pos10_f200_0108_132248_110.01Joff-tag-loss_g_loss_z_div.csv',
    #     #         'log/plot/run_smoke_pos10_f200_0108_132248_110.01Jon-tag-loss_g_loss_z_div.csv',
    #     #     ]
    #     #     yr = [1e-4, 1e-1]
    #     #     ytick = [1e-4, 1e-3, 1e-2, 1e-1]
    #     #     ypos = [0.003, 0.0003]
    #     #     label = ['+GAN w/ Velocity', '+GAN w/ Jacobian']
    #     # elif s == 3:
    #     #     title = 'div_z_max'
    #     #     file_list = [
    #     #         'log/plot/run_smoke_pos10_f200_0108_132248_110.01Joff-tag-loss_g_loss_z_div_max.csv',
    #     #         'log/plot/run_smoke_pos10_f200_0108_132248_110.01Jon-tag-loss_g_loss_z_div_max.csv',
    #     #     ]
    #     #     yr = [1e-3, 1e0]
    #     #     ytick = [1e-3, 1e-2, 1e-1, 1e0]
    #     #     ypos = [0.05, 0.009]
    #     #     label = ['+GAN w/ Velocity', '+GAN w/ Jacobian']

    #     if s == 0:
    #         title = 'divPlot'
    #         file_list = [
    #             'log/plot/run_smoke_pos10_f200_0110_171030_10-tag-loss_g_loss_div_max.csv',
    #             'log/plot/run_smoke_pos10_f200_0110_171030_11-tag-loss_g_loss_div_max.csv',
    #             'log/plot/run_smoke_pos10_f200_0110_171030_10-tag-loss_g_loss_div.csv',
    #             'log/plot/run_smoke_pos10_f200_0110_171030_11-tag-loss_g_loss_div.csv',
    #         ]
    #         yr = [1e-3, 1e0]
    #         ytick = [1e-3, 1e-2, 1e-1, 1e0]
    #         ypos = [
    #             -0.015, -0.001, 
    #             -0.001, 0,
    #             ]
    #         label = [
    #             r'w/ $\lambda_{\nabla \mathbf{u}}=0$ (max)',
    #             r'w/ $\lambda_{\nabla \mathbf{u}}=1$ (max)',
    #             r'w/ $\lambda_{\nabla \mathbf{u}}=0$',
    #             r'w/ $\lambda_{\nabla \mathbf{u}}=1$',
    #             ]

    #     elif s == 1:
    #         title = 'divPlot_z'
    #         file_list = [
    #             'log/plot/run_smoke_pos10_f200_0110_171030_10-tag-loss_g_loss_z_div_max.csv',
    #             'log/plot/run_smoke_pos10_f200_0110_171030_11-tag-loss_g_loss_z_div_max.csv',
    #             'log/plot/run_smoke_pos10_f200_0110_171030_10-tag-loss_g_loss_z_div.csv',
    #             'log/plot/run_smoke_pos10_f200_0110_171030_11-tag-loss_g_loss_z_div.csv',
    #         ]
    #         yr = [1e-3, 1e0]
    #         ytick = [1e-3, 1e-2, 1e-1, 1e0]
    #         ypos = [
    #             -0.01, -0.001,
    #             -0.001, -0.0005,
    #             ]
    #         label = [
    #             r'w/ $\lambda_{\nabla \mathbf{u}}=0$ (max)',
    #             r'w/ $\lambda_{\nabla \mathbf{u}}=1$ (max)',
    #             r'w/ $\lambda_{\nabla \mathbf{u}}=0$',
    #             r'w/ $\lambda_{\nabla \mathbf{u}}=1$',
    #             ]

    #     else:
    #         title = 'divPlot_both'
    #         file_list = [
    #             'log/plot/run_smoke_pos10_f200_0110_171030_10-tag-loss_g_loss_z_div_max.csv',
    #             'log/plot/run_smoke_pos10_f200_0110_171030_10-tag-loss_g_loss_div_max.csv',
    #             'log/plot/run_smoke_pos10_f200_0110_171030_11-tag-loss_g_loss_z_div_max.csv',
    #             'log/plot/run_smoke_pos10_f200_0110_171030_11-tag-loss_g_loss_div_max.csv',
    #             'log/plot/run_smoke_pos10_f200_0110_171030_10-tag-loss_g_loss_z_div.csv',
    #             'log/plot/run_smoke_pos10_f200_0110_171030_10-tag-loss_g_loss_div.csv',
    #             'log/plot/run_smoke_pos10_f200_0110_171030_11-tag-loss_g_loss_z_div.csv',
    #             'log/plot/run_smoke_pos10_f200_0110_171030_11-tag-loss_g_loss_div.csv',
    #         ]
    #         yr = [1e-3, 1e0]
    #         ytick = [1e-3, 1e-2, 1e-1, 1e0]
    #         ypos = [
    #             0.01,-0.01,-0.01,-0.015,0.001,0,0,-0.0001,
    #             # -0.015, -0.001, 
    #             # -0.001, 0,
    #             ]
    #         label = [
    #             r'Cont. w/ $\lambda_{\nabla \mathbf{u}}=0$ (max)',
    #             r'Disc. w/ $\lambda_{\nabla \mathbf{u}}=0$ (max)',
    #             r'Cont. w/ $\lambda_{\nabla \mathbf{u}}=1$ (max)',
    #             r'Disc. w/ $\lambda_{\nabla \mathbf{u}}=1$ (max)',
    #             r'Cont. w/ $\lambda_{\nabla \mathbf{u}}=0$',
    #             r'Disc. w/ $\lambda_{\nabla \mathbf{u}}=0$',
    #             r'Cont. w/ $\lambda_{\nabla \mathbf{u}}=1$',
    #             r'Disc. w/ $\lambda_{\nabla \mathbf{u}}=1$',
    #             ]

    #     csvplot(title, file_list, xr, yr, ytick, ypos)