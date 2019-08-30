import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas
import math
import sys
from scipy import stats


def calculateP(params, lc, chi2, normalize = False):
    degreesFreedom = len(lc) - len(params)
    if normalize:
        scaling = np.min(chi2)
    else: scaling = 1
    chi2 = (chi2 * degreesFreedom) / scaling
    probs = np.zeros_like(chi2)
    for i in range(len(chi2)):
        probs[i] = stats.chi2.sf(chi2[i], degreesFreedom)
    return probs



path = '/scratch/leuven/324/vsc32406/'

run = sys.argv[1]

path = path + run

x = pandas.read_csv(path + '/chi2.txt', sep=' ')
x = x.rename(columns = {'#requiv@primary': 'requiv@primary'})

lc = np.loadtxt(path + '/lc_in.txt')

params = np.loadtxt(path + '/params.txt', dtype='str')

probabilities = calculateP(params, lc, x['chi2'], True)

x = x.assign(P=probabilities)
params_dic = OrderedDict()
params_error = OrderedDict()


min_p = 0.05
best = np.argmax(x['P'])
ind = x['P'] > min_p

for i in params:
    params_dic[i[0]] = [float(i[1]), float(i[2])]
    params_error[i[0]] = [min(x[i[0]][ind]), max(x[i[0]][ind]), x[i[0]][best]]


param_keys = params_dic.keys()

fig = plt.figure(figsize=(20,13))
plt.subplots_adjust(left=0.050, right=0.98, top=0.95, bottom=0.08)
n_rows = int(math.ceil(len(param_keys)/4.))
n_cols = 4

for i in range(len(param_keys)):
    ax = fig.add_subplot(n_rows,n_cols,i+1)
    ax.axvspan(params_error[param_keys[i]][0], params_error[param_keys[i]][1], alpha=0.3, color='red')
    ax.scatter(x[param_keys[i]].values, x['fitness'].values)
    ax.set_xlabel(param_keys[i], fontsize=14)
    ax.set_xlim(params_dic[param_keys[i]][0], params_dic[param_keys[i]][1])
    print param_keys[i] + ' - ' + str(params_error[param_keys[i]][2]) + '    [' + str(params_error[param_keys[i]][0]) + ', ' +  str(params_error[param_keys[i]][1]) + ']'
    # ax.set_ylim(0, 9000)

plt.show()



#0055_0195










def plotResults_new(diagnostic_name, multi=False, pdf = False, im_num = False, path = '.'):
    #Parameters vs fitness plot
    fig = plt.figure(figsize=(20,13))

    plt.subplots_adjust(left=0.050, right=0.98, top=0.95, bottom=0.08)
    bestModel = models[findBestModel()]
    fileName = maindir + '/' + object +'/pfw_' + object + '.ini'
    xRanges = readIni_new(fileName, parameters = True)
    errorRanges = findErrorRanges_new()
    #    cmap = get_cmap('RdYlGn')
    cmap = get_cmap('gist_rainbow') #colormap
    vmin, vmax = vars(bestModel)[diagnostic_name]/3., vars(bestModel)[diagnostic_name]*2.2  #used to normalize colors, adjust when using other colormap

    fit_diag_param_vals = {}
    fit_diag_params = list(fit_parameters)
    fit_diag_params.extend(diagnostic_parameters)
    fit_diag_params.extend([lines[i].ID for i in range(len(lines))])

    for i in fit_diag_params:
        fit_diag_param_vals[i] = []

    fit_diag_param_vals['alt_fitness']
    for i in range(len(models)):
        for j in fit_diag_params:
            fit_diag_param_vals[j].append(vars(models[i])[j])

    n_rows = int(ceil(len(fit_parameters)/4.))
    n_cols = 4
    for j,i in enumerate(fit_parameters):
        plot_results_pannel(i, diagnostic_name, fit_diag_param_vals, fig, [n_rows, n_cols, j+1], errorRanges[i], vmin, vmax, cmap, bestModel, xRanges)

    print
    #plt.suptitle('Parameters vs '+all_parameter_names_dict[diagnostic_name]+' for '+object,fontsize=20)
    plt.show()
    return


def plot_results_pannel(fit_param, diagnostic_param, fit_diag_param_vals, fig, fig_params, errorRanges, vmin, vmax, cmap, bestModel, xRanges):
    print 'Preparing ' + all_parameter_names_dict[fit_param] + ' vs. ' + all_parameter_names_dict[diagnostic_param] + '...'
    fig.subplots_adjust(hspace=0.4)
    ax = fig.add_subplot(fig_params[0],fig_params[1],fig_params[2])
    ax.fill([errorRanges[0], errorRanges[1], errorRanges[1], errorRanges[0]], [0, 0, vars(bestModel)[diagnostic_param] * 1.2, vars(bestModel)[diagnostic_param] * 1.2], 'b', alpha=0.1)
    ax.scatter(fit_diag_param_vals[fit_param], fit_diag_param_vals[diagnostic_param], c = fit_diag_param_vals[diagnostic_param], vmin=vmin, vmax=vmax, cmap = cmap)
    ax.set_xlabel(all_parameter_names_dict[fit_param], fontsize=14)
    ax.tick_params(labelleft=False, labelsize=14)
    if fig_params[2]%fig_params[1] == 1:
        ax.set_ylabel(all_parameter_names_dict[diagnostic_param], fontsize=14)
        ax.tick_params(labelleft=True, labelsize=14)
    #ax.set_title('Best model: ' + all_parameter_names_dict[fit_param] + ' = ' + str(vars(bestModel)[fit_param]), fontsize=12)
    ax.axvline(x=errorRanges[0], c = 'r')
    ax.axvline(x=errorRanges[1], c = 'r')
    ax.axvline(x=vars(bestModel)[fit_param], c = 'black', alpha=0.8, linewidth=1.5)
    ax.set_xlim(xRanges[fit_param][0], xRanges[fit_param][1])
    ax.set_ylim(0.6 * vars(bestModel)[diagnostic_param], vars(bestModel)[diagnostic_param] * 2)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    ax.text(0.5, 0.05, 'Best model: ' + all_parameter_names_dict[fit_param] + ' = ' + str(round(vars(bestModel)[fit_param], 3)), transform=ax.transAxes, fontsize=14, verticalalignment='bottom', horizontalalignment='center', bbox=props)
    ax.grid(True)
