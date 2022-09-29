"""
We put here function for plotting various quantities.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def Plots_Motivating_Example(csv_file_name):
    csv_file_name = "./motivating_example.csv"
    results = pd.read_csv(csv_file_name)
    results['Method'].replace({'Lasso': 'Lasso', 'OMP': 'OMP'},inplace=True)


    plt.rcParams['text.usetex'] = True
    SMALL_SIZE = 20
    MEDIUM_SIZE = 21
    BIGGER_SIZE = 22

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



    # plot recovery error as a function of M
    df = results
    df = df[df["Cardinality"]==3]
    df = df[df["Noise STD"]==0.1]
    df = df[df["Method"].isin(['Exhaustive L0', 'Lasso', 'OMP'])]



    sns.lineplot(x = "M", y = "Infeasible X-Recovery L2 Err", ci=68, err_style='bars', style="Method",markers=True, dashes=False,markersize=11,linewidth=3,
                 data = df[["Experiment", "Method","M","Infeasible X-Recovery L2 Err"]],hue='Method')
    plt.ylabel(r'Reconst. Err $\mathbf{\|x-\hat{x}\|_2/\|x\|_2}$')
    plt.xlabel(r'Number of rows $M$')
    plt.xlim([5.5,16.5])
    plt.xticks([6, 8, 10, 12, 14, 16])
    plt.ylim([-0.05,1])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.savefig('motivating_rec_err_vs_M__k=3_noise_std_01.png', bbox_inches='tight', dpi=300)
    plt.show()



    # plot recovery error as a function of noise std
    df = results
    df = df[df["Cardinality"]==3]
    df = df[df["M"]==8]
    df = df[df["Noise STD"]<=0.3]
    df = df[df["Method"].isin(['Exhaustive L0', 'Lasso', 'OMP'])]


    sns.lineplot(x = "Noise STD", y = "Infeasible X-Recovery L2 Err", ci=68, err_style='bars', style="Method",markers=True, dashes=False,markersize=11,linewidth=3,
                 data = df[["Experiment", "Method","Noise STD","Infeasible X-Recovery L2 Err"]],hue='Method')
    plt.ylabel(r'Reconst. Err $\mathbf{\|x-\hat{x}\|_2/\|x\|_2}$')
    plt.xlabel(r'Noise STD $\sigma$')
    plt.legend().remove()
    plt.ylim([-0.05,1])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.savefig('motivating_rec_err_vs_noise__k=3_M_8.png', bbox_inches='tight', dpi=300)
    plt.show()

    # plot recovery error as a function of cardinality
    df = results
    df = df[df["Cardinality"]<=4]
    df = df[df["M"]==8]
    df = df[df["Noise STD"]==0.1]
    df = df[df["Method"].isin(['Exhaustive L0', 'Lasso', 'OMP'])]


    sns.lineplot(x = "Cardinality", y = "Infeasible X-Recovery L2 Err", ci=68, err_style='bars', style="Method",markers=True, dashes=False,markersize=11,linewidth=3,
                 data = df[["Experiment", "Method","Cardinality","Infeasible X-Recovery L2 Err"]],hue='Method')
    plt.ylabel(r'Reconst. Err $\mathbf{\|x-\hat{x}\|_2/\|x\|_2}$')
    plt.xlabel(r'Cardinality $k$')
    plt.legend().remove()
    plt.ylim([-0.05,1])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.savefig('motivating_rec_err_vs_cardinality__noise_std_01_M_8.png', bbox_inches='tight', dpi=300)
    plt.show()



     # plot L0 error as a function of M
    df = results
    df = df[df["Cardinality"]==3]
    df = df[df["Noise STD"]==0.1]
    df = df[df["Method"].isin(['Exhaustive L0', 'Lasso', 'OMP'])]


    sns.lineplot(x = "M", y = "Infeasible X-Recovery L0 Err", ci=68, err_style='bars', style="Method",markers=True, dashes=False,markersize=11,linewidth=3,
                 data = df[["Experiment", "Method","M","Infeasible X-Recovery L0 Err"]],hue='Method')
    plt.ylabel(r'Support Err $\mathbf{\|\mathcal{S} - \hat{\mathcal{S}} \|_0}$')
    plt.xlabel(r'Number of rows $M$')
    plt.xlim([5.5,16.5])
    plt.xticks([6, 8, 10, 12, 14, 16])
    plt.ylim([-0.1,3.1])
    plt.yticks([0.0, 1.0, 2.0, 3.0])

    plt.legend().remove()

    plt.savefig('motivating_supp_err_vs_M__k=3_noise_std_01.png', bbox_inches='tight', dpi=300)
    plt.show()


    # plot recovery error as a function of noise std
    df = results
    df = df[df["Cardinality"]==3]
    df = df[df["M"]==8]
    df = df[df["Noise STD"]<=0.3]
    df = df[df["Method"].isin(['Exhaustive L0', 'Lasso', 'OMP'])]


    sns.lineplot(x = "Noise STD", y = "Infeasible X-Recovery L0 Err", ci=68, err_style='bars', style="Method",markers=True, dashes=False,markersize=11,linewidth=3,
                 data = df[["Experiment", "Method","Noise STD","Infeasible X-Recovery L0 Err"]],hue='Method')
    plt.ylabel(r'Support Err $\mathbf{\|\mathcal{S} - \hat{\mathcal{S}} \|_0}$')
    plt.xlabel(r'Noise STD $\sigma$')
    plt.legend().remove()
    plt.ylim([-0.1,3.1])
    plt.yticks([0.0, 1.0, 2.0, 3.0])

    plt.savefig('motivating_supp_err_vs_noise__k=3_M_8.png', bbox_inches='tight', dpi=300)
    plt.show()

    # plot recovery error as a function of cardinality
    df = results
    df = df[df["Cardinality"]<=4]
    df = df[df["M"]==8]
    df = df[df["Noise STD"]==0.1]
    df = df[df["Method"].isin(['Exhaustive L0', 'Lasso', 'OMP'])]


    sns.lineplot(x = "Cardinality", y = "Infeasible X-Recovery L0 Err", ci=68, err_style='bars', style="Method",markers=True, dashes=False,markersize=11,linewidth=3,
                 data = df[["Experiment", "Method","Cardinality","Infeasible X-Recovery L0 Err"]],hue='Method')
    plt.ylabel(r'Support Err $\mathbf{\|\mathcal{S} - \hat{\mathcal{S}} \|_0}$')
    plt.xlabel(r'Cardinality $k$')
    plt.legend().remove()
    plt.ylim([-0.1,3.1])
    plt.yticks([0.0, 1.0, 2.0, 3.0])

    plt.savefig('motivating_supp_err_vs_cardinality__noise_std_01_M_8.png', bbox_inches='tight', dpi=300)
    plt.show()




def Plot_Binary_Experiments(csv_file_name):
    csv_file_name = "./binary_baselines.csv"
    results_lasso_omp = pd.read_csv(csv_file_name)
    results_lasso_omp['Method'].replace({'Lasso': 'Lasso', 'OMP': 'OMP'},inplace=True)

    xls_file_name = "../results/binary_lightsolver.csv"
    results_ours = pd.read_csv(xls_file_name)
    column_names = ['Type A', 'type_x0', 'N', 'M', 'Cardinality', 'Noise STD','Experiment', 'Infeasible X-Recovery L0 Err', 'Method']
    results_ours.columns = column_names
    results_ours['Method'].replace({'QUBO L0': 'LightSolver'},inplace=True)
    results_ours['Infeasible X-Recovery L2 Err'] = np.sqrt(results_ours['Infeasible X-Recovery L0 Err']/results_ours['Cardinality'])

    column_names.append('Infeasible X-Recovery L2 Err')
    results_lasso_omp = results_lasso_omp[column_names]
    all_results = pd.concat([results_ours,results_lasso_omp], ignore_index=True)

    plt.rcParams['text.usetex'] = True
    SMALL_SIZE = 20
    MEDIUM_SIZE = 21
    BIGGER_SIZE = 22

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



    # plot recovery error as a function of M
    df = all_results
    df = df[df["Cardinality"]==30]
    df = df[df["Noise STD"]==0.1]
    df = df[df["Method"].isin(['LightSolver', 'Lasso', 'OMP'])]
    df = df[df["M"].isin([60, 80, 100, 130, 160])]

    sns.lineplot(x = "M", y = "Infeasible X-Recovery L2 Err", ci=68, err_style='bars', style="Method",markers=True, dashes=False,markersize=11,linewidth=3,
                 data = df[["Experiment", "Method","M","Infeasible X-Recovery L2 Err"]],hue='Method')
    plt.ylabel(r'Reconst. Err $\mathbf{\|x-\hat{x}\|_2/\|x\|_2}$')
    plt.xlabel(r'Number of rows $M$')
    plt.xlim([55,165])
    plt.xticks([60, 80, 100, 130, 160])
    plt.ylim([-0.05,1])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.savefig('binary_rec_err_vs_M__k=30_noise_std_01.png', bbox_inches='tight', dpi=300)
    plt.show()



    # plot recovery error as a function of noise std
    df = all_results
    df = df[df["Cardinality"]==30]
    df = df[df["M"]==80]
    df = df[df["Noise STD"]<=0.3]
    df = df[df["Method"].isin(['LightSolver', 'Lasso', 'OMP'])]


    sns.lineplot(x = "Noise STD", y = "Infeasible X-Recovery L2 Err", ci=68, err_style='bars', style="Method",markers=True, dashes=False,markersize=11,linewidth=3,
                 data = df[["Experiment", "Method","Noise STD","Infeasible X-Recovery L2 Err"]],hue='Method')
    plt.ylabel(r'Reconst. Err $\mathbf{\|x-\hat{x}\|_2/\|x\|_2}$')
    plt.xlabel(r'Noise STD $\sigma$')
    plt.legend().remove()
    plt.ylim([-0.05,1])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.savefig('binary_rec_err_vs_noise__k=30_M_80.png', bbox_inches='tight', dpi=300)
    plt.show()

    # plot recovery error as a function of cardinality
    df = all_results
    df = df[df["Cardinality"]<=30]
    df = df[df["M"]==80]
    df = df[df["Noise STD"]==0.1]
    df = df[df["Method"].isin(['LightSolver', 'Lasso', 'OMP'])]


    sns.lineplot(x = "Cardinality", y = "Infeasible X-Recovery L2 Err", ci=68, err_style='bars', style="Method",markers=True, dashes=False,markersize=11,linewidth=3,
                 data = df[["Experiment", "Method","Cardinality","Infeasible X-Recovery L2 Err"]],hue='Method')
    plt.ylabel(r'Reconst. Err $\mathbf{\|x-\hat{x}\|_2/\|x\|_2}$')
    plt.xlabel(r'Cardinality $k$')
    plt.legend().remove()
    plt.ylim([-0.05,1])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks([10, 20, 30])

    plt.savefig('binary_rec_err_vs_cardinality__noise_std_01_M_80.png', bbox_inches='tight', dpi=300)
    plt.show()



    # plot L0 error as a function of M
    df = all_results
    df = df[df["Cardinality"]==30]
    df = df[df["Noise STD"]==0.1]
    df = df[df["Method"].isin(['LightSolver', 'Lasso', 'OMP'])]
    df = df[df["M"].isin([60, 80, 100, 130, 160])]


    sns.lineplot(x = "M", y = "Infeasible X-Recovery L0 Err", ci=68, err_style='bars', style="Method",markers=True, dashes=False,markersize=11,linewidth=3,
                 data = df[["Experiment", "Method","M","Infeasible X-Recovery L0 Err"]],hue='Method')
    plt.ylabel(r'Support Err $\mathbf{\|\mathcal{S} - \hat{\mathcal{S}} \|_0}$')
    plt.xlabel(r'Number of rows $M$')
    plt.xlim([55,165])
    plt.xticks([60, 80, 100, 130, 160])
    plt.ylim([-1,27])
    plt.yticks([0.0, 5, 10, 15, 20, 25])

    plt.legend().remove()

    plt.savefig('binary_supp_err_vs_M__k=30_noise_std_01.png', bbox_inches='tight', dpi=300)
    plt.show()



    # plot L0 error as a function of noise std
    df = all_results
    df = df[df["Cardinality"]==30]
    df = df[df["M"]==80]
    df = df[df["Noise STD"]<=0.3]
    df = df[df["Method"].isin(['LightSolver', 'Lasso', 'OMP'])]


    sns.lineplot(x = "Noise STD", y = "Infeasible X-Recovery L0 Err", ci=68, err_style='bars', style="Method",markers=True, dashes=False,markersize=11,linewidth=3,
                 data = df[["Experiment", "Method","Noise STD","Infeasible X-Recovery L0 Err"]],hue='Method')
    plt.ylabel(r'Support Err $\mathbf{\|\mathcal{S} - \hat{\mathcal{S}} \|_0}$')
    plt.xlabel(r'Noise STD $\sigma$')
    plt.legend().remove()
    plt.ylim([-1,27])
    plt.yticks([0.0, 5, 10, 15, 20, 25])

    plt.savefig('binary_supp_err_vs_noise__k=30_M_80.png', bbox_inches='tight', dpi=300)
    plt.show()

    # plot L0 error as a function of cardinality
    df = all_results
    df = df[df["Cardinality"]<=30]
    df = df[df["M"]==80]
    df = df[df["Noise STD"]==0.1]
    df = df[df["Method"].isin(['LightSolver', 'Lasso', 'OMP'])]


    sns.lineplot(x = "Cardinality", y = "Infeasible X-Recovery L0 Err", ci=68, err_style='bars', style="Method",markers=True, dashes=False,markersize=11,linewidth=3,
                 data = df[["Experiment", "Method","Cardinality","Infeasible X-Recovery L0 Err"]],hue='Method')
    plt.ylabel(r'Support Err $\mathbf{\|\mathcal{S} - \hat{\mathcal{S}} \|_0}$')
    plt.xlabel(r'Cardinality $k$')
    plt.legend().remove()
    plt.ylim([-1,27])
    plt.yticks([0.0, 5, 10, 15, 20, 25])
    plt.xticks([10, 20, 30])

    plt.savefig('binary_supp_err_vs_cardinality__noise_std_01_M_80.png', bbox_inches='tight', dpi=300)
    plt.show()


def Plot_Err_VS_M_FP(csv_file_name):
    csv_file_name = "./fixed_point_baselines.csv"
    results_lasso_omp = pd.read_csv(csv_file_name)
    results_lasso_omp['Method'].replace({'Lasso': 'Lasso', 'OMP': 'OMP'},inplace=True)

    xls_file_name = "./fixed_point_lightsolver.csv"
    results_ours = pd.read_csv(xls_file_name)
    column_names = ['Type A', 'type_x0', 'N', 'M', 'Cardinality', 'Noise STD','Experiment', 'Infeasible X-Recovery L0 Err', 'Infeasible X-Recovery L2 Err', 'Method']
    results_ours.columns = column_names
    results_ours['Method'].replace({'QUBO L0': 'LightSolver'},inplace=True)

    results_lasso_omp = results_lasso_omp[column_names]
    all_results = pd.concat([results_ours,results_lasso_omp], ignore_index=True)

    plt.rcParams['text.usetex'] = True
    SMALL_SIZE = 20
    MEDIUM_SIZE = 21
    BIGGER_SIZE = 22

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



    # plot recovery error as a function of M
    df = all_results
    df = df[df["Cardinality"]==10]
    df = df[df["Noise STD"]==0.1]
    df = df[df["Method"].isin(['LightSolver', 'Lasso', 'OMP'])]
    df = df[df["M"].isin([30, 35, 40, 50, 60])]

    sns.lineplot(x = "M", y = "Infeasible X-Recovery L2 Err", ci=68, err_style='bars', style="Method",markers=True, dashes=False,markersize=11,linewidth=3,
                 data = df[["Experiment", "Method","M","Infeasible X-Recovery L2 Err"]],hue='Method')
    plt.ylabel(r'Reconst. Err $\mathbf{\|x-\hat{x}\|_2/\|x\|_2}$')
    plt.xlabel(r'Number of rows $M$')

    plt.savefig('fixed_point_rec_err_vs_M__k=10_noise_std_01.png', bbox_inches='tight', dpi=300)
    plt.show()


    # plot L0 error as a function of M
    df = all_results
    df = df[df["Cardinality"]==10]
    df = df[df["Noise STD"]==0.1]
    df = df[df["Method"].isin(['LightSolver', 'Lasso', 'OMP'])]
    df = df[df["M"].isin([30, 35, 40, 50, 60])]


    sns.lineplot(x = "M", y = "Infeasible X-Recovery L0 Err", ci=68, err_style='bars', style="Method",markers=True, dashes=False,markersize=11,linewidth=3,
                 data = df[["Experiment", "Method","M","Infeasible X-Recovery L0 Err"]],hue='Method')
    plt.ylabel(r'Support Err $\mathbf{\|\mathcal{S} - \hat{\mathcal{S}} \|_0}$')
    plt.xlabel(r'Number of rows $M$')

    plt.legend().remove()

    plt.savefig('fixed_point_supp_err_vs_M__k=10_noise_std_01.png', bbox_inches='tight', dpi=300)
    plt.show()
