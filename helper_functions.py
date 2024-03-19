from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import slack
from math import sqrt

rcParams.update({'figure.autolayout': True})
sns.set_theme()


def compute_std(values, probs):
    n = len(values)
    mu = sum(values[i] * probs[i] for i in range(1, n + 1))
    std = sqrt(sum((values[i] - mu) ** 2 * probs[i] for i in range(1, n + 1)))
    return std


def update_slack(message):
    SLACK_TOKEN = "xoxb-4821081014768-4797412824562-xVP6HLngtp8s4k9fbizVz5kb"
    client = slack.WebClient(token=SLACK_TOKEN)
    client.chat_postMessage(channel='#random', text=f'{message}')


def get_to_neighbors(node, edge_dict):
    to_neighbors = {j for (i, j) in edge_dict if i == node}
    return to_neighbors


def get_from_neighbors(node, edge_dict):
    from_neighbors = {i for (i, j) in edge_dict if j == node}
    return from_neighbors


def plot_infrastructure_by_scenario(risk_level_col, case_num, file_name, num_Distributions):
    capture = []
    storage = []
    transportation = []
    standard_dev = []
    risk_level = 0
    distributions = {1: 'Uniform', 2: 'Small Peak', 3: "Med. Peak", 4: 'Large Peak', 5: 'Spike'}
    print(list(distributions.values()))
    for distribution in range(1, num_Distributions + 1):
        cur_sheet_name = f'Distribution{distribution}'
        dataframe = pd.read_excel(f'Cases/Case{case_num}/Results/{file_name}', sheet_name=cur_sheet_name)
        capture_costs = dataframe.loc[risk_level_col]['Stage One Capture Costs']
        transportation_costs = dataframe.loc[risk_level_col]['Stage One Transportation Costs']
        storage_costs = dataframe.loc[risk_level_col]['Stage One Storage_costs']

        capture.append(capture_costs)
        transportation.append(transportation_costs)
        storage.append(storage_costs)

        if not risk_level:
            risk_level = dataframe.loc[risk_level_col]['Eta']

    capture = array(capture)
    storage = array(storage)
    transportation = array(transportation)
    x = list(range(1, 6))
    fig, ax = plt.subplots()
    plt.style.use("seaborn-v0_8-darkgrid")
    width = 0.4
    ax.bar(x, capture, width, label='Capture Costs')
    ax.bar(x, transportation, width, bottom=capture, label='Transportation Costs')
    ax.bar(x, storage, width, bottom=capture + transportation, label='Storage Costs')
    plt.ylim([0, 40])
    ax.set_xticks(x, labels=list(distributions.values()))


    ax.set_ylabel('Total Investment (Billion $)', fontsize=13)
    ax.set_xlabel('Distribution', fontsize=13)
    ax.set_title(f'First Stage Investment Under each Distribution for $\eta = {{{risk_level}}}$', fontsize=13)
    t = ax.yaxis.get_offset_text()
    t.set_x(-.1)
    ax.legend()

    plt.savefig(f'Cases/Case{case_num}/Results/Infastructure_by_scenario{risk_level}.png', bbox_inches='tight', dpi=300)
    plt.show()
    print()


def plot_infrastructure_by_risk(case_num, file_name, distribution_num):
    distributions = {1: 'Uniform', 2: 'Small Peak', 3: "Med. Peak", 4: 'Large Peak', 5: 'Spike'}

    capture = []
    storage = []
    transportation = []
    risk_list = []
    cur_sheet_name = f'Distribution{distribution_num}'
    dataframe = pd.read_excel(f'Cases/Case{case_num}/Results/{file_name}', sheet_name=cur_sheet_name)
    num_risk_levels = len(dataframe.index)

    for eta in range(0, num_risk_levels):
        capture_costs = dataframe.loc[eta]['Stage One Capture Costs']
        transportation_costs = dataframe.loc[eta]['Stage One Transportation Costs']
        storage_costs = dataframe.loc[eta]['Stage One Storage_costs']
        risk = dataframe.loc[eta]['Eta']

        capture.append(capture_costs)
        transportation.append(transportation_costs)
        storage.append(storage_costs)
        risk_list.append(risk)

    std_dev = dataframe.loc[1]['Standard Deviation']
    capture = array(capture)
    storage = array(storage)
    transportation = array(transportation)
    fig, ax = plt.subplots()
    plt.style.use("seaborn-v0_8-darkgrid")
    width = 0.4
    ax.bar(risk_list, capture, width, label='Capture Costs')
    ax.bar(risk_list, transportation, width, bottom=capture, label='Transportation Costs')
    ax.bar(risk_list, storage, width, bottom=capture + transportation, label='Storage Costs')

    ax.set_ylabel('Total Investment (Billion $)', fontsize=13)
    ax.set_xlabel(r'Risk Aversion Parameter $\eta$ (less risk averse $\longrightarrow$)', fontsize=13)
    ax.set_title(f'Investment vs Risk Aversion Level with Distribution: {distributions[distribution_num]}', fontsize=13)
    t = ax.yaxis.get_offset_text()
    t.set_x(-.1)
    ax.legend()
    plt.ylim([0, 40])
    plt.savefig(f'Cases/Case{case_num}/Results/Infrastructure_v_risk_{distributions[distribution_num]}.png',
                bbox_inches='tight', dpi=300)
    plt.show()


def plot_expected_total_CO2(case_num, filename, num_Distributions):
    plt.figure(figsize=(10, 10))
    plt.style.use("seaborn-v0_8-darkgrid")
    distributions = {1: 'Uniform', 2: 'Small Peak', 3: "Med. Peak", 4: 'Large Peak', 5: 'Spike'}
    for i in range(1, num_Distributions + 1):
        exp_string = f'Distribution{i}'

        sheet = pd.read_excel(f'Cases/Case{case_num}/Results/{filename}', sheet_name=exp_string)

        X_vals = list(sheet['Eta'])[::-1]
        Co_2 = list(sheet['Total Expected CO2 Captured'])[::-1]

        plt.plot(X_vals, Co_2, label=f'Distribution:  {distributions[i]}', linewidth=7)
    exp_string = f'Deterministic'
    sheet = pd.read_excel(f'Cases/Case{case_num}/Results/{filename}', sheet_name=exp_string)
    CO_2_captured = int(sheet['Total Expected CO2 Captured'][2])
    print(CO_2_captured)
    risk_vals = [0, 0.25, 0.5, 0.75, 1.5, 3, 7]
    plt.plot(risk_vals, [CO_2_captured for _ in range(len(risk_vals))], '--',
             label=f'Deterministic Scenario = (\$85,$60) (Control)', linewidth=7)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(r' Risk-Aversion Parameter $\eta$ (more risk averse $\longrightarrow$)', fontsize=20)
    plt.ylabel(r'Expected CO$_2$ Captured (Mt)', fontsize=20)
    plt.title(r'Expected CO$_2$ Captured vs Risk-Aversion Levels', fontsize=22)

    plt.savefig(f'Cases/Case{case_num}/Results/ExpectedCO_2.png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_first_stage_CO2(case_num, filename, num_Distributions):
    plt.figure(figsize=(10, 10))
    plt.style.use("seaborn-v0_8-darkgrid")
    distributions = {1: 'Uniform', 2: 'Small Peak', 3: "Med. Peak", 4: 'Large Peak', 5: 'Spike'}
    for i in range(1, num_Distributions + 1):
        exp_string = f'Distribution{i}'

        sheet = pd.read_excel(f'Cases/Case{case_num}/Results/{filename}', sheet_name=exp_string)

        X_vals = list(sheet['Eta'])[::-1]
        Co_2 = list(sheet['Total Stage one CO2 Captured'])[::-1]

        plt.plot(X_vals, Co_2, label=f'Distribution:  {distributions[i]}', linewidth=7)
    exp_string = f'Deterministic'
    sheet = pd.read_excel(f'Cases/Case{case_num}/Results/{filename}', sheet_name=exp_string)
    CO_2_captured = int(sheet['Total Stage one CO2 Captured'][2])/3
    print(CO_2_captured)
    risk_vals = [0, 0.25, 0.5, 0.75, 1.5, 3, 7]
    plt.plot(risk_vals, [CO_2_captured for _ in range(len(risk_vals))], '--',
             label=f'Deterministic Scenario = (\$85,$60) (Control)', linewidth=7)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(r' Risk-Aversion Parameter $\eta$ (more risk averse $\longrightarrow$)', fontsize=20)
    plt.ylabel(r'First Stage CO$_2$ Captured (Mt)', fontsize=20)
    plt.title(r'First Stage CO$_2$ Captured vs Risk-Aversion Levels', fontsize=22)

    plt.savefig(f'Cases/Case{case_num}/Results/first_stageCO2.png', bbox_inches='tight', dpi=300)
    plt.show()

def plot_second_stage_CO2(case_num, filename, num_Distributions):
    plt.figure(figsize=(10, 10))
    plt.style.use("seaborn-v0_8-darkgrid")
    distributions = {1: 'Uniform', 2: 'Small Peak', 3: "Med. Peak", 4: 'Large Peak', 5: 'Spike'}
    for i in range(1, num_Distributions + 1):
        exp_string = f'Distribution{i}'

        sheet = pd.read_excel(f'Cases/Case{case_num}/Results/{filename}', sheet_name=exp_string)

        X_vals = list(sheet['Eta'])[::-1]
        total_Co_2 = list(sheet['Total Expected CO2 Captured'])[::-1]
        first_stage_CO2 = list(sheet['Total Stage one CO2 Captured'])[::-1]
        Co_2 = [total_Co_2[i] - first_stage_CO2[i] for i in range(len(total_Co_2))]
        plt.plot(X_vals, Co_2, label=f'Distribution:  {distributions[i]}', linewidth=7)
    exp_string = f'Deterministic'
    sheet = pd.read_excel(f'Cases/Case{case_num}/Results/{filename}', sheet_name=exp_string)
    CO_2_captured = 2 * int(sheet['Total Stage one CO2 Captured'][2]) / 3
    print(CO_2_captured)
    risk_vals = [0, 0.25, 0.5, 0.75, 1.5, 3, 7]
    plt.plot(risk_vals, [CO_2_captured for _ in range(len(risk_vals))], '--',
             label=f'Deterministic Scenario = (\$85,$60) (Control)', linewidth=7)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(r' Risk-Aversion Parameter $\eta$ (more risk averse $\longrightarrow$)', fontsize=20)
    plt.ylabel(r'Expected CO$_2$ Captured (Mt)', fontsize=20)
    plt.title(r'Expected Second Stage CO$_2$ Captured vs Risk-Aversion Levels', fontsize=22)

    plt.savefig(f'Cases/Case{case_num}/Results/second_stageCO2.png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_det_CO2(case_num, filename, num_Distributions):
    plt.figure(figsize=(10, 10))
    plt.style.use("seaborn-v0_8-darkgrid")
    distributions = {1: 'Uniform', 2: 'Small Peak', 3: "Med. Peak", 4: 'Large Peak', 5: 'Spike'}
    for i in range(1, num_Distributions + 1):
        exp_string = f'Distribution{i}'

        sheet = pd.read_excel(f'Cases/Case{case_num}/Results/{filename}', sheet_name=exp_string)

        X_vals = list(sheet['Eta'])[::-1]
        Co_2 = list(sheet['Total Stage one CO2 Captured'])[::-1]

        plt.plot(X_vals, Co_2, label=f'Distribution:  {distributions[i]}', linewidth=7)
    exp_string = f'Deterministic'
    sheet = pd.read_excel(f'Cases/Case{case_num}/Results/{filename}', sheet_name=exp_string)
    CO_2_captured = int(sheet['Total Stage one CO2 Captured'][2])/3
    print(CO_2_captured)
    risk_vals = [0, 0.25, 0.5, 0.75, 1.5, 3, 7]
    plt.plot(risk_vals, [CO_2_captured for _ in range(len(risk_vals))], '--',
             label=f'Deterministic Scenario = (\$85,$60) (Control)', linewidth=7)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(r' Risk-Aversion Parameter $\eta$ (more risk averse $\longrightarrow$)', fontsize=20)
    plt.ylabel(r'Expected CO$_2$ Captured (Mt)', fontsize=20)
    plt.title(r'Expected CO$_2$ Captured vs Risk-Aversion Levels', fontsize=22)

    plt.savefig(f'Cases/Case{case_num}/Results/Figures/first_stageCO2.png', bbox_inches='tight', dpi=300)
    plt.show()

def plot_distribution(scenarios, probabilities, distribution_name, case_num):
    plt.bar(scenarios, probabilities, color='blue', width=0.5)
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.ylim(0, 1)
    plt.xlabel("45Q Scenario (Storage, Utilization) ($/ton)", fontsize=15)
    plt.ylabel("Probability", fontsize=16)
    plt.title(f"P.M.F. of Distribution: {distribution_name}", fontsize=19)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.savefig(f'Cases/Case{case_num}/Results/Figures/Distribution_Pmfs/distribution_{distribution_name}.png', dpi=300)
    plt.show()


# Deterministic Plotting

def plot_total_CO2(case_num, filename):
    exp_string = f'Det1'
    sheet = pd.read_excel(f'Cases/Case{case_num}/Results/{filename}', sheet_name=exp_string)

    X_vals = ['($0,$0)', '($42.5,$30)', '($85,$60)', '($127.5,$90)', '($170,$120)']
    Co_2 = list(sheet['Total CO2 Captured'])

    plt.bar(X_vals, Co_2, width=0.5)
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.xlabel(r'Future Incentive Values (Storage, EOR)', fontsize=18)
    plt.ylabel(r'Total CO$_2$ Captured (Mt)', fontsize=18)
    plt.title(r'Total CO$_2$ Captured by Scenario', fontsize=18)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=13)

    plt.savefig(f'Cases/Case{case_num}/Results/Figures/Det_CO_2_Final.png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_infrastructure_by_scenario_Det(case_num, file_name):
    capture = []
    storage = []
    transportation = []
    print('$')
    distribution_list = ['(\$0,$0)', '(\$42.5,$30)', '(\$85,$60)', '(\$127.5,$90)', '(\$170,$120)']
    sheet_name = f'Det1'
    dataframe = pd.read_excel(f'Cases/Case{case_num}/Results/Figures/{file_name}', sheet_name=sheet_name)
    num_distributions = len(dataframe.index)

    for dist in range(0, num_distributions):
        capture_costs = dataframe.loc[dist]['Capture Costs']
        transportation_costs = dataframe.loc[dist]['Transportation Costs']
        storage_costs = dataframe.loc[dist]['Storage Costs']

        capture.append(capture_costs)
        transportation.append(transportation_costs)
        storage.append(storage_costs)

    capture = array(capture)
    storage = array(storage)
    transportation = array(transportation)
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots()
    width = 0.5
    ax.bar(distribution_list, capture, width, label='Capture Costs')
    ax.bar(distribution_list, transportation, width, bottom=capture, label='Transportation Costs')
    ax.bar(distribution_list, storage, width, bottom=capture + transportation, label='Storage Costs')
    ax.legend(fontsize=16)

    ax.set_ylabel('Total Investment (Billion $)', fontsize=20)
    ax.set_xlabel('Future Incentive Values (Storage, EOR)', fontsize=18)
    ax.set_title(f'Investment vs Scenario', fontsize=22)

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=12)

    t = ax.yaxis.get_offset_text()
    t.set_x(-.1)

    plt.savefig(f'Cases/Case{case_num}/Results/Figures/Infrastructure_v_risk_std_Final.png',
                bbox_inches='tight',
                dpi=300)
    plt.show()
