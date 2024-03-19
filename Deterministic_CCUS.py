# We will implement a two-stage, risk-neutral, stochastic program for the CCUS Infrastructure problem.
import pyomo.environ as pyo
from math import inf
import csv
from numpy import format_float_scientific
import time
import pandas as pd
import matplotlib.pyplot as plt

from helper_functions import update_slack

# This model assumes you have the following files in your working directory:
# ++++++++++++++++++++
#     GeneralData.csv
#     Sources.csv
#     Sinks.csv
#     Pipeline.csv
# ++++++++++++++++++++
# See the Readme file for a description of the parameters in each file.  TODO: Add Readme


def get_to_neighbors(node, edge_dict):
    to_neighbors = {j for (i, j) in edge_dict if i == node}
    return to_neighbors


def get_from_neighbors(node, edge_dict):
    from_neighbors = {i for (i, j) in edge_dict if j == node}
    return from_neighbors


# =====================================================================================================================
# ================================  Load in General Model Data  =======================================================
# =====================================================================================================================
def run_exp(storage_incentive, utilization_incentive, base_storage=85, base_util=60, second_stage_start=inf, existing_sources={}, existing_sinks={}, existing_pipelines={}):
    start_time = time.time()
    file = open("Cases/Case6/DeterministicGeneralData.csv", encoding='latin-1')
    print('Reading in data')
    csv_reader = csv.reader(file)
    next(csv_reader)  # skip the header
    general_parameters = [int(x) for x in next(csv_reader) if x]
    num_D, num_pipelines, num_nodes, num_sources, num_sinks, num_time_periods, stage_one_storage_tax_credit, stage_one_util_tax_credit = general_parameters

    next(csv_reader)  # Skip the  secondary header.
    discount_and_years = [float(x) for x in next(csv_reader) if x]
    discount_rate = discount_and_years[0]
    period_length = int(discount_and_years[1])

    file.close()
    # ====================================================================================================================
    # =======================================  Model parameters ==========================================================
    # ====================================================================================================================
    # Initialize the data structures to contain the data for our problem.
    # Initialize our model instance
    model = pyo.ConcreteModel()

    model.OM_src = dict()  # dict of fixed costs for O/M at online sources. F_src_OM[i,t] is fixed cost for operating source i at time t
    model.B_src = dict()  # dict of fixed costs for constructing sources. B_src[i,t] is fixed cost for building source i at time t
    model.V_src = dict()  # The list of variable costs for capturing C02 at sources ($/ton C0_2). V_src[i,t] will be the variable cost per unit of C0_2 captured in time t at source i.
    model.Q_source_rate = dict()  # max amount of CO_2 that can be produced in a time period at t source i
    model.existing_sources = existing_sources  # The indices of sources that exist at T = 0 (initial conditions)
    model.source_loc = dict()

    model.OM_res = dict()  # dict of fixed costs for O/M of reservoirs. F_res[j,t] is fixed cost for operating reservoir j in time t.
    model.B_res = dict()  # dict of fixed costs for constructing reservoirs. B_res[j,t] is fixed cost for building reservoir j in time t.
    model.V_res = dict()  # Variable cost of capturing CO2 at reservoir j ($/ton C0_2). V_res[j,t] will be the variable cost per unit C0_2 in reservoir j.
    model.Q_res = dict()  # Capacity of reservoir j (MtCO2)
    model.Q_res_rate = dict()  # max injection into reservoir j in time t.
    model.existing_res = existing_sinks  # The indices of reservoirs that exist at T = 0 (initial conditions)

    model.OM_pipe = dict()  # F_pipe[i,j,d,t] is fixed cost for maintaining the pipeline between node i and j with diameter d in time period t.
    model.B_pipe = dict()  # F_pipe[i,j,d,t] is fixed cost for building the pipeline between node i and j with diameter d in time period t.
    model.V_pipe = dict()  # Cost to transport one tonne of CO2 on pipeline (i,j) with diameter d (MtCO2/yr) during period t.
    model.Q_pipe_max = dict()  # Maximum capacity of pipeline (i,j) with diameter d (MtCO2/yr).
    model.Q_pipe_min = dict()  # Minimum capacity of pipeline (i,j) with diameter d (MtCO2/yr).
    model.existing_pipelines = existing_pipelines  # The set of existing pipelines at (T = 0)

    model.V_util = dict()  # dict of the utilization returns $/unit captured at utilization site k in period t.

    model.storage_incentive_1 = {t: base_storage if t < second_stage_start else storage_incentive for t in
                                 range(1, num_time_periods + 1)}
    model.util_incentive_1 = {t: base_util if t < second_stage_start else utilization_incentive for t in
                              range(1, num_time_periods + 1)}

    # =====================================================================================================================
    # ======================= Read in data from files =====================================================================
    # =====================================================================================================================

    # +++++++++++++++++++++ Read in Source data ++++++++++++++++++++++++++++++++++++++
    file = open("Cases/Case6/Sources_6.csv", encoding='latin-1')
    csv_reader = csv.reader(file)
    source_set = set()
    next(csv_reader)  # Skip Header
    for _ in range(num_sources):

        values = next(csv_reader)
        cur_node = int(values[0])
        source_set.add(cur_node)

        values = [float(x) for x in values[1:-1]]  # Cut out ID # and Type. Just the capacity and cost components
        for t in range(1, num_time_periods + 1):
            model.Q_source_rate[cur_node], model.B_src[cur_node, t], model.OM_src[cur_node, t], model.V_src[
                cur_node, t], = values
            # Update the parameters based on if the change with the length of each period
            model.Q_source_rate[cur_node] *= period_length  # Each source can now capture (# yrs in period) x as much
            model.OM_src[cur_node, t] *= period_length  # O/M costs increase proportionally

    file.close()

    # +++++++++++++++++++++ Read in Sink data ++++++++++++++++++++++++++++++++++++++
    file = open("Cases/Case6/Sinks_6.csv", encoding='latin-1')
    csv_reader = csv.reader(file)
    sink_set = set()
    util_set = set()
    next(csv_reader)  # Skip the header
    for _ in range(num_sinks):

        values = next(csv_reader)
        cur_node = int(values[0])
        sink_set.add(cur_node)

        values = [float(x) for x in values[1:]]  # Cut out ID.
        is_util = 0  # Flag to check if res. is a utility site
        for t in range(1, num_time_periods + 1):
            model.Q_res[cur_node], model.Q_res_rate[cur_node], model.B_res[cur_node, t], model.OM_res[cur_node, t], \
                model.V_res[cur_node, t], is_util, model.V_util[cur_node] = values
            # Update values that depend on length of period
            model.Q_res_rate[cur_node] *= period_length
            model.OM_res[cur_node, t] *= period_length

        if is_util:
            util_set.add(cur_node)

    file.close()

    # Determine which nodes are part of the transition/junction node set (those that are neither sources nor sinks)
    transition_node_set = set()
    # for i in range(1, num_nodes + 1):
    #     if i not in sink_set and i not in source_set:
    #         transition_node_set.add(i)

    # +++++++++++++++++++++ Read in Pipeline data ++++++++++++++++++++++++++++++++++++++

    file = open("Cases/Case6/Arcs_6.csv", encoding='latin-1')
    csv_reader = csv.reader(file)
    next(csv_reader)  # skip the initial header

    # Read in the diameter capacity data
    diameter_data = next(csv_reader)
    edge_set = set()
    line = [float(x) for x in diameter_data if x]

    for d, val in enumerate(line):
        model.Q_pipe_max[d + 1] = val * period_length

    next(csv_reader)  # skip secondary header
    # Get cost multiplication factors
    pipe_cost_factors = dict()
    factors = next(csv_reader)
    factors = [float(x) for x in factors if x]

    for d in range(num_D):
        pipe_cost_factors[d + 1] = factors[d]

    next(csv_reader)
    for _ in range(num_pipelines):

        values = next(csv_reader)  # Get the next pipeline

        _, _, i, j, _, cap_cost, OM_cost, Var_cost = [int(float(x)) for x in values if x]

        edge_set.add((i, j))
        if i not in source_set and i not in sink_set:
            transition_node_set.add(i)

        if i not in source_set and i not in sink_set:
            transition_node_set.add(j)

        for t in range(1, num_time_periods + 1):
            for d in range(
                    num_D):  # Duplicate with indices flipped for the pipeline that could be built with flow in opposite direction
                model.B_pipe[i, j, d + 1, t] = cap_cost * pipe_cost_factors[d + 1]
                model.B_pipe[j, i, d + 1, t] = cap_cost * pipe_cost_factors[d + 1]

            for d in range(num_D):
                model.OM_pipe[i, j, d + 1, t] = OM_cost * period_length
                model.OM_pipe[j, i, d + 1, t] = OM_cost * period_length

            model.V_pipe[i, j, t] = Var_cost
            model.V_pipe[j, i, t] = Var_cost
    file.close()

    # =====================================================================================================================
    # ================================  Model Index Sets  =================================================================
    # =====================================================================================================================

    # Initialize the Index sets for our model

    model.E = edge_set  # The set of potential edges (i,j) for our model.
    model.R = sink_set  # Reservoir node set
    model.S = source_set  # Source nodes.
    model.U = util_set  # Set of utilization nodes.
    model.N = source_set.union(sink_set).union(transition_node_set)  # Set of all nodes.
    model.D = pyo.RangeSet(num_D)  # Set of pipeline diameters indices
    model.T = pyo.RangeSet(num_time_periods)  # Set of model time periods
    model.T_1 = [1]
    # Now we preprocess to generate sets of nodes that point to .
    model.to_neighbor_dict = dict()
    model.from_neighbor_dict = dict()
    for i in model.N:
        model.from_neighbor_dict[i] = get_from_neighbors(i, model.E)
        model.to_neighbor_dict[i] = get_to_neighbors(i, model.E)

    # ====================================================================================================================
    # ===================================First-Stage Decision Variables ==================================================
    # ====================================================================================================================

    # +++++++++++++++ First stage variables. They will all have the "var_1" subscript +++++++++++++++++++++++++++
    model.x_1 = pyo.Var(model.N, model.N, model.T, domain=pyo.Reals)  # Amount of flow from i -> j in period t
    model.y_1 = pyo.Var(model.N, model.N, model.D, model.T,
                        domain=pyo.Binary)  # Indicates if pipeline (i,j) with diameter d exists at time t
    model.abs_flow_1 = pyo.Var(model.N, model.N, model.T, domain=pyo.PositiveReals)

    model.a_1 = pyo.Var(model.S, model.T,
                        domain=pyo.PositiveReals)  # Amount of CO2 captured at source i during period t
    model.b_1 = pyo.Var(model.R, model.T,
                        domain=pyo.PositiveReals)  # Amount of CO2 injected into reservoir j (tCO2/yr)

    model.s_1 = pyo.Var(model.S, model.T, domain=pyo.Binary)  # Indicates if source at node i is open during period t
    model.r_1 = pyo.Var(model.R, model.T,
                        domain=pyo.Binary)  # Indicates if reservoir at node j is open during period t

    model.sigma_1 = pyo.Var(model.S, model.T,
                            domain=pyo.Binary)  # The indicator variable for if source i is constructed in period t
    model.rho_1 = pyo.Var(model.R, model.T,
                          domain=pyo.Binary)  # The indicator variable for if reservoir j is constructed in period t
    model.gamma_1 = pyo.Var(model.N, model.N, model.D, model.T,
                            domain=pyo.Binary)  # The indicator variable for if pipeline

    # ====================================================================================================================
    # ====================================== Model Constraints  =========================================================
    # ====================================================================================================================

    # +++++++++++++++++ First - Stage Constraints +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Read in and add existing
    # (Cons 1) Constraint that flow in a built pipeline must fall within min and max allowable flows for the given diameter
    model.max_min_flows = pyo.ConstraintList()

    for i, j in model.E:
        for t in model.T:  # Iterate through the neighbors of i and over each time period

            max_flow = sum(model.Q_pipe_max[d] * model.y_1[i, j, d, t] for d in model.D)

            flow = model.x_1[i, j, t]
            model.max_min_flows.add(-max_flow <= flow)
            model.max_min_flows.add(flow <= max_flow)

            model.max_min_flows.add(model.abs_flow_1[i, j, t] >= flow)
            model.max_min_flows.add(model.abs_flow_1[i, j, t] >= - flow)

    # (Cons 1a) If pipeline (i,j,d) exists/is built at time t. Then it exists for time t+1,t+2,..., T_1
    model.pipe_continuity = pyo.ConstraintList()
    for (i, j) in model.E:
        for d in model.D:
            for t in model.T:
                if t == 1:  # Check if this piece of infrastructure initially exists
                    y_ijd0 = (i, j, d) in model.existing_pipelines
                    model.pipe_continuity.add(
                        y_ijd0 <= model.y_1[i, j, d, t])  # Mark as existing in period 1,if already exists.
                else:
                    exists_in_prev_period = model.y_1[i, j, d, t - 1]
                    exists_in_curr_period = model.y_1[i, j, d, t]

                    model.pipe_continuity.add(exists_in_prev_period <= exists_in_curr_period)

    # Cons (2) The net-CO_2 flow in a given time period for a given node must be balanced.

    model.flow_conservation = pyo.ConstraintList()

    for i in model.N:
        for t in model.T:

            if not model.to_neighbor_dict[i] and not model.from_neighbor_dict[i]:  # If node has no edges, then we have a redundant constraint
                tmp_lhs = 0
            else:
                # Let flow < 0 => out of node, flow > 0 => into node
                tmp_lhs = - sum(model.x_1[i, j, t] for j in model.to_neighbor_dict[i]) + sum(
                    model.x_1[j, i, t] for j in model.from_neighbor_dict[i])
            if i in model.R:
                rhs = model.b_1[i, t]
            elif i in model.S:
                rhs = - model.a_1[i, t]
            else:
                rhs = 0

            model.flow_conservation.add(tmp_lhs == rhs)

    # Cons (3): We can only capture CO_2 from source if it is active and then it is capped by the max output in a time unit.

    model.source_output = pyo.ConstraintList()

    for i in model.S:
        for t in model.T:
            model.source_output.add(model.a_1[i, t] <= model.Q_source_rate[i] * model.s_1[i, t])
            # Cons (3a): Once we build a power-plant/retrofit it stays built
            if t == 1:  # Check if this source is already built and if so mark it as existing in period 1.
                s_i0 = i in model.existing_sources
                model.source_output.add(s_i0 <= model.s_1[i, t])
            else:
                model.source_output.add(model.s_1[i, t - 1] <= model.s_1[i, t])
    # Cons (4) : We can only store CO_2 at a sink if it is brought online and we can only store so much per time period, we can't exceed the cap

    model.sink_input = pyo.ConstraintList()
    for j in model.R:
        for t in model.T:
            model.sink_input.add(model.b_1[j, t] <= model.Q_res_rate[j] * model.r_1[j, t])
            # Cons (4a): Once we build a sink/retrofit on, it stays active.
            if t == 1:  # Check if this reservoir is already built and if so mark it as existing in period 1.
                r_j0 = j in model.existing_res
                model.sink_input.add(r_j0 <= model.r_1[j, t])
            else:
                model.sink_input.add(model.r_1[j, t - 1] <= model.r_1[j, t])

        # Add the constraint that we may not overfill the reservoir
        total_stored = sum(model.b_1[j, t] for t in model.T)
        model.sink_input.add(total_stored <= model.Q_res[j])

    # Cons(1.5): Picking out the source build times.
    model.source_build_times = pyo.ConstraintList()
    for t in model.T:
        for i in model.S:
            if t == 1:
                s_i0 = (i in model.existing_sources)
                rhs = model.s_1[i, t] - s_i0
            else:
                rhs = model.s_1[i, t] - model.s_1[i, t - 1]

            model.source_build_times.add(model.sigma_1[i, t] == rhs)

    # Cons(1.5): Picking out the sink build times.
    model.res_build_times = pyo.ConstraintList()
    for t in model.T:
        for j in model.R:
            if t == 1:
                r_i0 = (j in model.existing_res)
                rhs = model.r_1[j, t] - r_i0
            else:
                rhs = model.r_1[j, t] - model.r_1[j, t - 1]

            model.res_build_times.add(model.rho_1[j, t] == rhs)

    # (Fc) Constraint that determines the period in which the pipelines are built
    model.pipe_build_times = pyo.ConstraintList()
    for t in model.T:
        for i, j in model.E:
            for d in model.D:
                if t == 1:
                    y_ijd0 = (i, j, d) in model.existing_pipelines
                    rhs = model.y_1[i, j, d, t] - y_ijd0
                else:
                    rhs = model.y_1[i, j, d, t] - model.y_1[i, j, d, t - 1]

                model.pipe_build_times.add(model.gamma_1[i, j, d, t] == rhs)

    # ====================================================================================================================
    # ====================================== Objective Function  =========================================================
    # ====================================================================================================================

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++ First Stage Costs +++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    capture_costs_1 = sum((model.B_src[i, t] * model.sigma_1[i, t] + model.OM_src[i, t] * model.s_1[i, t] + model.V_src[
        i, t] * model.a_1[i, t]) * ((1 + discount_rate) ** (- period_length * t)) for t in model.T for i in model.S)
    # ----------  Pipeline costs ------------------------------
    pipeline_use_costs_1 = sum(
        model.V_pipe[i, j, t] * model.abs_flow_1[i, j, t] * ((1 + discount_rate) ** (- period_length * t)) for i, j in
        model.E for t in model.T)

    pipeline_build_costs_1 = sum(
        (model.B_pipe[i, j, d, t] * model.gamma_1[i, j, d, t] + model.OM_pipe[i, j, d, t] * model.y_1[i, j, d, t]) * ((
                1 + discount_rate) ** (- period_length * t)) for i, j in model.E for d in model.D for t in model.T)

    # -------- Storage Costs ----------------------------------

    storage_costs_1 = sum((model.rho_1[j, t] * model.B_res[j, t] + model.OM_res[j, t] * model.r_1[j, t] + model.V_res[
        j, t] * model.b_1[j, t]) * ((1 + discount_rate) ** (- period_length * t)) for t in model.T for j in model.R)

    # ------ Utilization Revenue ------------------------------

    util_rev_1 = sum(
        model.V_util[j] * model.b_1[j, t] * (1 + discount_rate) ** (- period_length * t) for j in model.U for t in
        model.T)

    # ------ q45 Tax Revenue -----------------------------------

    storage_tax_rev_1 = sum(model.storage_incentive_1[t] *
        model.b_1[j, t] * ((1 + discount_rate) ** (- period_length * t)) for j in model.R.difference(model.U) for t in
        model.T)
    util_tax_rev_1 = sum(model.util_incentive_1[t] *
        model.b_1[j, t] * ((1 + discount_rate) ** (- period_length * t)) for j in model.U for t in model.T)

    tax_rev_1 = util_tax_rev_1 + storage_tax_rev_1
    # ---------- Final objective function Expression -----------

    first_stage_costs = (capture_costs_1 + pipeline_use_costs_1 + pipeline_build_costs_1 + storage_costs_1) - util_rev_1 - tax_rev_1

    # ---------- Final objective function Expression -----------
    obj_expression = first_stage_costs

    model.OBJ = pyo.Objective(expr=obj_expression)

    print('running optimization model!')
    opt = pyo.SolverFactory('cplex',
                            executable='/Users/connor/Applications/CPLEX_Studio1210/cplex/bin/x86-64_osx/cplex')
    opt.options['mip_tolerances_mipgap'] = 0.01
    results = opt.solve(model)

    # =================================================================================================================
    # ============================= Write our Solutions to File =======================================================
    # =================================================================================================================


    print(results)
    print(f'Objective value = {pyo.value(model.OBJ)}')
    print(
        f'Total CO_2 captured: {int(sum(pyo.value(model.b_1[j, t]) for j in model.R for t in model.T)) / 10 ** 6} MT CO_2.')

    print(f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n'
          f'++++++++++++++++++++++++++++++++ Objective Function Price Breakdown ++++++++++++++++++++++++++++\n'
          f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'The tax credit values for the "Second Stage" were {storage_incentive, utilization_incentive}.')
    capture_costs = int(sum((model.B_src[i, t] * pyo.value(model.sigma_1[i, t]) +
                            model.OM_src[i, t] * pyo.value(model.s_1[i, t]) + model.V_src[i, t] * pyo.value(model.a_1[i, t])) * ((1 + discount_rate) ** (- period_length * t)) for t in model.T for i in model.S))



    print(f'Total Capture costs: {capture_costs}')

    pipeline_use_costs = int(
        sum(model.V_pipe[i, j, t] * pyo.value(model.abs_flow_1[i, j, t] * ((1 + discount_rate) ** (- period_length * t))) for i, j in model.E for t in model.T))

    pipeline_build_costs = int(sum((model.B_pipe[i, j, d, t] * pyo.value(model.gamma_1[i, j, d, t]) +
                                   model.OM_pipe[i, j, d, t] * pyo.value(model.y_1[i, j, d, t]))*((1 + discount_rate) ** (- period_length * t)) for i, j in model.E for
                                   d in model.D for t in model.T))

    transportation_costs = pipeline_build_costs + pipeline_use_costs
    print(f'Total pipeline costs: {transportation_costs}')

    storage_costs = int(sum((model.B_res[j, t] * pyo.value(model.rho_1[j, t]) +
                            model.OM_res[j, t] * pyo.value(model.r_1[j, t]) +
                            model.V_res[j, t] * pyo.value(model.b_1[j, t])) * ((1 + discount_rate) ** (- period_length * t)) for t in model.T for j in
                            model.R))
    print(f'Total Storage costs: {storage_costs}')
    # ADD FIRST STAGE COSTS

    capture_costs_period_1 = int(sum((model.B_src[i, t] * pyo.value(model.sigma_1[i, t]) +
                             model.OM_src[i, t] * pyo.value(model.s_1[i, t]) + model.V_src[i, t] * pyo.value(
                model.a_1[i, t])) * ((1 + discount_rate) ** (- period_length * t)) for t in model.T_1 for i in model.S))

    print(f'Total Capture costs: {capture_costs}')

    pipeline_use_costs_period_1 = int(
        sum(model.V_pipe[i, j, t] * pyo.value(
            model.abs_flow_1[i, j, t] * ((1 + discount_rate) ** (- period_length * t))) for i, j in model.E for t in
            model.T_1))

    pipeline_build_costs_period_1 = int(sum((model.B_pipe[i, j, d, t] * pyo.value(model.gamma_1[i, j, d, t]) +
                                    model.OM_pipe[i, j, d, t] * pyo.value(model.y_1[i, j, d, t])) * (
                                               (1 + discount_rate) ** (- period_length * t)) for i, j in model.E for
                                   d in model.D for t in model.T_1))

    transportation_costs_period_1 = pipeline_build_costs_period_1 + pipeline_use_costs_period_1


    storage_costs_period_1 = int(sum((model.B_res[j, t] * pyo.value(model.rho_1[j, t]) +
                             model.OM_res[j, t] * pyo.value(model.r_1[j, t]) +
                             model.V_res[j, t] * pyo.value(model.b_1[j, t])) * (
                                        (1 + discount_rate) ** (- period_length * t)) for t in model.T_1 for j in
                            model.R))
    total_cost_1 = capture_costs_period_1 + storage_costs_period_1 + transportation_costs_period_1
    print(f'First stage infastructure costs {total_cost_1}')

    util_rev = int(sum(model.V_util[j] * pyo.value(model.b_1[j, t]) * ((1 + discount_rate) ** (- period_length * t)) for j in model.U for t in model.T))
    storage_tax_rev = sum(pyo.value(model.storage_incentive_1[t]) *
                            pyo.value(model.b_1[j, t]) * ((1 + discount_rate) ** (- period_length * t)) for j in
                            model.R.difference(model.U) for t in model.T)

    util_tax_rev = sum(pyo.value(model.util_incentive_1[t]) *
                         pyo.value(model.b_1[j, t]) * ((1 + discount_rate) ** (- period_length * t)) for j in model.U for t in
                         model.T)
    tax_rev = storage_tax_rev + util_tax_rev
    total_cost = capture_costs + storage_costs + transportation_costs
    profit = - (capture_costs + storage_costs + pipeline_use_costs + pipeline_build_costs - util_rev - tax_rev)
    print(
        f'Total Profit : {format_float_scientific(profit, precision=5)}\n')
    print(f'Revenue gained from utilization: {util_rev}')
    print(f'Revenue gained through Tax Incentive: {tax_rev}\n')
    # ADD TOTAL FIRST STAGE INVESTMENT

    if total_cost != 0:
        labels = 'Capture ', 'Storage', 'Transportation'

        sizes = [capture_costs / total_cost, storage_costs / total_cost, transportation_costs / total_cost]
        print(capture_costs, storage_costs, transportation_costs)

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        plt.show()

    print(f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n'
          f'+++++++++++++++++++++++++ Pipeline Construction Breakdown ++++++++++++++++++++++++++++++++++++++\n'
          f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for t in model.T:
        print()
        for d in model.D:
            for (i, j) in model.E:
                if pyo.value(model.gamma_1[i, j, d, t]) > 0:
                    print(f'In time period {t} the pipeline between {i} <-> {j} of diameter {d} is built.')
    print()
    print(f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n'
          f'+++++++++++++++++++++++++ CO_2 Movement Breakdown ++++++++++++++++++++++++++++++++++++++++++++++\n'
          f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n ')
    for t in model.T:
        print(f'-------------------------  In time period: t = {t} ----------------------------- ')
        for i, j in model.E:
            if int(pyo.value(model.x_1[i, j, t])) > 0:
                print(f'We send {int(pyo.value(model.x_1[i, j, t]) / 10 ** 6)} units {i} --> {j}.')
            elif int(pyo.value(model.x_1[i, j, t])) < 0:
                print(f'We send {- int(pyo.value(model.x_1[i, j, t]) / 10 ** 6)} units {j} --> {i}.')

    print()
    print(f'------------------ Source Capture Rates ----------------------------\n')
    for t in model.T:
        for i in model.S:
            if pyo.value(model.a_1[i, t]) > 0:
                print(f'Source {i} produces {int(pyo.value(model.a_1[i, t]) / 10 ** 6)} units in period {t}')

    print()
    print(f'------------------ Geological Storage Capture Rates ------------------------------\n')
    for t in model.T:
        for j in model.R:
            if j not in model.U and pyo.value(model.b_1[j, t]) != 0:
                print(f'Sink {j} captures {int(pyo.value(model.b_1[j, t]) / 10 ** 6)} units in period {t}')
    print()
    print(f'------------------Utilization Capture Rates ------------------------------\n')
    for t in model.T:
        for j in model.U:
            if pyo.value(model.b_1[j, t]) != 0:
                print(
                    f'Utilization site {j} captures {int(pyo.value(model.b_1[j, t]) / 10 ** 6)} MT of CO2 in period {t}')
    print()
    print(f'------------------Total Capture Information ------------------------------\n')
    for j in model.R:
        total_captured_j = sum(pyo.value(model.b_1[j, t]) for t in model.T)
        if total_captured_j > 1:
            print(
                f'Sink {j} captures {total_captured_j / 10 ** 6} MT CO_2. Which is {100 * (total_captured_j / model.Q_res[j]):.3f}% of its storage capacity')

    print(f'--------------- Carbon Goals--------------------------\n')
    total_captured = int(sum(pyo.value(model.b_1[j, t]) for j in model.R for t in model.T))
    print(
        f'The infrastructure captured {int(total_captured / 10 ** 6)} Megatons of CO_2 over the {num_time_periods} first-stage periods.')

    print(f'--- The total runtime was  {time.time() - start_time:.2f} seconds ---" ')

    return profit, total_captured, capture_costs, transportation_costs, storage_costs, util_rev, tax_rev


if __name__ == '__main__':
    import sys

    storage_incentives = [85]
    util_incentives = [60]
    incentives = zip(storage_incentives, util_incentives)
    text_results = 'Cases/Case6/Results/Debugging/Deterministic_max_profit_for_scaling.txt'
    df = pd.DataFrame()
    sys.stdout = open(text_results, 'a')
    num_cases = len(storage_incentives)

    for i, (storage_incentive, util_incentive) in enumerate(incentives):
        results = run_exp(storage_incentive, util_incentive, base_storage=85, base_util=60)

        data = {'Incentive Values': i, 'Profit': results[0],
                'Total CO2 Captured': results[1],
                'Capture Costs': results[2], 'Transportation Costs': results[3], 'Storage_costs': results[4],
                'Utilization Rev': results[5], '45Q Revenue': results[6]}

        test = pd.DataFrame(data, index=[i])
        df = pd.concat([df, test])
        update_slack(f'Case {i} complete, {num_cases - 1 - i} left!')
    print(df)
    sys.stdout.close()
    # with pd.ExcelWriter('Cases/Case6/Results/Debugging/Deterministic_.xlsx', mode='a', if_sheet_exists="replace") as writer:
    #     df.to_excel(writer, sheet_name='Deterministic')
    storage_incentives = [0]
    util_incentives = [0]
    update_slack('I finished running!')

    # run_exp(0, 0, base_storage=85, base_util=60, second_stage_start=2)