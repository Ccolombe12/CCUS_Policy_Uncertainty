# We will implement a two-stage, risk-neutral, stochastic program for the CCUS Infrastructure problem.
import pyomo.environ as pyo
from helper_functions import *
import csv
from numpy import format_float_scientific
import time

import pandas as pd
import matplotlib.pyplot as plt


# This model assumes you have the following files in your working directory:
# ++++++++++++++++++++
#     GeneralData.csv
#     Sources.csv
#     Sinks.csv
#     Pipeline.csv
# ++++++++++++++++++++
# See the Readme file for a description of the parameters in each file.

# The following code assumes the following directory

# =====================================================================================================================
# ================================  Load in General Model Data  =======================================================
# =====================================================================================================================

def run_experiment(eta, case_num, max_profit):
    start_time = time.time()
    file = open(f"Cases/Case{case_num}/GeneralData.csv", encoding='latin-1')
    print('Reading in data')
    csv_reader = csv.reader(file)
    next(csv_reader)  # skip the header
    general_parameters = [int(x) for x in next(csv_reader) if x]
    num_D, num_pipelines, num_nodes, num_sources, num_sinks, num_time_periods, num_periods_in_first_stage,\
        stage_one_storage_tax_credit, stage_one_util_tax_credit, num_scenarios = general_parameters

    next(csv_reader)  # Skip the  secondary header.
    discount_and_years = [float(x) for x in next(csv_reader) if x]
    discount_rate = discount_and_years[0]
    period_length = int(discount_and_years[1])

    next(csv_reader)
    tax_credit_storage_realizations = [float(x) for x in next(csv_reader) if x]
    tax_credit_util_realizations = [float(x) for x in next(csv_reader) if x]
    next(csv_reader)  # Skip the next header.
    scenario_probabilities = [float(x) for x in next(csv_reader) if x]

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
    model.existing_sources = set()  # The indices of sources that exist at T = 0 (initial conditions)
    model.source_loc = dict()

    model.OM_res = dict()  # dict of fixed costs for O/M of reservoirs. F_res[j,t] is fixed cost for operating reservoir j in time t.
    model.B_res = dict()  # dict of fixed costs for constructing reservoirs. B_res[j,t] is fixed cost for building reservoir j in time t.
    model.V_res = dict()  # Variable cost of capturing CO2 at reservoir j ($/ton C0_2). V_res[j,t] will be the variable cost per unit C0_2 in reservoir j.
    model.Q_res = dict()  # Capacity of reservoir j (MtCO2)
    model.Q_res_rate = dict()  # max injection into reservoir j in time t.
    model.existing_res = set()  # The indices of reservoirs that exist at T = 0 (initial conditions)

    model.OM_pipe = dict()  # F_pipe[i,j,d,t] is fixed cost for maintaining the pipeline between node i and j with diameter d in time period t.
    model.B_pipe = dict()  # F_pipe[i,j,d,t] is fixed cost for building the pipeline between node i and j with diameter d in time period t.
    model.V_pipe = dict()  # Cost to transport one tonne of CO2 on pipeline (i,j) with diameter d (MtCO2/yr) during period t.
    model.Q_pipe_max = dict()  # Maximum capacity of pipeline (i,j) with diameter d (MtCO2/yr).
    model.Q_pipe_min = dict()  # Minimum capacity of pipeline (i,j) with diameter d (MtCO2/yr).
    model.existing_pipelines = set()  # The set of existing pipelines at (T = 0)

    model.C_tax = dict()  # For each scenario s = 1, ..., S, have what the tax credit is
    model.S_prob = dict()  # For each scenario s = 1, ..., S, have what the probability of that scenario is
    model.Scenarios = pyo.RangeSet(num_scenarios)
    model.V_util = dict()  # dict of the utilization returns $/unit captured at utilization site k in period t.

    # Helper function for debugging
    model.second_stage_profit = dict()
    model.second_stage_revenue = dict()
    # =====================================================================================================================
    # ======================= Read in data from files =====================================================================
    # =====================================================================================================================

    # +++++++++++++++++++++ Read in Source data ++++++++++++++++++++++++++++++++++++++
    file = open(f'Cases/Case{case_num}/Sources.csv', encoding='latin-1')
    csv_reader = csv.reader(file)
    source_set = set()
    next(csv_reader)  # Skip Header
    for _ in range(num_sources):

        values = next(csv_reader)
        cur_node = int(values[0])
        source_set.add(cur_node)

        values = [float(x) for x in values[1:-1]]  # Cut out ID # and Type. Just the capacity and cost components
        for t in range(1, num_time_periods + 1):
            model.Q_source_rate[cur_node], model.B_src[cur_node], model.OM_src[cur_node], model.V_src[
                cur_node], = values
            # Update the parameters based on if the change with the length of each period
            model.Q_source_rate[cur_node] *= period_length  # Each source can now capture (# yrs in period) * as much
            model.OM_src[cur_node] *= period_length  # O/M costs increase proportionally

    file.close()

    # +++++++++++++++++++++ Read in Sink data ++++++++++++++++++++++++++++++++++++++
    file = open(f'Cases/Case{case_num}/Sinks.csv', encoding='latin-1')
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
            model.Q_res[cur_node], model.Q_res_rate[cur_node], model.B_res[cur_node], model.OM_res[cur_node], \
                model.V_res[cur_node], is_util, model.V_util[cur_node] = values
            # Update values that depend on length of period
            model.Q_res_rate[cur_node] *= period_length
            model.OM_res[cur_node] *= period_length

        if is_util:  # If the node is a utility node, add it to the set of utility nodes.
            util_set.add(cur_node)

    file.close()




    # +++++++++++++++++++++ Read in Pipeline data ++++++++++++++++++++++++++++++++++++++

    file = open(f"Cases/Case{case_num}/Arcs.csv", encoding='latin-1')
    csv_reader = csv.reader(file)
    next(csv_reader)  # skip the initial header

    # Read in the diameter capacity data
    diameter_data = next(csv_reader)
    edge_set = set()
    transition_node_set = set()
    line = [float(x) for x in diameter_data if x]

    for d, val in enumerate(line):
        model.Q_pipe_max[d + 1] = val * period_length

    next(csv_reader)  # skip secondary header
    # Get cost multiplication factors
    pipe_cost_factors = dict()
    factors = next(csv_reader)
    factors = [float(x) for x in factors if x]

    for d in range(
            num_D):  # Goofy fix to offset the index to start at 1. Probably should make this cleaner. todo: Clean this up
        pipe_cost_factors[d + 1] = factors[d]

    next(csv_reader)
    for _ in range(num_pipelines):

        values = next(csv_reader)  # Get the next pipeline
        _, _, i, j, _, cap_cost, OM_cost, Var_cost = [int(float(x)) for x in values if x]

        edge_set.add((i, j))  # as we loop over edges, if a node on edge is not a source/sink, it is transition node.
        if i not in source_set and i not in sink_set:
            transition_node_set.add(i)

        if j not in source_set and j not in sink_set:
            transition_node_set.add(j)

        for d in range(num_D):
            model.B_pipe[i, j, d + 1] = cap_cost * pipe_cost_factors[d + 1]
            model.OM_pipe[i, j, d + 1] = OM_cost * period_length

        model.V_pipe[i, j] = Var_cost  # todo: Should this vary with d? Probably.

    file.close()

    # Update the stage one tax incentive. Assume it is known.
    model.storage_incentive_1 = stage_one_storage_tax_credit
    model.util_incentive_1 = stage_one_util_tax_credit
    model.S_prob = {s: scenario_probabilities[s - 1] for s in model.Scenarios}
    model.C_tax_storage = {s: tax_credit_storage_realizations[s - 1] for s in model.Scenarios}
    model.C_tax_util = {s: tax_credit_util_realizations[s - 1] for s in model.Scenarios}
    update_slack(model.S_prob)
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
    model.T_1 = pyo.RangeSet(num_periods_in_first_stage)  # Time periods for the first stage (starting at 1)
    model.T_2 = pyo.RangeSet(num_periods_in_first_stage + 1, num_time_periods)  # Second stage time periods
    model.Scenarios = pyo.RangeSet(num_scenarios)

    # Now we preprocess to generate sets of nodes that point to .
    model.to_neighbor_dict = dict()
    model.from_neighbor_dict = dict()
    for i in model.N:
        model.from_neighbor_dict[i] = get_from_neighbors(i, model.E)
        model.to_neighbor_dict[i] = get_to_neighbors(i, model.E)
    model.second_stage_profit = dict()

    # ====================================================================================================================
    # ===================================First-Stage Decision Variables ==================================================
    # ====================================================================================================================

    # +++++++++++++++ First stage variables. They will all have the "foo_1" subscript +++++++++++++++++++++++++++
    model.x_1 = pyo.Var(model.E, model.T_1, domain=pyo.Reals)  # Amount of flow from i -> j in period t
    model.y_1 = pyo.Var(model.E, model.D, model.T_1, domain=pyo.Binary)  # if pipe (i,j) with diam. d exists in time t
    model.z_1 = pyo.Var(model.E, model.T_1, domain=pyo.PositiveReals)

    model.a_1 = pyo.Var(model.S, model.T_1, domain=pyo.PositiveReals)  # Amt. CO2 captured at source i during period t
    model.b_1 = pyo.Var(model.R, model.T_1, domain=pyo.PositiveReals)  # Amt. of CO2 put in reservoir j in period t

    model.s_1 = pyo.Var(model.S, model.T_1, domain=pyo.Binary)  # If source at node i is open during period t
    model.r_1 = pyo.Var(model.R, model.T_1, domain=pyo.Binary)  # If reservoir at node j is open during period t

    model.sigma_1 = pyo.Var(model.S, model.T_1, domain=pyo.Binary)  # If source i is built during period t
    model.rho_1 = pyo.Var(model.R, model.T_1, domain=pyo.Binary)  # If res. j is built during period t
    model.gamma_1 = pyo.Var(model.E, model.D, model.T_1, domain=pyo.Binary)  # If pipe(i, j,d) is built during period t

    # Add some 'helper variables to reduce the gross amount of loops'
    model.first_stage_capture_costs = pyo.Var(domain=pyo.Reals)
    model.first_stage_storage_costs = pyo.Var(domain=pyo.Reals)
    model.first_stage_pipeline_costs = pyo.Var(domain=pyo.Reals)
    model.first_stage_revenue = pyo.Var(domain=pyo.Reals)
    model.first_stage_profit = pyo.Var(domain=pyo.Reals)
    model.normalized_scenario_profit = pyo.Var(model.Scenarios, domain=pyo.Reals)
    # ====================================================================================================================
    # =================================== Second-Stage Decision Variables ================================================
    # ====================================================================================================================

    model.x_2 = pyo.Var(model.E, model.T_2, model.Scenarios, domain=pyo.Reals)  # Amount of flow from i -> j in period t
    model.y_2 = pyo.Var(model.E, model.D, model.T_2, model.Scenarios, domain=pyo.Binary)  # if pipe (i,j,d) exists in t
    model.z_2 = pyo.Var(model.E, model.T_2, model.Scenarios, domain=pyo.PositiveReals)
    model.a_2 = pyo.Var(model.S, model.T_2, model.Scenarios, domain=pyo.PositiveReals)  # Amt of CO2 at source i in t
    model.b_2 = pyo.Var(model.R, model.T_2, model.Scenarios, domain=pyo.PositiveReals)  # Amt. of CO2 into  res j in t

    model.s_2 = pyo.Var(model.S, model.T_2, model.Scenarios, domain=pyo.Binary)  # if source i is open in per. t
    model.r_2 = pyo.Var(model.R, model.T_2, model.Scenarios, domain=pyo.Binary)  # if res j is open in per. t

    model.sigma_2 = pyo.Var(model.S, model.T_2, model.Scenarios, domain=pyo.Binary)  # if source i is built in period t in s
    model.rho_2 = pyo.Var(model.R, model.T_2, model.Scenarios, domain=pyo.Binary)  # if res j is built in period t under s
    model.gamma_2 = pyo.Var(model.E, model.D, model.T_2, model.Scenarios, domain=pyo.Binary)  # The indicator variable for if pipeline

    model.scenario_profit = pyo.Var(model.Scenarios, domain=pyo.Reals)
    model.omega = pyo.Var(model.Scenarios, domain=pyo.Reals)
    # ====================================================================================================================
    # ====================================== Model Constraints  =========================================================
    # ====================================================================================================================

    # +++++++++++++++++ First - Stage Constraints +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # (Cons 1) Constraint that flow in existing pipeline must fall within max allowable flows for the given diameter
    model.max_min_flows = pyo.ConstraintList()

    for i, j in model.E:
        for t in model.T_1:  # Iterate through the neighbors of i and over each time period

            max_flow = sum(model.Q_pipe_max[d] * model.y_1[i, j, d, t] for d in model.D)

            flow = model.x_1[i, j, t]
            model.max_min_flows.add(-max_flow <= flow)
            model.max_min_flows.add(flow <= max_flow)

            model.max_min_flows.add(model.z_1[i, j, t] >= flow)
            model.max_min_flows.add(model.z_1[i, j, t] >= - flow)

    # (Cons 1a) If pipeline (i,j,d) exists/is built at time t. Then it exists for time t+1,t+2,..., T_1
    model.pipe_continuity = pyo.ConstraintList()
    for i, j in model.E:
        for d in model.D:
            for t in model.T_1:
                if t == 1:  # Check if this piece of infrastructure initially exists
                    y_ijd0 = int((i, j, d) in model.existing_pipelines)
                    model.pipe_continuity.add(y_ijd0 <= model.y_1[i, j, d, t])  # Mark as existing in period 1, if already exists.
                else:
                    model.pipe_continuity.add(model.y_1[i, j, d, t - 1] <= model.y_1[i, j, d, t])

    # Cons (2) The net-CO_2 flow in a given time period for a given node must be balanced.

    model.flow_conservation = pyo.ConstraintList()
    for i in model.N:
        for t in model.T_1:
            # If node has no edges, we have a redundant constraint
            if (not model.to_neighbor_dict[i]) and (not model.from_neighbor_dict[i]):  # Has no neighbors.
                tmp_lhs = 0
            else:
                # Let flow < 0 => out of node, flow > 0 => into node
                tmp_lhs = - sum(model.x_1[i, j, t] for j in model.to_neighbor_dict[i]) \
                          + sum(model.x_1[j, i, t] for j in model.from_neighbor_dict[i])
            if i in model.R:
                rhs = model.b_1[i, t]
            elif i in model.S:
                rhs = - model.a_1[i, t]
            else:
                rhs = 0

            model.flow_conservation.add(tmp_lhs == rhs)

    # Cons (3): We can only capture CO_2 from source if it is active, and then it is capped by the max output in a time unit.

    model.source_output = pyo.ConstraintList()

    for i in model.S:
        for t in model.T_1:
            model.source_output.add(model.a_1[i, t] <= model.Q_source_rate[i] * model.s_1[i, t])
            # Cons (3a): Once we build a power-plant/retrofit it stays built
            if t == 1:  # Check if this source is already built and if so mark it as existing in period 1.
                s_i0 = i in model.existing_sources
                model.source_output.add(s_i0 <= model.s_1[i, t])
            else:
                model.source_output.add(model.s_1[i, t - 1] <= model.s_1[i, t])  # Built sources stay built.
    # Cons (4) : We can only store CO_2 at a sink if it is brought online. We can only store so much per time period
    # ,and we can't exceed the cap.

    model.sink_input = pyo.ConstraintList()
    for j in model.R:
        for t in model.T_1:
            model.sink_input.add(model.b_1[j, t] <= model.Q_res_rate[j] * model.r_1[j, t])
            # Cons (4a): Once we build a sink/retrofit on, it stays active.
            if t == 1:  # Check if this reservoir is already built and if so mark it as existing in period 1.
                r_j0 = j in model.existing_res
                model.sink_input.add(r_j0 <= model.r_1[j, t])
            else:
                model.sink_input.add(model.r_1[j, t - 1] <= model.r_1[j, t])

        # Add the constraint that we may not overfill the reservoir
        total_stored = sum(model.b_1[j, t] for t in model.T_1)
        model.sink_input.add(total_stored <= model.Q_res[j])

    # Cons(1.5): Picking out the source build times.
    model.source_build_times = pyo.ConstraintList()
    for t in model.T_1:
        for i in model.S:
            if t == 1:
                s_i0 = int(i in model.existing_sources)  # 1 if true, 0 if false
                rhs = model.s_1[i, t] - s_i0
            else:
                rhs = model.s_1[i, t] - model.s_1[i, t - 1]

            model.source_build_times.add(model.sigma_1[i, t] == rhs)

    # Cons(1.5): Picking out the sink build times.
    model.res_build_times = pyo.ConstraintList()
    for t in model.T_1:
        for j in model.R:
            if t == 1:
                r_i0 = int(j in model.existing_res)
                rhs = model.r_1[j, t] - r_i0
            else:
                rhs = model.r_1[j, t] - model.r_1[j, t - 1]

            model.res_build_times.add(model.rho_1[j, t] == rhs)

    # (Fc) Constraint that determines the period in which the pipelines are built
    model.pipe_build_times = pyo.ConstraintList()
    for t in model.T_1:
        for i, j in model.E:
            for d in model.D:
                if t == 1:
                    y_ijd0 = int((i, j, d) in model.existing_pipelines)
                    rhs = model.y_1[i, j, d, t] - y_ijd0
                else:
                    rhs = model.y_1[i, j, d, t] - model.y_1[i, j, d, t - 1]

                model.pipe_build_times.add(model.gamma_1[i, j, d, t] == rhs)
    # ====================================================================================================================
    # ================================== Second-Stage Model Constraints  =================================================
    # ====================================================================================================================
    print('starting two stage model construction!')
    # (Cons 34 and 35) Constraint that flow in a built pipeline must fall within min and max allowable flows for the given diameter

    for i, j in model.E:
        for t in model.T_2:
            for s in model.Scenarios:
                max_flow = sum(model.Q_pipe_max[d] * model.y_2[i, j, d, t, s] for d in model.D)

                flow = model.x_2[i, j, t, s]  # Here we add constraints (34) from the paper
                model.max_min_flows.add(- max_flow <= flow)
                model.max_min_flows.add(flow <= max_flow)

                model.max_min_flows.add(
                    model.z_2[i, j, t, s] >= flow)  # Here we add constraints (35) from the paper
                model.max_min_flows.add(model.z_2[i, j, t, s] >= - flow)

    # (Cons 47 and 41) If pipeline (i,j,d) exists/is built at time t. Then it exists for time t+1,t+2,..., T_2

    for (i, j) in model.E:
        for d in model.D:
            for t in model.T_2:
                for s in model.Scenarios:
                    # Here we add constraint (47) or constraint (41) depending on the value of t.
                    if t == num_periods_in_first_stage + 1:  # Check if this piece of infrastructure already exists.
                        model.pipe_continuity.add(model.y_2[i, j, d, t, s] >= model.y_1[i, j, d, t - 1])  # (47)
                        # Mark as existing in period T_1 + 1 ,if already exists.
                    else:
                        model.pipe_continuity.add(model.y_2[i, j, d, t, s] >= model.y_2[i, j, d, t - 1, s])  # (41)

    # Cons (36) The net-CO_2 flow in a given time period for a given node must be balanced.
    for i in model.N:
        for t in model.T_2:
            for s in model.Scenarios:
                # If this node is isolated, we have no net flow.
                if not model.to_neighbor_dict[i] and not model.from_neighbor_dict[i]:
                    tmp_lhs = 0
                # Otherwise we have (flow in - flow out) = net flow
                else:
                    tmp_lhs = sum(model.x_2[j, i, t, s] for j in model.from_neighbor_dict[i]) - \
                              sum(model.x_2[i, j, t, s] for j in model.to_neighbor_dict[i])

                if i in model.R:
                    tmp_rhs = model.b_2[i, t, s]  # net flow can only be positive if node is a sink
                elif i in model.S:
                    tmp_rhs = - model.a_2[i, t, s]  # net flow can only be negative if node is a source
                else:
                    tmp_rhs = 0  # Junction nodes must have even net flow

                model.flow_conservation.add(tmp_lhs == tmp_rhs)

    # Cons (37, 48, 42): We can only capture CO_2 from source if it is active. If so, then it is capped by the max output in a time unit.
    for i in model.S:
        for t in model.T_2:
            for s in model.Scenarios:
                model.source_output.add(model.a_2[i, t, s] <= model.Q_source_rate[i] * model.s_2[i, t, s])

                # Cons (48 and 42): Once we build a power-plant/retrofit it stays built
                if t == num_periods_in_first_stage + 1:  # Check if this source is already built and if so mark it as existing in period 1.
                    model.source_output.add(model.s_2[i, t, s] >= model.s_1[i, t - 1])
                else:
                    model.source_output.add(model.s_2[i, t, s] >= model.s_2[i, t - 1, s])

    # Cons (38) : We can only store CO_2 at a sink if it is brought online. If so, we can only store so much per time period, and we can't exceed the cap.
    for j in model.R:
        for t in model.T_2:
            for s in model.Scenarios:

                model.sink_input.add(model.b_2[j, t, s] <= model.Q_res_rate[j] * model.r_2[j, t, s])  # Cons (38)

                # Cons (49): # Check if this source is already built and if so mark it as existing in period 1.
                if t == num_periods_in_first_stage + 1:
                    model.sink_input.add(model.r_2[j, t, s] >= model.r_1[j, t - 1])
                # Cons (43):  Once we build a sink/retrofit on, it stays active.
                else:
                    model.sink_input.add(model.r_2[j, t, s] >= model.r_2[j, t - 1, s])

    # Cons (39): Can't exceed remaining storage capacity
    for j in model.R:
        total_stored_in_first_stage = sum(model.b_1[j, t] for t in model.T_1)

        for s in model.Scenarios:
            total_stored_in_second_stage = sum(model.b_2[j, t, s] for t in model.T_2)
            model.sink_input.add(total_stored_in_second_stage + total_stored_in_first_stage <= model.Q_res[j])

    # Cons(44): Picking out the source build times for second stage.

    for t in model.T_2:
        for i in model.S:
            for s in model.Scenarios:

                if t == num_periods_in_first_stage + 1:
                    rhs = model.s_2[i, t, s] - model.s_1[i, t - 1]
                else:
                    rhs = model.s_2[i, t, s] - model.s_2[i, t - 1, s]

                model.source_build_times.add(model.sigma_2[i, t, s] == rhs)

    # Cons(45): Picking out the sink build times.

    for t in model.T_2:
        for j in model.R:
            for s in model.Scenarios:

                if t == num_periods_in_first_stage + 1:
                    rhs = model.r_2[j, t, s] - model.r_1[j, t - 1]
                else:
                    rhs = model.r_2[j, t, s] - model.r_2[j, t - 1, s]

                model.res_build_times.add(model.rho_2[j, t, s] == rhs)

    # Cons(46): Picking out the pipeline build times.
    for t in model.T_2:
        for i, j in model.E:
            for d in model.D:
                for s in model.Scenarios:

                    if t == num_periods_in_first_stage + 1:
                        rhs = model.y_2[i, j, d, t, s] - model.y_1[i, j, d, t - 1]
                    else:
                        rhs = model.y_2[i, j, d, t, s] - model.y_2[i, j, d, t - 1, s]

                    model.pipe_build_times.add(model.gamma_2[i, j, d, t, s] == rhs)

    # ====================================================================================================================
    # ====================================== Objective Function  =========================================================
    # ====================================================================================================================

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++ First Stage Costs +++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    model.obj_constraints = pyo.ConstraintList()

    model.obj_constraints.add(sum((model.B_src[i] * model.sigma_1[i, t]
                                   + model.OM_src[i] * (model.a_1[i, t] / model.Q_source_rate[i]) + model.V_src[i] * model.a_1[i, t]) for t in model.T_1 for i in model.S) == model.first_stage_capture_costs)
    # ----------  Pipeline costs ------------------------------

    pipeline_use_costs_1 = sum(
        model.V_pipe[i, j] * model.z_1[i, j, t] for i, j in model.E for t in model.T_1)

    pipeline_build_costs_1 = sum(
        model.B_pipe[i, j, d] * model.gamma_1[i, j, d, t] + model.OM_pipe[i, j, d] * (model.z_1[i, j, t] / model.Q_pipe_max[d])
        for i, j in model.E for d in model.D for t in model.T_1)

    model.obj_constraints.add(model.first_stage_pipeline_costs == pipeline_build_costs_1 + pipeline_use_costs_1)

    # -------- Storage Costs ----------------------------------
    model.obj_constraints.add(model.first_stage_storage_costs == sum(model.rho_1[j, t] * model.B_res[j] +
                                                                     model.OM_res[j] * (model.b_1[j, t] / model.Q_res_rate[j]) + model.V_res[j] * model.b_1[j, t] for t in model.T_1 for j in model.R))
    # ------ Utilization Revenue ------------------------------

    util_rev_1 = sum(model.V_util[j] * model.b_1[j, t] for j in model.U for t in model.T_1)

    # ------ 45-Q Tax Revenue -----------------------------------

    storage_tax_rev_1 = model.storage_incentive_1 * sum(
        model.b_1[j, t] for j in model.R.difference(model.U) for t in model.T_1)

    util_tax_rev_1 = model.util_incentive_1 * sum(model.b_1[j, t] for j in model.U for t in model.T_1)

    tax_rev_1 = util_tax_rev_1 + storage_tax_rev_1

    model.obj_constraints.add(model.first_stage_revenue == util_rev_1 + tax_rev_1)
    # ---------- Define the total profit from our first stage operations -----------

    model.obj_constraints.add(model.first_stage_profit == model.first_stage_revenue - (model.first_stage_storage_costs + model.first_stage_pipeline_costs + model.first_stage_capture_costs))

    # Iteratively build the term that will be appended to the end of the first stage profit
    for s in model.Scenarios:
        #  ************** CALCULATE TOTAL COSTS *************
        # ----------  Capture costs --------------
        capture_costs_2s = sum((model.B_src[i] * model.sigma_2[i, t, s] + model.OM_src[i] * (model.a_2[i, t, s] / model.Q_source_rate[i]) +
                                model.V_src[i] * model.a_2[i, t, s]) for t in model.T_2 for i in model.S)
        # ---------- Pipeline costs --------------

        pipeline_use_costs_2s = sum(
            model.V_pipe[i, j] * model.z_2[i, j, t, s] for i, j in model.E for t in model.T_2)

        pipeline_build_costs_2s = sum(model.B_pipe[i, j, d] * model.gamma_2[i, j, d, t, s] +
                                      model.OM_pipe[i, j, d] * (model.z_2[i, j, t, s] / model.Q_pipe_max[d])
                                      for i, j in model.E for d in model.D for t in model.T_2)

        pipeline_costs_2s = pipeline_build_costs_2s + pipeline_use_costs_2s
        # -------- Storage Costs ------------------

        storage_costs_2s = sum(model.B_res[j] * model.rho_2[j, t, s] + model.OM_res[j] * (model.b_2[j, t, s] / model.Q_res_rate[j]) +
                               model.V_res[j] * model.b_2[j, t, s] for t in model.T_2 for j in model.R)
        #  ************** CALCULATE TOTAL REVENUE *************
        # ------ Utilization Revenue -----------------

        util_rev_2s = sum(model.V_util[j] * model.b_2[j, t, s] for j in model.U for t in model.T_2)

        # ------ q45 Tax Revenue ---------------------

        C_tax_storage_s = model.C_tax_storage[s]  # For scenario s, we get the value of the 2nd stage storage incentive
        # Pure storage revenue in the second stage for realization s
        pure_storage_rev_2s = C_tax_storage_s * sum(model.b_2[j, t, s] for j in model.R.difference(model.U) for t in model.T_2)
        # We add up all the CO_2 captured in the second stage

        C_tax_util_s = model.C_tax_util[s]
        # 45Q utilization revenue in the second stage for realization s
        Q45_utilization_rev_2s = C_tax_util_s * sum(model.b_2[j, t, s] for j in model.U for t in model.T_2)

        # Total Q45 incentive revenue for second stage in scenario s
        Q45_incentive_rev_2s = pure_storage_rev_2s + Q45_utilization_rev_2s

        model.second_stage_revenue[s] = util_rev_2s + Q45_incentive_rev_2s  # Total revenue in the second stage for scenario s
        #  ************** CALCULATE TOTAL PROFIT FOR SCENARIO s *************
        total_costs_2s = capture_costs_2s + pipeline_costs_2s + storage_costs_2s
        second_stage_profit_s = model.second_stage_revenue[s] - total_costs_2s
        model.second_stage_profit[s] = second_stage_profit_s

    # ---------- Final objective function Expression -----------

    # --------- Construct the piecewise approximation to the utility function ---------
    # Here we will use L = 10 lines to approximate our function. This can be changed by changing the value of L
    L = 0.5
    U = 2
    K = 16

    # Now we need to add the constraints for the objective. For each scenerio, we have that omega_s <= profit if s is realised
    for s in model.Scenarios:
        model.obj_constraints.add(model.normalized_scenario_profit[s] == (model.first_stage_profit + model.second_stage_profit[s]) / max_profit) # The normalized scenario profits
        d = (U - L) / (K - 1)
        for i in range(K):

            model.obj_constraints.add(model.omega[s] <= (((1 + model.normalized_scenario_profit[s]) - (L + d * i)) / (d * (1 - eta))) * ((L + d * (i + 1)) ** (1 - eta) - (L + d * i) ** (1 - eta) ) + ((L + d * i) ** (1 - eta)) / (1 - eta))

    # all the new constraints have been added so lets no write the final objective.
    # todo: This is scaled up. What should the scaling be?
    obj_expression = (10 ** 5) * sum(model.S_prob[s] * model.omega[s] for s in model.Scenarios)

    model.OBJ = pyo.Objective(expr=obj_expression, sense=pyo.maximize)



    # =========================================================================================================================
    # =========================================================================================================================
    # ================= End of modeling =======================================================================================
    # =========================================================================================================================
    # =========================================================================================================================
    # =========================================================================================================================
    # =========================================================================================================================




    print('running optimization model!')
    opt = pyo.SolverFactory('cplex')
    opt.options['mipgap'] = 0.001
    results = opt.solve(model)

    # =================================================================================================================
    # ============================= Write our Solutions to File =======================================================
    # =================================================================================================================

    print(results)
    print(f'The incentives in the first stage were {model.storage_incentive_1, model.util_incentive_1}.')
    print(f'Total first stage CO_2 captured: {int(sum(pyo.value(model.b_1[j, t]) for j in model.R for t in model.T_1) / (10 ** 6))} MT CO_2.')  # The total first stage CO_2 captured
    print()
    print(f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n'
          f'++++++++++++++++++++++++++++++++ Objective Function Price Breakdown ++++++++++++++++++++++++++++\n'
          f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    L = 0.5
    U = 2
    K = 16
    d = (U - L) / (K - 1)
    alt_obj = 0
    for s in model.Scenarios:
        print(f'The total normalized profit if scenario {s} occurs is: {1 + pyo.value(model.normalized_scenario_profit[s])}. This gives us a value of omega of {pyo.value(model.omega[s])} ')
        max_func = min(((1 + pyo.value(model.normalized_scenario_profit[s])) - (L + d * i)) / (d * (1 - eta)) * ((L + d * (i + 1)) ** (1 - eta) - (L + d * i) ** (1-eta)) + (L + d * i) / (1 - eta) for i in range(K))
        print(f'This apparently gives us a min omega value of {max_func}')
        alt_obj += pyo.value(model.S_prob[s]) * max_func

    print(f'The total objective value was {pyo.value(model.OBJ)}')
    print(f'The total objective if we use the "true" normalized profit is {alt_obj}')

    capture_costs = pyo.value(model.first_stage_capture_costs)
    storage_costs = pyo.value(model.first_stage_storage_costs)
    pipeline_costs = pyo.value(model.first_stage_pipeline_costs)
    total_costs = capture_costs + storage_costs + pipeline_costs
    print(f'Total Capture costs: {capture_costs:,.0f}')
    print(f'Total pipeline costs: {pipeline_costs:,.0f}')
    print(f'Total Storage costs: {storage_costs:,.0f}')
    profit = format_float_scientific(pyo.value(model.first_stage_profit), precision=2)
    print(f'Total First Stage Profit : {profit} \n')
    print(profit)


    if total_costs != 0:
        labels = 'Capture ', 'Storage', 'Transportation'

        sizes = [capture_costs / total_costs, storage_costs / total_costs, pipeline_costs / total_costs]

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        plt.show()

    print(f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n'
          f'+++++++++++++++++++++++++ Pipeline Construction Breakdown ++++++++++++++++++++++++++++++++++++++\n'
          f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for t in model.T_1:
        print()
        for d in model.D:
            for (i, j) in model.E:
                if pyo.value(model.gamma_1[i, j, d, t]) > 0:
                    print(f'In time period {t} the pipeline between {i} <-> {j} of diameter {d} is built.')
    print()
    print(f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n'
          f'+++++++++++++++++++++++++ CO_2 Movement Breakdown ++++++++++++++++++++++++++++++++++++++++++++++\n'
          f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n ')
    for t in model.T_1:
        print(f'-------------------------  In time period: t = {t} ----------------------------- ')
        for i, j in model.E:
            cur_flow = int(pyo.value(model.x_1[i, j, t]))
            if cur_flow > 0:
                print(f'We send {cur_flow / 10 ** 6} units {i} --> {j}.')
            elif cur_flow < 0:
                print(f'We send {- cur_flow / 10 ** 6} units {j} --> {i}.')

    print()
    print(f'------------------ Source Capture Rates ----------------------------\n')
    for t in model.T_1:
        for i in model.S:
            cur_capture = pyo.value(model.a_1[i, t])
            if cur_capture > 0:
                print(f'Source {i} produces {int(cur_capture / 10 ** 6)} units in period {t}')

    print()
    print(f'------------------ Geological Storage Capture Rates ------------------------------\n')
    for t in model.T_1:
        for j in model.R:
            cur_storage = pyo.value(model.b_1[j, t])
            if j not in model.U and cur_storage != 0:
                print(f'Sink {j} captures {int(cur_storage / 10 ** 6)} units in period {t}')
    print()
    print(f'------------------Utilization Capture Rates ------------------------------\n')
    for t in model.T_1:
        for j in model.U:
            cur_storage = pyo.value(model.b_1[j, t])
            if cur_storage != 0:
                print(
                    f'Utilization site {j} captures {int(cur_storage / 10 ** 6)} MT of CO2 in period {t}')
    print()
    print(f'------------------Total Capture Information ------------------------------\n')
    for j in model.R:
        total_captured_j = int(sum(pyo.value(model.b_1[j, t]) for t in model.T_1))
        if total_captured_j > 1:
            print(
                f'Sink {j} captures {int(total_captured_j / 10 ** 6)} MT CO_2. Which is {100 * (total_captured_j / model.Q_res[j]):.3f}% of its storage capacity')

    print(f'--------------- Carbon Goals--------------------------\n')
    total_captured_1 = sum(pyo.value(model.b_1[j, t]) for j in model.R for t in model.T_1) / (10 ** 6)
    print(f'The infrastructure captured {total_captured_1:,.0f} Megatons of CO_2 over the {num_periods_in_first_stage} first-stage periods.')
    expected_co2_captured_2 = sum(model.S_prob[s] * sum(pyo.value(model.b_2[j, t, s]) for j in model.R for t in model.T_2) for s in model.Scenarios) / (10 ** 6)
    print(f'The expected CO2  captured in the second-stage is {expected_co2_captured_2:,.0f} Megatons of CO_2.')
    print(f'The risk averse parameter is given by eta = {eta}\n')

    print(f'--- The total runtime was  {time.time() - start_time:.2f} seconds ---" ')

    total_CO_2_by_scenario = {
        s: (total_captured_1 + sum(pyo.value(model.b_2[j, t, s]) for j in model.R for t in model.T_2)) / (10 ** 6) for s
        in model.Scenarios}
    total_expected_co2 = sum(model.S_prob[s] * total_CO_2_by_scenario[s] for s in model.Scenarios) + total_captured_1

    standard_dev = compute_std(model.C_tax_storage, model.S_prob)

    # print(
    #     '=========================== STAGE 2 ANALYSIS ============================================================================================ ')
    # print()
    # for s in model.Scenarios:
    #     print(
    #         f'++++++++++++++++++++++ Analysis for realization {s}: (geo: {model.C_tax_storage[s]}, util: {model.C_tax_util[s]}) +++++++++++++++++++++++++')
    #     print()
    #     print(f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n'
    #           f'+++++++++++++++++++++++++ Construction Breakdown for scenario {s} ++++++++++++++++++++++++++++++\n'
    #           f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    #     for t in model.T_2:
    #         print()
    #         for d in model.D:
    #             for (i, j) in model.E:
    #                 if pyo.value(model.gamma_2[i, j, d, t, s]) > 0:
    #                     print(f'In time period {t} the pipeline between {i} <-> {j} of diameter {d} is built.')
    #
    #     for t in model.T_2:
    #         print()
    #         for i in model.S:
    #             if pyo.value(model.sigma_2[i, t, s]) > 0:
    #                 print(f'In time period {t}, source {i} is built.')
    #
    #     for t in model.T_2:
    #         print()
    #         for j in model.R:
    #             if pyo.value(model.rho_2[j, t, s]) > 0:
    #                 print(f'In time period {t}, sink {j} is built.')
    #     # ----------  Capture costs --------------
    #     capture_costs_2s = sum(
    #         model.B_src[i] * pyo.value(model.sigma_2[i, t, s]) + model.OM_src[i] * pyo.value(model.a_2[i, t, s]) / model.Q_source_rate[i] + model.V_src[i] * pyo.value(model.a_2[i, t, s]) for t in model.T_2 for i in model.S)
    #     print(f'The capture cost for scenario {s} is {capture_costs_2s}')
    #     # ---------- Pipeline costs --------------
    #
    #     pipeline_use_costs_2s = sum(
    #         model.V_pipe[i, j] * pyo.value(model.z_2[i, j, t, s]) for i, j in model.E for t in model.T_2)
    #     print(f'The pipeline use cost for scenario {s} is {pipeline_use_costs_2s}')
    #
    #     pipeline_build_costs_2s = sum((model.B_pipe[i, j, d] * pyo.value(model.gamma_2[i, j, d, t, s]) +
    #                                    model.OM_pipe[i, j, d] * pyo.value(model.z_2[i, j, t, s]) / model.Q_pipe_max[d])
    #                                          for i, j in model.E for d in model.D for t in model.T_2)
    #     print(f'The pipeline build cost for scenario {s} is {pipeline_build_costs_2s}')
    #
    #     # -------- Storage Costs ------------------
    #
    #     storage_costs_2s = sum(
    #         (model.B_res[j] * pyo.value(model.rho_2[j, t, s]) + model.OM_res[j] * pyo.value(model.b_2[j, t, s]) / model.Q_res_rate[j] +
    #          model.V_res[j] * pyo.value(model.b_2[j, t, s]))
    #         for t in model.T_2 for j in model.R)
    #     print(f'The storage cost for scenario {s} is {storage_costs_2s}. \n')
    #
    #     total_cost = capture_costs_2s + pipeline_use_costs_2s + pipeline_build_costs_2s + storage_costs_2s
    #     print()
    #     C_tax_storage_s = model.C_tax_storage[s]
    #     print(f'The incentive was {C_tax_storage_s}\n')
    #     revenue = C_tax_storage_s * sum(
    #         pyo.value(model.b_2[j, t, s]) for j in model.R.difference(model.U) for
    #         t in model.T_2)
    #     print(f'We earn {revenue}. So the total profit if {s} were to occur is {revenue - total_cost}')
    #
    #     print(
    #         ' ++++++++++++++++++++++++ Dynamics ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    #     for t in model.T_2:
    #         print(f'-------------------------  In time period: t = {t} ----------------------------- ')
    #         for i, j in model.E:
    #             if pyo.value(model.x_2[i, j, t, s]) > 1:
    #                 print(f'We send {int(pyo.value(model.x_2[i, j, t, s]) / 10 ** 6)} units {i} --> {j}.')
    #                 if int(pyo.value(model.x_2[i, j, t, s]) / 10 ** 6) == 0:
    #                     print(f'this is really {pyo.value(model.x_2[i, j, t, s])} tones of co2')
    #             elif pyo.value(model.x_2[i, j, t, s]) < -1:
    #                 print(f'We send {- int(pyo.value(model.x_2[i, j, t, s]) / 10 ** 6)} units {j} --> {i}.')
    #                 if int(pyo.value(model.x_2[i, j, t, s]) / 10 ** 6) == 0:
    #                     print(f'this is really {pyo.value(model.x_2[i, j, t, s])} tones of co2')
    #
    #     print()
    #     print(f'------------------ Source Capture Rates ----------------------------\n')
    #     for t in model.T_2:
    #         for i in model.S:
    #             if pyo.value(model.a_2[i, t, s]) > 0:
    #                 print(f'Source {i} produces {int(pyo.value(model.a_2[i, t, s]) / 10 ** 6)} units in period {t}')
    #
    #     print()
    #     print(f'------------------ Geological Storage Capture Rates ------------------------------\n')
    #     for t in model.T_2:
    #         for j in model.R:
    #             if j not in model.U and pyo.value(model.b_2[j, t, s]) != 0:
    #                 print(f'Sink {j} captures {int(pyo.value(model.b_2[j, t, s]) / 10 ** 6)} units in period {t}')
    #     print()
    #     print(f'------------------Utilization Capture Rates ------------------------------\n')
    #     for t in model.T_2:
    #         for j in model.U:
    #             if pyo.value(model.b_2[j, t, s]) != 0:
    #                 print(
    #                     f'Utilization site {j} captures {int(pyo.value(model.b_2[j, t, s]) / 10 ** 6)} MT of CO2 in period {t}')
    #     print()
    #     print(f'------------------Total Capture Information ------------------------------\n')
    #     for j in model.R:
    #         total_captured_j = sum(pyo.value(model.b_2[j, t, s]) for t in model.T_2)
    #         if total_captured_j > 1:
    #             print(
    #                 f'Sink {j} captures {total_captured_j / 10 ** 6} MT CO_2. Which is {100 * (total_captured_j / model.Q_res[j]):.3f}% of its storage capacity')
    #
    #     print(f'--------------- Carbon Goals--------------------------\n')
    #     total_captured = sum(pyo.value(model.b_2[j, t, s]) for j in model.R for t in model.T_2)
    #     print(f'Total captured in this realization is {int(total_captured):.1e}')
    #     print()
    # print(f'Total captured for the realization with eta = {eta}')
    #
    # for s in model.Scenarios:
    #     total_captured_s = sum(pyo.value(model.b_2[j, t, s]) for j in model.R for t in model.T_2)
    #     print(f'Total captured in realization {s} is {int(total_captured_s):.1e} ')

    # expected profit
    normalized_expected_profit = sum(model.S_prob[s] * pyo.value(model.normalized_scenario_profit[s]) for s in model.Scenarios)

    return profit, total_captured_1, total_expected_co2, capture_costs, pipeline_costs, storage_costs, standard_dev, normalized_expected_profit


if __name__ == '__main__':
    case_num = 7 # The redo file
    import sys

    max_profit = 2.24534 * 10 ** 10
    distribution_num = 'deterministic_low_first_stage'

    risk_values = [0]
    num_experiments = len(risk_values)
    update_slack(f'Experiments for distribution {distribution_num} have started.')


    cur = pd.DataFrame()
    for i, eta in enumerate(risk_values):
        text_results = f'Cases/Case{case_num}/Results/Dist{distribution_num}_eta{eta}.txt'
        sys.stdout = open(text_results, 'w')
        print(f'Experiment for eta = {eta} has completed! ')
        results = run_experiment(eta, case_num, max_profit)

        update_slack(f'Experiment for eta = {eta} has completed! {num_experiments - 1 - i} experiments remain.')
        data = {'Eta': eta, 'Profit': results[0], 'Total Stage one CO2 Captured': results[1],
                'Total Expected CO2 Captured': results[2], 'Stage One Capture Costs': results[3],
                'Stage One Transportation Costs': results[4], 'Stage One Storage_costs': results[5], 'Standard Deviation': results[6], 'normalized expected profit': results[7]}

        df = pd.DataFrame(data, index=[i])
        cur = pd.concat([cur, df])
        sys.stdout.close()
    #
    with pd.ExcelWriter(f'Cases/Case{case_num}/Results.xlsx', mode='a',
                        if_sheet_exists="replace") as writer:
        cur.to_excel(writer, sheet_name=f'Distribution{distribution_num}')

    # ==================================================================================================================
    # ============== Uncomment code below to plot out the results ======================================================
    # ==================================================================================================================
    # num_risk_vals = len(risk_values)
    # for experiment in range(num_risk_vals):
    #     plot_infrastructure_by_scenario(experiment, case_num, 'Results.xlsx', 5)
    #
    # for experiment in range(1, 6):
    #     plot_infrastructure_by_risk(case_num, 'Results.xlsx', experiment)
    #
    # plot_expected_total_CO2(case_num, 'Results.xlsx', 5)
    # plot_first_stage_CO2(case_num, 'Results.xlsx', 5)
    # plot_second_stage_CO2(case_num, 'Results.xlsx', 5)

    # plot_total_CO2(6,'Results.xlsx')
    # plot_infrastructure_by_scenario_Det(6, 'Results.xlsx')
    # scenarios = ['(\$0,\$0)', '(\$42.5, \$30)', '(\$85, \$60)', '(\$127.5, \$90)', '(\$170, \$120)']
    # probs = [0.03, 0.07, 0.8, 0.07,0.03]
    # plot_distribution(scenarios, probs, 'Spike')